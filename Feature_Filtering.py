import pandas as pd
import numpy as np
import pickle
import os
import glob
from collections import Counter
from LASSOCV_Boruta_perc import feature_preprocessing
import argparse
#PREREQUISITE: Input_dir must contain DataFrame files from both training and test set with this naming convention:
#target^timeperiod_trainLabel|testLabel.csv  e.g. HYPER_balanced^TP2_test.csv
#Target must be a feature in your target_file.
#Time can contain a label like in the above example or simply be an integer.
#There must be an integer in timeperiod as defined above.
#trainLabel is a string like train in above example that is common to all training set filenames
#testLabel is a string like test in above example that is common to all test set filenames
parser = argparse.ArgumentParser()
parser.add_argument('inputdir',help='path to input features')
parser.add_argument('output_dir', help='path to write output')
parser.add_argument('target_file',help='path to target file')
parser.add_argument('target_list', help='comma separated list of targets or ALL')
parser.add_argument('classification_or_regression', help='True for classification, False for regression')
parser.add_argument('lassoBorutaBoth', help='should be in [lasso,boruta,both]')
parser.add_argument('IDKey', help='primary key of DataFrame e.g. src_subject_id')
parser.add_argument('trainLabel', help='common label found in training set filenames')
parser.add_argument('testLabel', help='common label found in test set filenames')
parser.add_argument('lasso_threshold', help='lasso threshold')
parser.add_argument('perc_Boruta', help='perc parameter for Boruta algorithm')
args = parser.parse_args()
input_dir = args.inputdir
output_dir = args.output_dir
targetfile = args.target_file
target_list = args.target_list
c_or_r = str(args.classification_or_regression)
lassoBorutaBoth = args.lassoBorutaBoth
IDKey = args.IDKey
trainLabel = args.trainLabel
testLabel = args.testLabel
lasso_threshold = float(args.lasso_threshold)
perc_Boruta = int(args.perc_Boruta)
print('Running LBWrapper with parameters',args)
assert c_or_r in ['True', 'False']
c_or_r = c_or_r.lower() in ("yes", "true", "t", "1") #convert string to boolean

targetdf = pd.read_csv(targetfile)
#Make sure input_dir ends with separator for later use
if not input_dir.endswith(os.path.sep):
   input_dir += os.path.sep
#The reverse here since this is a root name we will add to   
if output_dir.endswith(os.path.sep):
   output_dir = output_dir[:-1]
output_dir = output_dir + '_' + lassoBorutaBoth + '_L' + str(lasso_threshold) + '_B' + str(perc_Boruta) + os.path.sep
output_importances_dir = output_dir.replace(lassoBorutaBoth, lassoBorutaBoth + '_Importances')
output_allTPs_dir = output_dir.replace(lassoBorutaBoth, lassoBorutaBoth + '_AllTPs')
if lassoBorutaBoth != 'None':
   os.makedirs(output_dir, exist_ok = True)
   os.makedirs(output_importances_dir, exist_ok = True)
os.makedirs(output_allTPs_dir, exist_ok = True)
#targets can be manually specified by editing target_list below and commenting out loop
if target_list == 'ALL':
   target_list = []
   #by default all targets will be run.
   for afile in os.listdir(input_dir):
      target = afile.split('^')[0]
      target_list.append(target)
   target_list = np.unique(target_list)
else:
   target_list = target_list.split(',')

#this dictionary keeps track of unique features across all time periods for each target.
#keys = target, values = unique features
features_across_time = {}
#initialize dictionary
for target in target_list:
   features_across_time[target] = []
   
for target in target_list:
 for afile in glob.glob(input_dir + target + '^*' + trainLabel + '*'): #os.listdir(input_dir):     
   df = pd.read_csv(afile) 
   tdf = targetdf[ [IDKey,target] ].copy() 
   if target == IDKey: #ensure IDKey is not used as a target
         continue
   print("Starting",afile,target,'processing')
   newframe = df.merge(tdf, how='inner', on=IDKey)
   print(newframe.shape, df.shape,tdf.shape)
   newtarget = newframe[ [IDKey,target] ].copy() #Order could have changed after merge
   newframe = newframe.drop(target,axis=1)
      
   #Check if IDs are in same order in DLFS and target
   DLFS_sks = newframe[IDKey]
   target_sks = newtarget[IDKey]
   #print(len(DLFS_sks), len(target_sks))
   for i in range(len(DLFS_sks)):
        if DLFS_sks[i] != target_sks[i]:
           print("mismatch found in row",i)
           exit(1)
   if 'level_0' in newframe.columns.tolist() or 'Unnamed:' in newframe.columns.tolist():
      print('Merge error. Exiting.')
      exit(1)
   #make sure no nulls in data   
   null_DLFS_entries = np.where(pd.isnull(newframe))
   null_target_entries = np.where(pd.isnull(newtarget[target]))
   if len(np.unique(null_target_entries[0])) > 0:
      print('Null target entries found:',tdf.iloc[np.unique(null_target_entries[0])], 'Exiting.' )
      exit(1)
   for q in range(len(null_DLFS_entries[0])):
      print('Null DLFS entries found:',null_DLFS_entries[0][q], null_DLFS_entries[1][q], 'Exiting.' )
      exit(1)
   
   if lassoBorutaBoth != 'None':
      DLFS_to_save = None
      DLFS_to_save, df_coef = feature_preprocessing(newframe, newtarget, IDKey, target, classification=c_or_r, 
                                                    lassoBorutaBoth=lassoBorutaBoth, lasso_threshold=lasso_threshold,
                                                   perc_Boruta=perc_Boruta) 
      #DLFS_to_save is the original DataFrame (newframe) but subsetted with only the features flagged as important
      if DLFS_to_save is not None:
         print(afile,'Before Lasso/Boruta, dataframe shape=',newframe.shape[1], 'afterwards=',DLFS_to_save.shape[1])
         DLFS_to_save.to_csv(output_dir + os.path.basename(afile), index=False)
         df_coef.to_csv(output_importances_dir + os.path.basename(afile).replace('.csv','_filtering_output.csv'), index=False)
      else:
         print('Lasso/Boruta returned None. Skipping',afile)
   
      features_across_time[target] = features_across_time[target] + df_coef['Feature'].tolist()
   else: #Use all features
      features_across_time[target] = features_across_time[target] + newframe.columns.tolist()
      
#Now to consolidate and get unique features across time per target to build final two-dimensional dataframes.
unique_features_across_time = {}
for k,v in features_across_time.items():
   unique_features_across_time[k] = np.append(np.unique(v), IDKey) #IDKey is metadata and is not in lasso results

#Now we do basically the same loop again. The difference is this time we have an important set of features
#that we want to retrieve across all time periods, even if they were not flagged as important in a time period.
for target in target_list:
   unique_features = unique_features_across_time[target]
   for dlfsfile in glob.glob(input_dir + target + '^*'): 
      time = dlfsfile.split('^')[1].split('_')[0]
      ttorrep = dlfsfile.split('^')[1].split('_')[1].split('.')[0]
      #we need to extract the time value here which must be an int. It is ok if there is an accompanying string label
      #e.g. TP3 but from this point forward we only want the number.
      time = int(''.join(filter(str.isdigit, time))) #We want to crash here if this is not an int
      dlfs = pd.read_csv(dlfsfile)
      cols_to_get = []
      #next 4 lines are to reduce dlfs to what is in unique_features
      #we can't just say dlfs = dlfs[unique_features] because some won't exist
      #e.g. a feature may not exist in a certain time period at all
      for feature in unique_features:
         if feature in dlfs.columns.tolist():
            cols_to_get.append(feature)
      dlfs = dlfs[cols_to_get] 
      print('processing',dlfsfile,dlfs.shape)         
      if dlfs is None or dlfs.shape[1] == 1: #only col might be IDKey
          print('ERROR - nothing to save', time, target, ttorrep)
          exit(1)
      else:
         print('saving time period',time,dlfs.shape)
         dlfs.to_csv(output_allTPs_dir + target + '^' + str(time) + '^' + ttorrep + '^combined_DLFS.csv',index=False)      


