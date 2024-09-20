import pandas as pd
import numpy as np
import copy
import sys
import argparse
import os
from collections import Counter
import pickle
import glob

parser = argparse.ArgumentParser()
parser.add_argument('input_dir', help='path to input files from previous step')
parser.add_argument('targetfileprefix',help='path to target file minus the train_test_label.csv')
parser.add_argument('targets', help='comma separated list of targets')
parser.add_argument('IDkey',help='DB Subject Identifier header')
parser.add_argument('tps',help='should look like 0,1,2,3')
parser.add_argument('trainLabel', help='common label found in training set filenames')
parser.add_argument('testLabel', help='common label found in test set filenames')
parser.add_argument('output_dir', help='output directory')

args = parser.parse_args()
input_dir = args.input_dir
if not input_dir.endswith('/'):
   input_dir += '/'
target_file_prefix = args.targetfileprefix
targets = args.targets.split(',')
IDkey = args.IDkey
time_periods = args.tps #comma separated list of TPs e.g. 2011,2012,2013 or 0,1,2,3
trainLabel = args.trainLabel
testLabel = args.testLabel
output_dir = args.output_dir
output_dir_3D = output_dir + '_' + time_periods.replace(',','-') + '_3D'
output_dir_2D = output_dir + '_' + time_periods.replace(',','-') + '_2D'
time_periods = time_periods.split(',')
print('Using input_dir',input_dir,'output_dir',output_dir)

for targ in targets:
   for tt_or_rep in [trainLabel, testLabel]:
      dfsWeWant = []
      timelabels = []
      local_output_dir_2D = output_dir_2D + os.path.sep + targ + os.path.sep
      local_output_dir_3D = output_dir_3D + os.path.sep + targ + os.path.sep
      os.makedirs(local_output_dir_2D, exist_ok=True) 
      os.makedirs(local_output_dir_3D, exist_ok=True)
      #Sorting must be done carefully here because of the '10' < '2' problem. This is why filename formatting
      #must be done in a precise manner in previous step and we can not use sorted()
      wanted_files = glob.glob(input_dir + targ + "^*^" + tt_or_rep + "^*.csv")
      #first we have to get all possible time values
      time_list = []
      for wf in wanted_files:
         time = int(wf.split('^')[1])
         if str(time) in time_periods: #you may not want to use all time periods when assembling matrix
            time_list.append(time)
      time_list = sorted(time_list) #Now we can use sorted because we have all integers   
      #Now we use this to manually iterate through files in sorted order
      wanted_files_final = []
      for i in time_list:
         for wf in wanted_files:
            temp_time = int(wf.split('^')[1])
            if temp_time == i:
               wanted_files_final.append(wf)
      
      for f in wanted_files_final: 
         df_to_add = pd.read_csv(f)        
         print('adding',f,'shape',df_to_add.shape)
         dfsWeWant.append(df_to_add)
         timelabels.append(f.split('^')[1])
      
      #Make this a runtime param?
      nanfill = 1
      targetdf = pd.read_csv(target_file_prefix + tt_or_rep + '.csv') 
      if targ not in targetdf.columns.tolist():
         print(targ,'not found. Exiting.')
         exit(1)

      for i in range(len(dfsWeWant)):
         if i == 0:
            subsInAll = dfsWeWant[0][[IDkey]]
         else:
            subsInAll = subsInAll.merge(dfsWeWant[i][[IDkey]], how='inner',on=IDkey)
      for i in range(len(dfsWeWant)):
         dfsWeWant[i] = subsInAll.merge(dfsWeWant[i], how='left', on=IDkey)
         #print("in subject merge loop",i,dfsWeWant[i].shape)
      for i in range(1,len(dfsWeWant)):
         otherDF = dfsWeWant[i]
         if dfsWeWant[0][[IDkey]].compare(otherDF[[IDkey]]).empty == False:
            print("subject mismatch. Exiting.")
            exit(1)
      y = dfsWeWant[0][[IDkey]].merge(targetdf, how='inner',on=IDkey)
      if y[IDkey].compare(dfsWeWant[0][IDkey]).empty == False:
         print('subject/target mismatch')
         exit(1)

      #BEGIN 2D matrix construction 
      final_2D_df = None
      for i in range(len(dfsWeWant)):
         print(i,timelabels[i],'shape',dfsWeWant[i])
         #rename cols with TP
         rename_dict = {}
         for col in dfsWeWant[i].columns.tolist():
            if col == IDkey: #not this one
               continue
            rename_dict[col] = col + '_TP' + str(i) #If doing one year only, '_TP3'SPECIAL CASE, UNDO!!!!

         if i == 0:
            final_2D_df = dfsWeWant[i].rename(columns=rename_dict)
         else:
            final_2D_df = final_2D_df.merge(dfsWeWant[i].rename(columns=rename_dict), how='left', on=IDkey)

      if not final_2D_df[IDkey].equals(y[IDkey]):
         print('ERROR AFTER MERGE. EXITING')
         exit(1) 
      
      y = y[targ]   
      y = y.to_numpy(dtype='int64')

      print('target length should match earliest row count',len(y))
      print("0s/1s count for",targ,len(np.where(y == 0)[0]), len(np.where(y == 1)[0]))
      #Save 2D matrices and information
      final_2D_df = final_2D_df.drop(IDkey, axis=1)
      unique_columns_2D = final_2D_df.columns.tolist()
      final_2D_df = final_2D_df.to_numpy()
      np.save(local_output_dir_2D + 'X_' + tt_or_rep.replace('rep','test') + '.npy', final_2D_df)
      np.save(local_output_dir_2D + 'y_' + tt_or_rep.replace('rep','test') + '.npy', y)
      uc_df_2D = pd.DataFrame(unique_columns_2D, columns=['Feature'])
      uc_df_2D.to_csv(local_output_dir_2D + 'unique_columns.csv', index=False)
      #for fun thjing I was meaning to check anyway
      for c in range(len(unique_columns_2D)):
         if unique_columns_2D[c] != uc_df_2D.at[c,'Feature']:
            print('I was afraid of this')      
            exit(1)
      #END 2D matrix construction
      #count columns by year and determine whether they are timeseries cols or not
      columnsByYear = {}
      #want list of unique cols that appear in all dfs
      all_cols_list = None
      for i in range(len(dfsWeWant)):
         columnsByYear[timelabels[i]] = dfsWeWant[i].columns.tolist()
         if i == 0:
            all_cols_list = columnsByYear[timelabels[i]]
         else:
            all_cols_list = all_cols_list + columnsByYear[timelabels[i]]

      #dict_by_count is for statistics gathering
      dict_by_count = {}
      for yy in range(len(dfsWeWant)):
         dict_by_count[yy + 1] = []  #We don't want zero-based here because all cols will appear at least once.      
      #all_cols_list has all columns in all time periods. So there will be duplicates for timeseries.
      countAllCols = Counter(all_cols_list)
      uniques = []
      timeseries = []
      for k,v in countAllCols.items():
         dict_by_count[v].append(k)
         if v == 0:
            print("ERROR, every variable should be appearing at least once or we got problems")
            exit(1)
         elif v == 1:
            uniques.append(k)
         else:
            timeseries.append(k)
      
      unique_columns = np.unique(all_cols_list) #NOTE this still has IDkey which we will want to remove at the very end.
      print("Hopefully uniques(one time only) + timeseries = num_unique_columns",  len(uniques), len(timeseries), len(unique_columns) )
      print('number of rows in baseline df',dfsWeWant[0].shape[0])
      #subject list sanity check
      for i in range(len(dfsWeWant)):
         if dfsWeWant[0][IDkey].equals(dfsWeWant[i][IDkey]) == False:
            print("IDkey mismatch in final_array assembly. Exiting.")
            exit(1) 

      #need to write out subject list here
      final_subject_list = dfsWeWant[0][[IDkey]]
      final_subject_list.to_csv(local_output_dir_2D + 'subject_list_' + tt_or_rep + '.csv', index=False)
      final_subject_list.to_csv(local_output_dir_3D + 'subject_list_' + tt_or_rep + '.csv', index=False)

      unique_columns = unique_columns.tolist()
      #Time for array conversion, IDkey must be dropped to not appear in final data
      unique_columns.remove(IDkey)
      num_unique_columns = len(unique_columns)
      runningColIndexes = np.zeros((num_unique_columns), dtype=int)
      coef_lookup = dict.fromkeys(unique_columns, [])
      print(tt_or_rep, targ,'Final_array shape',dfsWeWant[0].shape[0], len(dfsWeWant), num_unique_columns)
      X_complete = np.empty((dfsWeWant[0].shape[0], len(dfsWeWant), num_unique_columns))
      X_complete[:] = nanfill 
      for timep in range(len(dfsWeWant)):      
         for col in dfsWeWant[timep].columns:
            if col in [IDkey, 'eventname']:
               continue
            #need to get index of this in unique_columns
            bigindex = unique_columns.index(col)
            X_complete[:, timep, bigindex] = dfsWeWant[timep][col].to_numpy()
            runningColIndexes[bigindex] += 1
            coef_lookup[col].append(timelabels[timep])  

      #Below statistics show what percentage of matrix is real data
      all_matrix_vartps =  len(dfsWeWant) * (num_unique_columns + 1) #includes IDkey
      cells_filled = 0
      for k,v in dict_by_count.items():
         #print(k, len(v))
         cells_this_count = int(k) * len(v)
         #print('cells filled per tp',k,'
         cells_filled += cells_this_count
      matrix_fill_percent = cells_filled / all_matrix_vartps
      print(tt_or_rep, targ, 'overall matrix fill percentage', matrix_fill_percent)

      #All done, write out results.
      np.save(local_output_dir_3D + 'X_' + tt_or_rep + '.npy', X_complete) 
      np.save(local_output_dir_3D + 'y_' + tt_or_rep + '.npy', y) 

      if tt_or_rep == trainLabel:
         save_uniques = pd.DataFrame(unique_columns, columns=['Feature'])
         save_uniques.to_csv(local_output_dir_3D + 'unique_columns.csv')
         np.save(local_output_dir_3D + 'matrix_fill_percent.npy', matrix_fill_percent)

