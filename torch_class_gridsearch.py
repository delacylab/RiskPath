import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,average_precision_score,recall_score
from sklearn.metrics import f1_score, roc_curve, auc, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, KFold
from datetime import datetime
import multiprocessing
from multiprocessing import Pool,Process,Queue, current_process
import subprocess
import shap
import shutil
import copy
import sys
import argparse
import os
from collections import Counter
import time
import psutil
import pickle
import matplotlib
from matplotlib import pyplot as plt
import glob
import math

matplotlib.use('Agg')

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser()
parser.add_argument('input_dir',help='path to files created by Matrix_Assembly')
parser.add_argument('target',help='target to predict')
parser.add_argument('output_dir', help='directory for results')
parser.add_argument('numOfGPUs',type=int,help='Number of GPUs to use. -1 for CPU only.')
parser.add_argument('bidir', help="Bidirectional LSTM, True or False. STill need to say this even if no LSTM")
parser.add_argument('k', help='Number of cross-validation folds')
parser.add_argument('layer_size_min',help='Learning rate minimum')
parser.add_argument('layer_size_max',help='Learning rate maximum')
parser.add_argument('preset_feature_list', help= 'use hard coded feature list')
parser.add_argument('trainLabel', help='common label found in training set filenames')
parser.add_argument('testLabel', help='common label found in test set filenames')
#NN params
numLayers = 2
fitness_func = 'BIC'
epochs = 150
learning = 1e-5
#wd = 0.1
args = parser.parse_args()
prog_start_time = time.time()
input_dir = args.input_dir
targ = args.target
fname = args.target
basepath = args.output_dir
numOfGPUs = args.numOfGPUs
bidir = args.bidir
k = int(args.k)
layer_size_min = int(args.layer_size_min)
layer_size_max = int(args.layer_size_max)
dimension = input_dir[-2:]  #Last two characters of input_dir should be dimension of input matrix
#this could be determined by matrix shape but we would like it here for naming conventions.
preset_feature_list = args.preset_feature_list
trainLabel = args.trainLabel
testLabel = args.testLabel
print('running with command line args',args)
feature_col_index = 2 #3D matrix in shape rownum, tp, feature - is 0-based
if dimension == '2D':
   feature_col_index = 1

gpuString = 'GPU'
if numOfGPUs == -1:
   gpuString = 'CPU'

if not input_dir.endswith(os.path.sep):
   input_dir += os.path.sep
if not basepath.endswith(os.path.sep):
   basepath += os.path.sep   
os.makedirs(basepath, exist_ok=True) 

#This is the deep learning part of the script.
def parallel_func(iq, outq):
   for learning, wd, layer_size, partNum, gpuNum,procNum, k in iter(iq.get,'STOP'):
      #imports are done here because of a CUDA limitation that prevents
      #forked multiprocesses from getting another CUDA context.    
      import torch
      import torch.nn.functional as F
      from torch import nn
      from torch.utils.data import DataLoader
      from torchvision import datasets, transforms
      import torch.optim as optim
      from captum.attr import (GradientShap, DeepLift, DeepLiftShap, IntegratedGradients, LayerConductance,NeuronConductance,NoiseTunnel,)      
      num_classes=2
      if gpuNum == -1:
         device = torch.device('cpu')
      elif gpuNum != 99:
         cudaString = "cuda:" + str(gpuNum)
         device = torch.device("cuda:" + str(gpuNum))

      #Use same random seed for all processes
      np.random.seed(42)
      torch.manual_seed(42) 

      #print('CUDA properties', torch.cuda.get_device_properties(device))
      class EarlyStopping:
         def __init__(self, patience=1, min_delta=0):
           self.patience = patience
           self.min_delta = min_delta
           self.counter = 0
           self.min_validation_loss = float('inf')

         def early_stop(self, validation_loss):
           if validation_loss < self.min_validation_loss:
               self.min_validation_loss = validation_loss
               self.counter = 0
           elif validation_loss > (self.min_validation_loss + self.min_delta):
               self.counter += 1
               if self.counter >= self.patience:
                   return True
           return False
           
      #For 2D matrices
      class ANN(nn.Module):
       def init_xavier_weights(self):
            #Will probably be different here.
            for name, param in self.named_parameters():
               #print(name,param)
               if 'weight' in name:
                  torch.nn.init.xavier_uniform_(param.data)
               elif 'bias' in name:
                  param.data.fill_(0)   
                         
       def print_weights(self):
          for name, param in self.named_parameters():
             print(name,param)
             
       def __init__(self, num_layers, layer_size, num_features): #,device): 
         super().__init__()
         self.ann = nn.Sequential(nn.Linear(num_features, layer_size), nn.ReLU(), nn.Linear(layer_size, layer_size), nn.ReLU(), nn.Linear(layer_size, layer_size), nn.ReLU() )
         self.output_layer = nn.Linear(layer_size,num_classes) 
       
       def forward(self, x):
         output = self.ann(x)
         retme=self.output_layer(output)
         return retme            
             
      #LSTM for 3D matrices
      class MyLSTM(nn.Module):
       def init_xavier_weights(self):
            #init_start_time = time.time()
            for name, param in self.named_parameters():
               if 'weight_ih' in name:
                  torch.nn.init.xavier_uniform_(param.data)
               elif 'weight_hh' in name:
                  torch.nn.init.orthogonal_(param.data)
               elif 'bias' in name:
                  param.data.fill_(0)
            #print('init time',time.time() - init_start_time)
            
       def print_weights(self):
          for name, param in self.named_parameters():
             print(name,param)

       def __init__(self, bidir, num_layers, layer_size, num_features): #,device): 
         super().__init__()
         #batch_first=True means first param (row) is the one to use to separate samples in a batch.
         #Documentation says default activation function is tanh for LSTM.
         self.lstm = nn.LSTM(input_size=num_features, hidden_size=int(layer_size), num_layers=num_layers, batch_first=True, bidirectional=str2bool(bidir))
         layer_multiplier = 1
         if str2bool(bidir):
            layer_multiplier = 2
         self.output_layer = nn.Linear(layer_size * layer_multiplier,num_classes) #Note the output layer is 2 for bidir or 1 for not.
       
       def forward(self, x):
         lstm_out, (hidden,cell) = self.lstm(x) #So x at this point will have all (layer_size * w layers) hidden states

         if str2bool(bidir): 
            cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
         else:
            cat = lstm_out[:,-1,:]
         retme=self.output_layer(cat)
         
         return retme  

      #We are going to use all features.
      features = unique_columns  
      #We also want to record importances for features per time period.         
      features_TPs = []
      if dimension == '3D':
         for f in features:
            for t in range(X.shape[1]):
               features_TPs.append(f + '_TP' + str(t))
         
      #Use test dataset if available
      X_rep_tensor = None
      if str2bool(useRep):      
         X_rep_tensor = np.array([x for x in X_rep], dtype=np.float32)
         X_rep_tensor = torch.from_numpy(X_rep_tensor)
         #y_rep_torch = torch.from_numpy(y_rep_tensor)
               
      num_features = len(features)
      skf = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)
      
      fitness_scores = []
      feature_importances_scores = []        
      feature_importances_scores_withTPs = []      
      #Get the train statistics.
      train_fitness_scores = []
      train_accuracy_scores = []
      train_precision_scores = []
      train_recall_scores = []
      train_spec_scores = []
      train_f1_scores = []
      train_auc_scores = []
      train_loss_scores = []
      #Unseen data scores
      rep_accuracy_scores = []
      rep_precision_scores = []
      rep_recall_scores = []
      rep_spec_scores = []
      rep_f1_scores = []
      rep_auc_scores = []
      rep_loss_scores = []
      rep_display_scores = []
      rep_importances_scores = []
      #Validation scores
      accuracy_scores = []
      precision_scores = []
      recall_scores = []
      spec_scores = []
      f1_scores = []
      auc_scores = []
      delta_scores = []
      elapsed_epochs_list = []
      loss_scores = []
      model_statistics = []
      for train,test in skf.split(X, y): 
         X_train = np.take(X, train, axis=0)
         X_test = np.take(X, test, axis=0)
         y_train = np.eye(2, dtype='float32')[y[train]]
         y_test = np.eye(2, dtype='float32')[y[test]] 
         #Following lines are because pytorch requires its own datatypes
         X_train_tensor = np.array([x for x in X_train], dtype=np.float32)
         X_train_tensor = torch.from_numpy(X_train_tensor)  #without above line, type is float64
         X_test_tensor = np.array([x for x in X_test], dtype=np.float32)
         X_test_tensor = torch.from_numpy(X_test_tensor)  #without above line, type is float64         
         y_train = torch.from_numpy(y_train)
         y_test = torch.from_numpy(y_test)
         #Manually move to GPU 
         X_train_GPU = X_train_tensor.to(device)
         X_test_GPU = X_test_tensor.to(device)
         y_train_GPU = y_train.to(device)
         y_test_GPU = y_test.to(device)
         
         model = None
         if dimension == '3D':
            numLayers = 2
            model = MyLSTM(bidir, numLayers, layer_size, num_features) 
         else:
            numLayers = 3
            model = ANN(numLayers, layer_size, num_features) 
         model.to(device)
         # Weight init is a very computationally expensive operation (uses 10 cores by itself) 
         # so needs to be done on GPU
         model.init_xavier_weights()
         opt = torch.optim.AdamW(model.parameters(), lr = learning, weight_decay = wd) #, betas=(beta_1, beta_2))        
         loss = nn.CrossEntropyLoss()    
         early_stopper = EarlyStopping(patience=5, min_delta = float(10 ** -8) ) #min_delta=0) #That was default in what we used before   
         elapsed_epochs = 0
         earlyStopped = False
         for e in range(epochs):
            epoch_start_time = time.time()
            model.train()
            y_predict = model(X_train_GPU) #NOTE these need to be softmaxd if using for predictions.
            #Pytorch loss functions automatically softmax the predictions 
            training_loss = loss(y_predict,y_train_GPU)
            opt.zero_grad() 
            training_loss.backward()
            opt.step()
            model.eval() #Freeze training
            #print('epoch train time in seconds',time.time() - epoch_start_time,flush=True)
            elapsed_epochs += 1
            with torch.no_grad():
               test_pred = model(X_test_GPU) 
               val_loss = loss(test_pred, y_test_GPU)
            if early_stopper.early_stop(val_loss):
               #print('stopping training at epoch',e)
               break
         loss_scores.append(val_loss.detach().cpu())      
         elapsed_epochs_list.append(elapsed_epochs)
         y_predict = model(X_test_GPU).detach().cpu() #.numpy()
         #print('model logit predictions',y_predict)
         #GPU Diagnostic functions
         #numGpuBytes = sys.getsizeof(X_train_GPU.storage()) + sys.getsizeof(X_test_GPU.storage()) + sys.getsizeof(y_train_GPU.storage()) + sys.getsizeof(y_test_GPU.storage())
         #print('X tensor sizes',numGpuBytes,'in kfold loop',torch.cuda.memory_allocated(device=device)) 
         
         #Add softmax layer as final layer of network
         sm = nn.Softmax(dim = 1)
         y_predict = sm(y_predict).numpy()
         y_predict = np.argmax(y_predict, axis=-1)  
         
         num_model_params = 0
         for name, param in model.named_parameters():
            param_list_shape = list(param.shape)
            k_add = param_list_shape[0] if len(param_list_shape) == 1 else param_list_shape[0] * param_list_shape[1]
            num_model_params += k_add
         model_statistics.append(num_model_params)
         
         def BIC_pop_fitness(test, predict, sample_size, num_params,  **kwargs):
          if fitness_func == 'BIC':
            resid = test - predict
            sse = sum(resid**2)
            sample_size = len(X)
            num_params = num_features
            #Below handles the sse=0 case
            with np.errstate(divide='raise'):
               try:
                  fitness = (sample_size * np.log(sse/sample_size)) + (num_params * np.log(sample_size))  
               except FloatingPointError:
                  fitness = (sample_size * np.log( 0.1 / sample_size)) + (num_params * np.log(sample_size))
          else:
            fitness = roc_auc_score(test, predict)
            
          return fitness 

         fitness_scores.append(BIC_pop_fitness(y[test], y_predict, len(X), num_features))
         accuracy = accuracy_score(y[test], y_predict)
         accuracy_scores.append(accuracy)
         
         y_train_predict = model(X_train_GPU).detach().cpu()
         y_train_predict_sm = sm(y_train_predict)
         y_train_predict = sm(y_train_predict).numpy()
         y_train_predict = np.argmax(y_train_predict, axis=-1) 
         train_fitness_scores.append(BIC_pop_fitness(y[train], y_train_predict, len(X), num_features))
         train_accuracy_scores.append(accuracy_score(y[train], y_train_predict))
         train_precision_scores.append(average_precision_score(y[train], y_train_predict))
         train_recall_scores.append(recall_score(y[train],y_train_predict))
         train_spec_scores.append(recall_score(y[train], y_train_predict, pos_label=0))
         train_f1_scores.append(f1_score(y[train],y_train_predict, average='binary'))
         fpr, tpr, thresholds = roc_curve(y[train], y_train_predict, pos_label=1) 
         train_auc_scores.append(auc(fpr, tpr))     
         train_loss_scores.append(loss(y_train_predict_sm, y_train))   

         torch.backends.cudnn.enabled=False  #Do not use this if we're going to do shap on CPU 

         #START GRADIENTSHAP for feature importances
         def model_sm_wrapper(*args, **kwargs):
           return torch.nn.functional.softmax( model(*args, **kwargs), dim=1 )
         gs = GradientShap(model_sm_wrapper)
         attributions, delta_c = gs.attribute(X_test_GPU, X_train_GPU, target=1, return_convergence_delta=True)
         delta_c = delta_c.cpu().detach().numpy()
         delta_c = np.mean(delta_c, axis=0) 
         delta_scores.append(delta_c)
         attributions = attributions.cpu().detach().numpy()
         attributions = np.abs(attributions)
         feature_importances = np.mean(attributions, axis=0) #Now shape is (num_TPs, numFeatures) in 3D, should just be numFeatures in 2D
         if dimension == '3D':
            feature_importances_scores_withTPs.append(feature_importances.copy())
            feature_importances = np.mean(feature_importances, axis=0) #Now we're down to numFeatures
            feature_importances_scores.append(feature_importances) 
         else:
            feature_importances_scores.append(feature_importances) 
         #END GRADIENTSHAP 
         #validation stats for validation part of kfold         
         precision_scores.append(average_precision_score(y[test], y_predict))
         recall_scores.append(recall_score(y[test],y_predict))
         spec_scores.append(recall_score(y[test], y_predict, pos_label=0))
         f1_scores.append(f1_score(y[test],y_predict, average='binary'))
         fpr, tpr, thresholds = roc_curve(y[test], y_predict, pos_label=1) #pos_label=1 because our y is in [0,1]
         auc_scores.append(auc(fpr, tpr))
         
         if str2bool(useRep):      
            #Test performance on unseen data
            model_back_to_cpu = model.to('cpu')
            y_rep_predict = model_back_to_cpu(X_rep_tensor).detach()
            y_rep_predict_sm = sm(y_rep_predict)
            y_rep_predict = sm(y_rep_predict).numpy()
            y_rep_predict = np.argmax(y_rep_predict, axis=-1)          
         
            rep_accuracy_scores.append(accuracy_score(y_rep, y_rep_predict))
            rep_precision_scores.append(average_precision_score(y_rep, y_rep_predict))
            rep_recall_scores.append(recall_score(y_rep, y_rep_predict))
            rep_spec_scores.append(recall_score(y_rep, y_rep_predict, pos_label=0))
            rep_f1_scores.append(f1_score(y_rep, y_rep_predict, average='binary'))
            fpr, tpr, thresholds = roc_curve(y_rep, y_rep_predict, pos_label=1) #pos_label=1 because our y is in [0,1]
            rep_area_under_curve = auc(fpr, tpr)
            rep_auc_scores.append(rep_area_under_curve) 
            rep_display_scores.append( RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=rep_area_under_curve, estimator_name='Unit_size=' + str(layer_size))  )

            def model_cpu_sm_wrapper(*args, **kwargs):
              return torch.nn.functional.softmax( model_back_to_cpu(*args, **kwargs), dim=1 )
            gs_rep = GradientShap(model_cpu_sm_wrapper)
            attributions_rep, rep_delta = gs_rep.attribute(X_rep_tensor, X_train_tensor, target=1, return_convergence_delta=True)
            rep_importances_scores.append(attributions_rep.detach().numpy() ) #.cpu().detach().numpy() )     
            #print(type(y_rep_predict_sm), type(y_rep))
            rep_loss_scores.append(loss(y_rep_predict_sm,torch.from_numpy(y_rep)))
      #END OF KFOLD LOOP
      #best rep (aka test unseen data) comes from model that did best in val stage based on AUC score
      #np.flip for sort descending since we want the best
      best_model_index = np.flip(np.argsort(auc_scores))[0] #(loss_scores)[0]
      if str2bool(useRep):
         best_rep_loss = rep_loss_scores[best_model_index].numpy()
         best_rep_accuracy = rep_accuracy_scores[best_model_index]
         best_rep_precision = rep_precision_scores[best_model_index]
         best_rep_recall = rep_recall_scores[best_model_index]
         best_rep_spec = rep_spec_scores[best_model_index]
         best_rep_f1 = rep_f1_scores[best_model_index]
         best_rep_auc = rep_auc_scores[best_model_index]      
         best_rep_display =  rep_display_scores[best_model_index]
         best_rep_importances = rep_importances_scores[best_model_index]
      else:
         best_rep_loss = None;best_rep_accuracy = None;
         best_rep_precision = None; best_rep_recall = None
         best_rep_spec = None; best_rep_f1 = None
         best_rep_auc = None; best_rep_display = None
         best_rep_importances = None 

      #Note the index is the same because it's best model on validation data
      best_accuracy_score = accuracy_scores[best_model_index]
      best_precision_score = precision_scores[best_model_index]
      best_recall_score = recall_scores[best_model_index]
      best_spec_score = spec_scores[best_model_index]
      best_f1_score = f1_scores[best_model_index]
      best_auc_score = auc_scores[best_model_index]
      best_loss = loss_scores[best_model_index].numpy()
     
      best_train_model_index = np.flip(np.argsort(train_auc_scores))[0]
      best_train_accuracy = train_accuracy_scores[best_train_model_index]
      best_train_precision = train_precision_scores[best_train_model_index]
      best_train_recall = train_recall_scores[best_train_model_index]
      best_train_spec = train_spec_scores[best_train_model_index]
      best_train_f1 = train_f1_scores[best_train_model_index]
      best_train_auc = train_auc_scores[best_train_model_index]  
      best_train_loss = train_loss_scores[best_train_model_index].numpy()
     
      mean_accuracy = np.mean(accuracy_scores) 
      mean_model_statistics = np.mean(model_statistics)
      mean_loss = np.mean(loss_scores)
      mean_fitness = np.mean(fitness_scores)
      train_mean_accuracy = np.mean(train_accuracy_scores)
      train_mean_fitness = np.mean(train_fitness_scores)
      mean_elapsed_epochs = np.mean(elapsed_epochs_list)
      mean_tps = np.array([np.nan,np.nan,np.nan,np.nan,np.nan])
      feature_importances_scores_folds=pd.DataFrame(data=feature_importances_scores) 
      feature_importances_scores_mean = []
      for column in feature_importances_scores_folds:
         col_mean = feature_importances_scores_folds[column].mean()
         feature_importances_scores_mean.append(col_mean)      
      features_TPs_flatten = []
      countercheck = 0
      if dimension == '3D':
         feature_importances_scores_withTPs_mean = np.mean(feature_importances_scores_withTPs, axis=0)
         for f in range(len(features)):
            for t in range(X.shape[1]):
               #print('Feature,value',features[f], feature_importances_scores_withTPs_mean[t, f ])
               if features[f] not in features_TPs[countercheck]:
                  print('feature name mismatch')
               features_TPs_flatten.append(feature_importances_scores_withTPs_mean[t, f ])
               countercheck += 1
      mean_precision = np.mean(precision_scores)
      mean_recall = np.mean(recall_scores)
      mean_spec = np.mean(spec_scores)
      mean_f1 = np.mean(f1_scores)
      mean_auc = np.mean(auc_scores)
      mean_delta = np.mean(delta_scores) 
      train_mean_precision = np.mean(train_precision_scores)
      train_mean_recall = np.mean(train_recall_scores)
      train_mean_spec = np.mean(train_spec_scores)
      train_mean_f1 = np.mean(train_f1_scores)
      train_mean_auc = np.mean(train_auc_scores)  
      train_mean_loss = np.mean(train_loss_scores)      
      
      outq.put( (features, feature_importances_scores_mean, features_TPs, features_TPs_flatten, mean_fitness, num_features, mean_accuracy, mean_precision, mean_recall, mean_spec, mean_f1, mean_auc, learning, wd, layer_size, mean_delta, train_mean_fitness, train_mean_accuracy, train_mean_precision, train_mean_recall, train_mean_spec, train_mean_f1, train_mean_auc, train_mean_loss, best_train_accuracy, best_train_precision, best_train_recall, best_train_spec, best_train_f1, best_train_auc, best_train_loss, mean_elapsed_epochs, mean_loss, best_rep_accuracy, best_rep_precision, best_rep_recall, best_rep_spec, best_rep_f1, best_rep_auc, best_rep_loss, best_rep_display, best_rep_importances, mean_model_statistics, best_accuracy_score, best_precision_score, best_recall_score, best_spec_score, best_f1_score, best_auc_score, best_loss, X_rep_tensor, procNum) )

unique_columns=None
try:       
   with open(input_dir + targ + '/unique_columns.pkl', 'rb') as handle:
      unique_columns = pickle.load(handle)
except FileNotFoundError:   
   unique_columns = pd.read_csv(input_dir + targ + '/unique_columns.csv')['Feature'].tolist()
   
num_unique_columns = len(unique_columns)
X = None; y = None;
try:
   X = np.load(input_dir + targ + '/X_train.npy')
   y = np.load(input_dir + targ + '/y_train.npy')
except FileNotFoundError:
   X = np.load(input_dir + targ + '/X_tt.npy')
   y = np.load(input_dir + targ + '/y_tt.npy')
useRep = "True"
X_rep=None;y_rep=None;y_rep_tensor=None
try:
   X_rep = np.load(input_dir + targ + '/X_test.npy')
   y_rep = np.load(input_dir + targ + '/y_test.npy')
   y_rep_tensor = np.eye(2, dtype='float32')[y_rep]
except FileNotFoundError:
   useRep = "False"
print('useRep=',useRep)

#BEGIN for new section - we have to go through lasso importances for all TPs and get top features from each, then get biggest importances from that.
use_this_feature_list = ''

if str2bool(preset_feature_list): #preset feature list defined below if desired
      copylist = copy.deepcopy(unique_columns)
      #2 potential situations. We have a hard coded list to use OR
      #we want to remove a few from original list. Below is an example of the latter
      use_this_feature_list = []
      for cc in copylist: #argh still dealing with bpm
         if 'cbcl_scr_syn_' not in cc and 'bpm_y_scr' not in cc:
            use_this_feature_list.append(cc)
      
      #below is an example of the former
      #use_this_feature_list = ['kbi_y_drop_in_grades', 'kbi_p_conflict_l_1_Very well Mu', 'cbcl_scr_syn_rulebreak_t', 'kbi_y_det_susp', 'neighborhood_crime_y_5_Strongly Agr', 'demo_comb_income_v2_wealthy_l', 'kbi_p_conflict_l_2_Some conflic', 'demo_prnt_marital_v2_l_1_Married Casa', 'cbcl_scr_syn_thought_t', 'demo_yrs_1_l_0_Never Nunca', 'sai_p_read', 'mature_content', 'kbi_p_c_bully', 'cbcl_scr_syn_attention_t', 'fes_y_ss_fc_pr', 'last_year_adverse']
      print('Using hard coded list instead')
      
if len(use_this_feature_list) > 0:
   print('USING FEATURES',use_this_feature_list)
   top_indexes = []; top_features = [] #I will do this manually because I'm not sure what unique will do to the feature order and it matters
   for f in use_this_feature_list:
      indx_of_feature = unique_columns.index(f)
      if indx_of_feature not in top_indexes:  #it could have been added already since there will be multiple rows for same feature.
         top_indexes.append(indx_of_feature)
         top_features.append(f)

   print('top_features',top_features)
   print('top_indexes', top_indexes)
   unique_columns = top_features
   num_unique_columns = len(unique_columns)

   X = np.take(X, top_indexes, axis=feature_col_index) 
   if str2bool(useRep):
      X_rep = np.take(X_rep, top_indexes, axis=feature_col_index) 
#END OF TOP FEATURES SECTION

#Support running on CPU or GPU
QUEUE_SIZE = -1
if numOfGPUs > 0: 
   #Don't want to run too many simultaneous processes on GPU due to memory constraints
   #Future work - automate determining of coresPerGPU by determining how memory arrays will take on GPU 
   #gpuInfo = str(subprocess.check_output(["nvidia-smi"]))
   #gpuMem = int(gpuInfo.split('MiB')[1].split('/')[1]) # * 1024 * 1024 # KB/Bytes conversion
   #coresPerGPU = math.floor( int(os.getenv("SLURM_NTASKS")) / (numOfGPUs * int(os.getenv("OMP_NUM_THREADS"))))
   coresPerGPU = 6 #I hope even this isn't too much

   #if gpuMem < 26000 or (numLayers == 3 and str2bool(bidir)):
   #   coresPerGPU = int(coresPerGPU / 2)
   QUEUE_SIZE = coresPerGPU * numOfGPUs 
   print('running with QUEUE_SIZE', QUEUE_SIZE,numOfGPUs,'GPUs,',coresPerGPU,'procs per GPU',os.getenv("OMP_NUM_THREADS"),'cores per proc')
else:
   QUEUE_SIZE =  math.floor( int(os.getenv("SLURM_NTASKS")) / (int(os.getenv("OMP_NUM_THREADS"))))
   print('CPU only mode, running with QUEUE_SIZE', QUEUE_SIZE)
         
if __name__ == '__main__':
 np.random.seed(42)
 inputQueue = Queue();outputQueue = Queue()
 lock = multiprocessing.Lock()
 print("Initialize process queue")
 for i in range(QUEUE_SIZE): #Number of sub procs to run
    Process(target=parallel_func,args=(inputQueue,outputQueue)).start()
 print("Process queue initialized",flush=True)  
 
 layersize_pop_first = [x for x in range(layer_size_min,99,5)]
 layersize_pop = [x for x in range(100,layer_size_max + 1,20)]
 layersize_pop = layersize_pop_first + layersize_pop
 #construct and zip arrays for grid search
 wd_pop = [0.1]  #, 0.01]
 wd_zip_pop = []
 for wd in wd_pop:
   wd_zip_pop = wd_zip_pop + [wd for q in range(len(layersize_pop))] 

 for a in range(len(wd_pop) - 1): 
   layersize_pop = layersize_pop + layersize_pop #For zipping we need full set of elements for every wd 
 print('layersize_pop',layersize_pop)
 if len(wd_zip_pop) != len(layersize_pop):
    print('error constructing zip variables. Exiting.')
    exit(1)
    
 feature_list = []
 feature_importances_list = []
 featureTP_list = []
 feature_importances_TP_list = []
 rep_roc_curve_list = []
 rep_importances_list = []
 rep_tensor_list = []
 newrows = [] 
 returned_layer_list = []
 overallCounter = 0
 receiveCounter = 0
 receiveTimeList = []
 counter = 0
 gpuNum = 0
 #k = int(X.shape[0] / 50)
 print('queue size is',QUEUE_SIZE)
 columns = ['learning', 'weight_decay', 'unitsize', 'elapsed_epochs', 'num_features', 'model_parameters', trainLabel + '_fitness_avg', trainLabel + '_accuracy_avg', trainLabel + '_precision_avg', trainLabel + '_recall_avg', trainLabel + '_spec_avg', trainLabel + '_f1_avg', trainLabel + '_auc_avg', trainLabel + '_loss_avg', 'best_' + trainLabel +  '_accuracy', 'best_' + trainLabel + '_precision', 'best_' + trainLabel + '_recall', 'best_' + trainLabel + '_spec', 'best_' + trainLabel + '_f1', 'best_' + trainLabel + '_auc', 'best_' + trainLabel + '_loss',  'validation_fitness_avg', 'validation_accuracy_avg', 'validation_precision_avg', 'validation_recall_avg', 'validation_spec_avg', 'validation_f1_avg', 'validation_auc_avg', 'validation_loss_avg', 'best_validation_accuracy', 'best_validation_precision', 'best_validation_recall', 'best_validation_spec', 'best_validation_f1', 'best_validation_auc', 'best_validation_loss', 'best_' + testLabel + '_accuracy', 'best_' + testLabel + '_precision', 'best_' + testLabel + '_recall', 'best_' + testLabel + '_spec', 'best_' + testLabel + '_f1', 'best_' + testLabel + '_auc', 'best_' + testLabel + '_loss', 'Shap_Delta', 'procNum' ] 
 for wd, layer_size_loop in zip(wd_zip_pop, layersize_pop):
       if counter < QUEUE_SIZE:
         if overallCounter < QUEUE_SIZE:
            gpuNum = (int  (  (int(overallCounter) / int(QUEUE_SIZE / numOfGPUs)  ) ))
         else:
            gpuNum = 99
         if numOfGPUs == -1:
            gpuNum = -1            
         #print("sending count",counter,flush=True)
         #print('current layer_size',layer_size)
         print('sending',learning,wd, layer_size_loop, 1, gpuNum, counter, k)
         inputQueue.put( (learning,wd, layer_size_loop, 1, gpuNum, counter, k) )
         counter += 1
         overallCounter += 1
       else: #Collect results
         #print("starting receive loop",counter,flush=True)
         for i in range(QUEUE_SIZE):
           print('RECEIVING',receiveCounter)
           receiveCounter += 1
           (features, feature_importances_scores_mean, features_TPs, features_TPs_flatten, mean_fitness, num_features, mean_accuracy, mean_precision, mean_recall, mean_spec, mean_f1, mean_auc, learning, wd, layer_size, mean_delta, train_mean_fitness, train_mean_accuracy, train_mean_precision, train_mean_recall, train_mean_spec, train_mean_f1, train_mean_auc, train_mean_loss, best_train_accuracy, best_train_precision, best_train_recall, best_train_spec, best_train_f1, best_train_auc, best_train_loss, mean_elapsed_epochs, mean_loss, best_rep_accuracy, best_rep_precision, best_rep_recall, best_rep_spec, best_rep_f1, best_rep_auc, best_rep_loss, best_rep_display, best_rep_importances, mean_model_statistics, best_accuracy_score, best_precision_score, best_recall_score, best_spec_score, best_f1_score, best_auc_score, best_loss, X_rep_tensor, procNum) = outputQueue.get()
           newrows.append([learning, wd, layer_size, mean_elapsed_epochs, num_features, mean_model_statistics, train_mean_fitness, train_mean_accuracy, train_mean_precision, train_mean_recall, train_mean_spec, train_mean_f1, train_mean_auc, train_mean_loss, best_train_accuracy, best_train_precision, best_train_recall, best_train_spec, best_train_f1, best_train_auc, best_train_loss, mean_fitness, mean_accuracy, mean_precision, mean_recall, mean_spec, mean_f1, mean_auc, mean_loss, best_accuracy_score, best_precision_score, best_recall_score, best_spec_score, best_f1_score, best_auc_score, best_loss, best_rep_accuracy, best_rep_precision, best_rep_recall, best_rep_spec, best_rep_f1, best_rep_auc, best_rep_loss, mean_delta, procNum])
           feature_list.append(features)
           feature_importances_list.append(feature_importances_scores_mean)           
           featureTP_list.append(features_TPs)
           feature_importances_TP_list.append(features_TPs_flatten)  
           rep_roc_curve_list.append(best_rep_display)      
           rep_importances_list.append(best_rep_importances)
           rep_tensor_list.append(X_rep_tensor)
           returned_layer_list.append(layer_size)
         #Start next job running after collecting all results
         if numOfGPUs == -1:
            gpuNum = -1  
         else:
            gpuNum = 99         
         #Send next process
         counter = 0
         print('extra sending',learning,wd, layer_size_loop, 1, gpuNum, counter, k)
         inputQueue.put( (learning,wd, layer_size_loop, 1, gpuNum, counter, k) )   
         counter = 1
         #overallCounter's job is done here, don't need to do anything with it here 
       #have to collect stragglers for when workload not evenly divisible by QUEUE_SIZE
       #print("Collecting stragglers: ", counter)
 for i in range(counter):
           #print('RECEIVING straggler',receiveCounter)
           receiveCounter += 1
           (features, feature_importances_scores_mean, features_TPs, features_TPs_flatten, mean_fitness, num_features, mean_accuracy, mean_precision, mean_recall, mean_spec, mean_f1, mean_auc, learning, wd, layer_size, mean_delta, train_mean_fitness, train_mean_accuracy, train_mean_precision, train_mean_recall, train_mean_spec, train_mean_f1, train_mean_auc, train_mean_loss, best_train_accuracy, best_train_precision, best_train_recall, best_train_spec, best_train_f1, best_train_auc, best_train_loss, mean_elapsed_epochs, mean_loss, best_rep_accuracy, best_rep_precision, best_rep_recall, best_rep_spec, best_rep_f1, best_rep_auc, best_rep_loss, best_rep_display, best_rep_importances, mean_model_statistics, best_accuracy_score, best_precision_score, best_recall_score, best_spec_score, best_f1_score, best_auc_score, best_loss, X_rep_tensor, procNum) = outputQueue.get()
           newrows.append([learning, wd, layer_size, mean_elapsed_epochs, num_features, mean_model_statistics, train_mean_fitness, train_mean_accuracy, train_mean_precision, train_mean_recall, train_mean_spec, train_mean_f1, train_mean_auc, train_mean_loss, best_train_accuracy, best_train_precision, best_train_recall, best_train_spec, best_train_f1, best_train_auc, best_train_loss, mean_fitness, mean_accuracy, mean_precision, mean_recall, mean_spec, mean_f1, mean_auc, mean_loss, best_accuracy_score, best_precision_score, best_recall_score, best_spec_score, best_f1_score, best_auc_score, best_loss, best_rep_accuracy, best_rep_precision, best_rep_recall, best_rep_spec, best_rep_f1, best_rep_auc, best_rep_loss, mean_delta, procNum])
           feature_list.append(features)
           feature_importances_list.append(feature_importances_scores_mean)
           featureTP_list.append(features_TPs)
           feature_importances_TP_list.append(features_TPs_flatten)  
           rep_roc_curve_list.append(best_rep_display)
           rep_importances_list.append(best_rep_importances)
           rep_tensor_list.append(X_rep_tensor)
           returned_layer_list.append(layer_size)

print("Gathering processes...",flush=True)
for i in range(QUEUE_SIZE): #Number of sub procs to run
       inputQueue.put('STOP')
print("Processess gathered...",flush=True)

#Write output
best_ann_ga_features = pd.DataFrame(data=feature_list).T
best_ann_ga_features.to_csv(basepath + fname + '_features.csv')
best_ann_ga_importances = pd.DataFrame(data=feature_importances_list).T
best_ann_ga_importances.to_csv(basepath + fname + '_importances.csv')
best_ann_ga_featuresTPs = pd.DataFrame(data=featureTP_list).T
best_ann_ga_featuresTPs.to_csv(basepath + fname + '_featuresTPs.csv')
best_ann_ga_importancesTPs = pd.DataFrame(data=feature_importances_TP_list).T
best_ann_ga_importancesTPs.to_csv(basepath + fname + '_importancesTPs.csv')

saveEverything = pd.DataFrame(newrows, columns=columns)
saveEverything.to_csv(basepath + fname + '_AllModels.csv')
unique_columns_df  = pd.DataFrame(unique_columns, columns=['Feature'])
unique_columns_df.to_csv(os.path.join(basepath,"Feature_Set.csv"))
elapsed_sec = time.time() - prog_start_time
elapsed_min = elapsed_sec / 60
elapsed_hours = elapsed_min / 60
rep_rows = 'NA'
if str2bool(useRep):
   rep_rows = X_rep.shape[0]
number_of_TPs = 'NA for 2D'
matrix_fill_percent = 'NA for 2D'
if dimension == '3D':
   matrix_fill_percent = float(np.load(input_dir + targ + '/matrix_fill_percent.npy'))
   number_of_TPs = X.shape[1]

timeDF = pd.DataFrame([[X.shape[0], rep_rows, number_of_TPs, num_unique_columns, k, matrix_fill_percent, elapsed_hours, elapsed_min, elapsed_sec]], columns=['Num_Rows_Train', 'Num_Rows_Test', 'Num_TPs', 'Num_Total_Features', 'Num_Kfolds', 'Matrix_Fill_Percent', 'Total Time in Hours', 'Total Time in Minutes', 'Total Time in Seconds'])
timeDF.to_csv(os.path.join(basepath,"Statistics.csv"), index=False)

if str2bool(useRep): #NOW FOR TEST IMPORTANCES AND PLOTS
   roc_plot_dir = basepath + '/ROC_Curves/'
   os.makedirs(roc_plot_dir)
   count=0 #for model labeling of output results
   for i,l in zip(rep_roc_curve_list, returned_layer_list):
      print('Synchronize with the list of layer sizes in AllModels file', l)
      i.plot()
      plt.savefig(roc_plot_dir + 'Model_' + str(count) + '_ROC_Curve.png')
      plt.close()
      count += 1
   roc_zip_filename = basepath + 'TEST_ROC_Curves'
   shutil.make_archive(roc_zip_filename, 'zip', roc_plot_dir)
   shutil.rmtree(roc_plot_dir)   

   shapley_scores_dir = basepath + '/Shapley_scores_best_model_test'
   os.makedirs(shapley_scores_dir)
   #Average feature importances over time periods
   count = 0      
   check_x_rep = np.array([x for x in X_rep], dtype=np.float32)
   if dimension == '3D':
      final_feature_TPs = np.array(featureTP_list[0])
      for onething in featureTP_list:
         if not np.array_equal(final_feature_TPs, np.array(onething)):
            print('feature_TP check failed. Exiting')
            exit(1)

   for rep_attribution in rep_importances_list:
      #BEGIN features averaged over time periods
      feature_mean_shap = None; test_mean = None
      if dimension == '3D':
         feature_mean_shap = np.mean(rep_attribution, axis=1) #This averages over TPs so is just features.
         test_mean = np.mean(check_x_rep, axis=1)
      else:
         feature_mean_shap = rep_attribution.copy()
         test_mean = check_x_rep.copy()
      data_frame = pd.DataFrame( test_mean, columns=unique_columns)  
      shap.summary_plot(feature_mean_shap, data_frame, max_display=20, plot_size=(7,5), show=False, color_bar=None, color_bar_label=[], class_names=[],title=None)
      plt.title("") #("Shapley value-impact on model output", x=-10)
      plt.xlabel("")
      plt.ylabel("")
      plt.gca().yaxis.set_ticks_position('none')
      plt.tight_layout()
      plt.savefig(shapley_scores_dir + '/Model_' + str(count) + '_shapley_plot.png')
      plt.close()
      #We want to write out these indvidual files
      shapley_frame = pd.DataFrame(feature_mean_shap, columns=unique_columns)
      shapley_frame.to_csv(shapley_scores_dir + '/Model_' + str(count) + '_shapley_values.csv')
      #END FEATURES AVGD OVER TPs
      if dimension == '3D':
         #shapley features and time periods plotting   
         shap_values_reshape = np.reshape(rep_attribution, (rep_attribution.shape[0], rep_attribution.shape[1] * rep_attribution.shape[2]), order='F')           
         X_rep_reshape = np.reshape(check_x_rep, (check_x_rep.shape[0], check_x_rep.shape[1] * check_x_rep.shape[2]), order='F') 
         data_frame = pd.DataFrame(X_rep_reshape, columns=final_feature_TPs)
         shap.summary_plot(shap_values_reshape, data_frame, max_display=15, plot_size=(7,5), show=False, color_bar=None, color_bar_label=[], class_names=[],title=None)
         plt.title("") #("Shapley value-impact on model output", x=-10)
         plt.xlabel("")
         plt.ylabel("")
         plt.gca().yaxis.set_ticks_position('none')
         plt.tight_layout()
         plt.savefig(shapley_scores_dir + '/Model_' + str(count) + '_shapley_plotTPs.png')
         plt.close()
         shapley_frame = pd.DataFrame(shap_values_reshape, columns=features_TPs)
         shapley_frame.to_csv(shapley_scores_dir + '/Model_' + str(count) + '_shapley_valuesTPs.csv')            
      count += 1

   #zip and save results
   shap_zip_filename = basepath + 'TEST_Shapley_Values_Plots'
   shutil.make_archive(shap_zip_filename, 'zip', shapley_scores_dir)
   shutil.rmtree(shapley_scores_dir)  #remove individual files that are now zipped

