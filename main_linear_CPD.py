import torch
from datetime import datetime

from Simulations.Linear_sysmdl import SystemModel
import Simulations.config as config
import Simulations.utils as utils
from Simulations.Linear_CPD.parameters import F_gen,F_CV,H_identity,H_onlyPos,\
   Q_gen,Q_CV,R_3,R_2,R_onlyPos,\
   m,m_cv

from Filters.KalmanFilter_test import KFTest

from KNet.KalmanNet_nn import KalmanNetNN

from Pipelines.Pipeline_EKF import Pipeline_EKF as Pipeline
from Pipelines.Pipeline_CPD import Pipeline_CPD as Pipeline_CPD

from CPDNet.CPDNet_nn import CPDNetNN

from Plot import Plot_extended as Plot

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)
path_results = 'KNet/'
path_results_CPD = 'CPDNet/'

print("Pipeline Start")
####################################
### Generative Parameters For CA ###
####################################
args = config.general_settings()
### Dataset parameters
args.N_E = 1000
args.N_CV = 100
args.N_T = 1000
offset = 0 ### Init condition of dataset
args.randomInit_train = True
args.randomInit_cv = True
args.randomInit_test = True

args.T = 100
args.T_test = 100
### training parameters
KnownRandInit_train = True # if true: use known random init for training, else: model is agnostic to random init
KnownRandInit_cv = True
KnownRandInit_test = True
args.use_cuda = True # use GPU or not
args.n_steps = 4000
args.n_batch = 10
args.lr = 1e-4
args.wd = 1e-4

if args.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")

if(args.randomInit_train or args.randomInit_cv or args.args.randomInit_test):
   std_gen = 1
else:
   std_gen = 0

if(KnownRandInit_train or KnownRandInit_cv or KnownRandInit_test):
   std_feed = 0
else:
   std_feed = 1

m1x_0 = torch.zeros(m) # Initial State
m1x_0_cv = torch.zeros(m_cv) # Initial State for CV
m2x_0 = std_feed * std_feed * torch.eye(m) # Initial Covariance for feeding to filters and KNet
m2x_0_gen = std_gen * std_gen * torch.eye(m) # Initial Covariance for generating dataset
m2x_0_cv = std_feed * std_feed * torch.eye(m_cv) # Initial Covariance for CV

#############################
###  Dataset Generation   ###
#############################
### PVA or P
Loss_On_AllState = False # if false: only calculate loss on position
Train_Loss_On_AllState = True # if false: only calculate training loss on position
CV_model = False # if true: use CV model, else: use CA model

CPDDatafolderName = 'Simulations/Linear_CPD/data/'
CPDDatafileName = 'CPD.pt'
DatafolderName = 'Simulations/Linear_CA/data/'
DatafileName = 'decimated_dt1e-2_T100_r0_randnInit.pt'

# System Model for KNet training
sys_model = SystemModel(F_gen, Q_gen, H_onlyPos, R_onlyPos, args.T, args.T_test)
sys_model.InitSequence(m1x_0, m2x_0)# x0 and P0

# Data Generation for CPD
sys_model_CPD = SystemModel(F_gen, Q_gen, H_onlyPos, R_onlyPos, args.T, args.T_test)
sys_model_CPD.InitSequence(m1x_0, m2x_0_gen)# x0 and P0
utils.DataGenCPD(args, sys_model_CPD, CPDDatafolderName+CPDDatafileName)
[train_input_CPD, train_target_CPD, cv_input_CPD, cv_target_CPD, test_input_CPD, test_target_CPD,train_init_CPD,cv_init_CPD,test_init_CPD] = torch.load(CPDDatafolderName+CPDDatafileName, map_location=device)

# Data Generation for KNet training
sys_model_gen = SystemModel(F_gen, Q_gen, H_onlyPos, R_onlyPos, args.T, args.T_test)
sys_model_gen.InitSequence(m1x_0, m2x_0_gen)# x0 and P0
print("Start Data Gen")
utils.DataGen(args, sys_model_gen, DatafolderName+DatafileName)
print("Load Original Data")
[train_input, train_target, cv_input, cv_target, test_input, test_target,train_init,cv_init,test_init] = torch.load(DatafolderName+DatafileName, map_location=device)

print("Data Shape")
print("testset state x size:",test_target.size())
print("testset observation y size:",test_input.size())
print("trainset state x size:",train_target.size())
print("trainset observation y size:",train_input.size())
print("cvset state x size:",cv_target.size())
print("cvset observation y size:",cv_input.size())

print("Compute Loss on All States (if false, loss on position only):", Loss_On_AllState)


##########################
### Evaluate KalmanNet ###
##########################
# Build Neural Network
KNet_model = KalmanNetNN()
KNet_model.NNBuild(sys_model, args)
print("Number of trainable parameters for KNet pass 1:",sum(p.numel() for p in KNet_model.parameters() if p.requires_grad))
## Train Neural Network
KNet_Pipeline = Pipeline(strTime, "KNet", "KNet")
KNet_Pipeline.setssModel(sys_model)
KNet_Pipeline.setModel(KNet_model)
KNet_Pipeline.setTrainingParams(args)

# Build Neural Network
CPDNet_model = CPDNetNN(args.sample_interval, 1, 1, 1)
print("Number of trainable parameters for CPDNet pass 1:",sum(p.numel() for p in KNet_model.parameters() if p.requires_grad))
## Train Neural Network
CPD_Pipeline = Pipeline_CPD(strTime, "CPDNet", "CPDNet")
CPD_Pipeline.setModel(CPDNet_model)
CPD_Pipeline.setssModel(sys_model_CPD)
CPD_Pipeline.setTrainingParams(args)





# if (KnownRandInit_train):
#    print("Train KNet with Known Random Initial State")
#    print("Train Loss on All States (if false, loss on position only):", Train_Loss_On_AllState)
#    [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, MaskOnState=not Train_Loss_On_AllState, randomInit = True, cv_init=cv_init,train_init=train_init)
# else:
#    print("Train KNet with Unknown Initial State")
#    print("Train Loss on All States (if false, loss on position only):", Train_Loss_On_AllState)
#    [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, MaskOnState=not Train_Loss_On_AllState)

if (KnownRandInit_test): 
   print("Generate CPD dataset with Known Random Initial State")
   ## Test Neural Network
   print("Compute Loss on All States (if false, loss on position only):", Loss_On_AllState)
   # [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,KNet_out,RunTime,error,index] = KNet_Pipeline.CPD_Dataset(sys_model, test_input_CPD, test_target_CPD, path_results,MaskOnState=not Loss_On_AllState,randomInit=True,test_init=test_init_CPD)
   KNet_Pipeline.CPD_Dataset(sys_model, test_input_CPD, test_target_CPD, path_results,path_results_CPD,MaskOnState=not Loss_On_AllState,randomInit=True,test_init=test_init_CPD)
else: 
   print("Generate CPD dataset with Unknown Initial State")
   ## Test Neural Network
   print("Compute Loss on All States (if false, loss on position only):", Loss_On_AllState)
   KNet_Pipeline.CPD_Dataset(sys_model, test_input_CPD, test_target_CPD, path_results,path_results_CPD,MaskOnState=not Loss_On_AllState)


# Load index_error data
index_error_data = torch.load(path_results_CPD+'/index_error.pt', map_location=device)

# Separate index and error
index = index_error_data['index']
error = index_error_data['error']



[MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = CPD_Pipeline.CPDNNTrain(sys_model_CPD,cv_input_CPD, cv_target_CPD, error, index, path_results_CPD, MaskOnState=not Train_Loss_On_AllState, randomInit = True, cv_init=cv_init_CPD)
