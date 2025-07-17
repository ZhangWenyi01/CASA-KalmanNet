import torch
from datetime import datetime
import numpy as np

from Simulations.Linear_sysmdl import SystemModel
import Simulations.config as config
import Simulations.utils as utils
# MEKF Test
from estimate_Q_with_EMKF import sliding_window_EMKF
from Simulations.Linear_CPD.parameters import F_gen,F_CV,F_rotated,H_identity,H_onlyPos,H_onlyPos_rotated,\
   Q_gen,Q_CV,R_3,R_2,R_onlyPos,\
   m,m_cv

from Filters.KalmanFilter_test import KFTest
from Filters.KalmanSmoother_test import KSTest
from Filters.Linear_KF import KalmanFilter

from KNet.KalmanNet_nn import KalmanNetNN

from Pipelines.Pipeline_EKF import Pipeline_EKF as Pipeline
from Pipelines.Pipeline_CPD import Pipeline_CPD as Pipeline_CPD
from Pipelines.Pipeline_Unsupervised import Pipeline_Unsupervised as Pipeline_Unsupervised
import matplotlib.pyplot as plt
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

# Change Point setting
change_point_params = {
   'changed_param':'R',
    'Q': 'grad',
    'R': 'grad',
    'F': F_rotated,
    'H': H_onlyPos_rotated
}


print("Pipeline Start")
####################################
### Generative Parameters For CA ###
####################################
args = config.general_settings()
### Dataset parameters
args.N_E = 1000
args.N_CV = 100
args.N_T = 10
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
args.n_steps = 50001
args.n_batch = 64
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

# System Model for KNet training
sys_model = SystemModel(F_gen, Q_gen, H_onlyPos, R_onlyPos, args.T, args.T_test)
sys_model.InitSequence(m1x_0, m2x_0)# x0 and P0

######################################### KalmanNet Training Stage #########################################

# Build Neural Network
KNet_model = KalmanNetNN()
KNet_model.NNBuild(sys_model, args)
print("Number of trainable parameters for KNet pass 1:",sum(p.numel() for p in KNet_model.parameters() if p.requires_grad))
## Train Neural Network
KNet_Pipeline = Pipeline(strTime, "KNet", "KNet")
KNet_Pipeline.setssModel(sys_model)
KNet_Pipeline.setModel(KNet_model)
KNet_Pipeline.setTrainingParams(args)

# if (KnownRandInit_train):
#    print("Train KNet with Known Random Initial State")
#    print("Train Loss on All States (if false, loss on position only):", Train_Loss_On_AllState)
#    [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, MaskOnState=not Train_Loss_On_AllState, randomInit = True, cv_init=cv_init,train_init=train_init)
# else:
#    print("Train KNet with Unknown Initial State")
#    print("Train Loss on All States (if false, loss on position only):", Train_Loss_On_AllState)
#    [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, MaskOnState=not Train_Loss_On_AllState)

# if (KnownRandInit_test): 
#    print("Test KNet with Known Random Initial State")
#    ## Test Neural Network
#    print("Compute Loss on All States (if false, loss on position only):", Loss_On_AllState)
#    [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,KNet_out,RunTime] = KNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results,MaskOnState=not Loss_On_AllState,randomInit=True,test_init=test_init)
# else: 
#    print("Test KNet with Unknown Initial State")
#    ## Test Neural Network
#    print("Compute Loss on All States (if false, loss on position only):", Loss_On_AllState)
#    [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,KNet_out,RunTime] = KNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results,MaskOnState=not Loss_On_AllState)

######################################### CPDNet #########################################
# This data will be passed to a fully trained KalmanNet
# and used to generate CPD dataset for training CPDNet. Change point inclued in 
# the dataset.



sys_model_CPD = SystemModel(F_gen, Q_gen, H_onlyPos, R_onlyPos, args.T, 
                           args.T_test,
                           Q_afterCPD=change_point_params['Q'],  
                           F_afterCPD=change_point_params['F'],    
                           H_afterCPD=change_point_params['H'],
                           R_afterCPD=change_point_params['R'],
                           param_to_change=change_point_params['changed_param']    
                           )
sys_model_CPD.InitSequence(m1x_0, m2x_0_gen)# x0 and P0
# Build Neural Network
CPDNet_model = CPDNetNN(args.sample_interval, 1, 1, 1)
print("Number of trainable parameters for CPDNet pass 1:",sum(p.numel() for p in CPDNet_model.parameters() if p.requires_grad))
## Train Neural Network
CPD_Pipeline = Pipeline_CPD(strTime, "CPDNet", "CPDNet")
CPD_Pipeline.setModel(CPDNet_model)
CPD_Pipeline.setssModel(sys_model_CPD)
CPD_Pipeline.setTrainingParams(args)

utils.DataGenCPD(args, sys_model_CPD, CPDDatafolderName+CPDDatafileName)

[train_input_CPD, train_target_CPD, cv_input_CPD, cv_target_CPD, test_input_CPD,
 test_target_CPD,train_init_CPD,cv_init_CPD,test_init_CPD,train_ChangePoint,
 cv_ChangePoint,test_ChangePoint] = torch.load(CPDDatafolderName+CPDDatafileName, map_location=device)

print("Generate CPD dataset with Known Random Initial State")
## Test Neural Network
print("Compute Loss on All States (if false, loss on position only):", Loss_On_AllState)
KNet_Pipeline.CPD_Dataset(sys_model, train_input_CPD, train_target_CPD,
                          cv_input_CPD,cv_target_CPD,test_input_CPD,test_target_CPD, 
                          path_results,path_results_CPD,MaskOnState=not Loss_On_AllState,
                          randomInit=True,train_init=train_init_CPD,test_init=test_init_CPD,cv_init=cv_init_CPD)

# Load index_error data
index_error_data = torch.load(path_results_CPD+'/index_error.pt', map_location=device)

# Dataset will be used in CPDNet training and test.
train_input = index_error_data['train_input']
train_target = index_error_data['train_target']
cv_input = index_error_data['cv_input']
cv_target = index_error_data['cv_target']
test_input = index_error_data['test_input']
test_target = index_error_data['test_target']

# Trajectory dataset from KNet
x_estimation_test = index_error_data['x_estimation_test']
x_ture_test = index_error_data['x_ture_test']
y_estimation_test = index_error_data['y_estimation_test']
y_ture_test = index_error_data['y_ture_test']

x_estimation_train = index_error_data['x_estimation_train']
x_ture_train = index_error_data['x_ture_train']
y_estimation_train = index_error_data['y_estimation_train']
y_ture_train = index_error_data['y_ture_train']

x_estimation_cv = index_error_data['x_estimation_cv']
x_ture_cv = index_error_data['x_ture_cv']
y_estimation_cv = index_error_data['y_estimation_cv']
y_ture_cv = index_error_data['y_ture_cv']

# [MSE_cv_linear_epoch, MSE_cv_dB_epoch, 
#  MSE_train_linear_epoch, MSE_train_dB_epoch] = CPD_Pipeline.CPDNNTrain(sys_model_CPD,
#                                                                        cv_input, 
#                                                                        cv_target, 
#                                                                        train_input, 
#                                                                        train_target, 
#                                                                        path_results_CPD, 
#                                                                        MaskOnState=not Train_Loss_On_AllState,
#                                                                        cv_init=cv_init_CPD)
 
# [MSE_test_linear_arr, MSE_test_linear_avg, 
#  MSE_test_dB_avg, x_out_test, t] = CPD_Pipeline.CPDNNTest(sys_model_CPD,
#                                                           test_input, test_target, 
#                                                           path_results_CPD,
#                                                           x_estimation_test,
#                                                           x_ture_test,
#                                                           y_estimation_test,
#                                                           y_ture_test, 
#                                                           MaskOnState=not Train_Loss_On_AllState)

# Unsupervised stage initialization
# Load CPDNet model
sys_model_online = SystemModel(F_gen, Q_gen, H_onlyPos, R_onlyPos, args.T, args.T_test)
sys_model_online.InitSequence(m1x_0, m2x_0)# x0 and P0
sys_model_KF = SystemModel(F_gen, Q_gen, H_onlyPos, R_onlyPos, args.T, args.T_test)
sys_model_KF.InitSequence(m1x_0, m2x_0)# x0 and P0

unsupervised_pipeline = Pipeline_Unsupervised()
unsupervised_pipeline.setCPDNet('CPDNet')
unsupervised_pipeline.setKNet('KNet')
unsupervised_pipeline.setssModel(sys_model_online)
args.n_batch = 1
unsupervised_pipeline.setTrainingParams(args)
# Kalman Filter processing

unsupervised_pipeline.Unsupervised_CPD_Online(sys_model_online,test_input_CPD,test_target_CPD,test_init_CPD)
[MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out, KF_detailed_results] = KFTest(args, 
                                 sys_model_KF, 
                                 test_input_CPD, 
                                 test_target_CPD, 
                                 allStates=True,
                                 test_init=test_init_CPD,
                                 randomInit= True,
                                 changepoint=test_ChangePoint,
                                 changeparameters=change_point_params)


# Run EMKF for all trajectories
emkf_mse_array = []
emkf_position_mse_array = []
emkf_full_state_mse_array = []
emkf_detailed_results = [] # New list to store detailed results

for traj_idx in range(args.N_T):
    
    current_input = test_input_CPD[traj_idx, :, :].T.cpu().numpy()  # (100, 1)
    current_target = test_target_CPD[traj_idx, :, :].T.cpu().numpy()  # (100, 3)
    current_init = test_init_CPD[traj_idx, :, 0].cpu().numpy()  # Get true initial state
    
    # Use more reasonable window parameters
    sliding_results = sliding_window_EMKF(
        observations=current_input,
        true_states=current_target,
        F_true=F_gen.cpu().numpy(),
        H_true=H_onlyPos.cpu().numpy(),
        R_true=R_onlyPos.cpu().numpy(),
        Q_initial=Q_gen.cpu().numpy(),
        window_size=50,      # Increase window size
        overlap=15,          # Reduce overlap
        verbose=False,
        true_init_state=current_init,
        allStates=Loss_On_AllState,  # Ensure consistency with KF test allStates parameter
        init_covariance=m2x_0.cpu().numpy()  # 🔥 Use the same initial covariance matrix as KF test
    )
    
    if sliding_results['filtered_states'] is not None:
        # Store linear MSE values (not dB) for proper averaging
        emkf_position_mse_array.append(sliding_results['position_mse_linear'])  # Store linear values for correct averaging
        emkf_full_state_mse_array.append(sliding_results['full_state_mse_linear'])  # Store linear values for correct averaging
        emkf_mse_array.append(sliding_results['mse_loss'])
        
        # Store detailed MSE results for dimension-wise analysis
        emkf_detailed_results.append(sliding_results['detailed_mse'])
    else:
        emkf_mse_array.append(float('inf'))
        emkf_position_mse_array.append(float('inf'))
        emkf_full_state_mse_array.append(float('inf'))
        emkf_detailed_results.append({
            'MSE_position_dB': float('inf'),
            'MSE_velocity_dB': float('inf'),
            'MSE_acceleration_dB': float('inf'),
            'mse_position_linear': float('inf'),
            'mse_velocity_linear': float('inf'),
            'mse_acceleration_linear': float('inf')
        })

# Calculate EMKF average performance with dimension-wise output (following KF test logic)
if emkf_mse_array:
    valid_results = [x for x in emkf_mse_array if x != float('inf')]
    if valid_results:
        # Get valid detailed results
        valid_detailed = [x for x in emkf_detailed_results if x['mse_position_linear'] != float('inf')]
        
        if valid_detailed:
            # Calculate dimension-wise statistics using linear MSE averaging (consistent with KF test)
            position_linear_results = [x['mse_position_linear'] for x in valid_detailed]
            velocity_linear_results = [x['mse_velocity_linear'] for x in valid_detailed if x['mse_velocity_linear'] is not None and x['mse_velocity_linear'] != float('inf')]
            acceleration_linear_results = [x['mse_acceleration_linear'] for x in valid_detailed if x['mse_acceleration_linear'] is not None and x['mse_acceleration_linear'] != float('inf')]
            
            # Calculate averages and convert to dB (following KF test and Pipeline_Unsupervised logic)
            position_avg_linear = np.mean(position_linear_results)
            position_avg_dB = 10 * np.log10(position_avg_linear)
            
            # Calculate overall MSE from linear values
            position_valid = [x for x in emkf_position_mse_array if x != float('inf')]
            full_state_valid = [x for x in emkf_full_state_mse_array if x != float('inf')]
            
            print("EMKF Results:")
            if full_state_valid:
                full_state_avg_linear = np.mean(full_state_valid)
                full_state_avg_dB = 10 * np.log10(full_state_avg_linear)
                print(f"  Overall MSE: {full_state_avg_dB:.6f} dB")
            
            print("EMKF MSE by Dimension:")
            print(f"  Position: {position_avg_dB:.6f} dB")
            if velocity_linear_results:
                velocity_avg_linear = np.mean(velocity_linear_results)
                velocity_avg_dB = 10 * np.log10(velocity_avg_linear)
                print(f"  Velocity: {velocity_avg_dB:.6f} dB")
            if acceleration_linear_results:
                acceleration_avg_linear = np.mean(acceleration_linear_results)
                acceleration_avg_dB = 10 * np.log10(acceleration_avg_linear)
                print(f"  Acceleration: {acceleration_avg_dB:.6f} dB")
        else:
            print(f"EMKF: All trajectories failed")
    else:
        print(f"EMKF: All trajectories failed")
else:
    print(f"EMKF: No results")
    
    
# # Kalman Smoother processing
# print("\n" + "="*60)
# print("Running Kalman Smoother for comparison...")
# print("="*60)
# [MSE_KS_linear_arr, MSE_KS_linear_avg, MSE_KS_dB_avg, KS_out, ks_detailed] = KSTest(args, 
#                                  sys_model_KF, 
#                                  test_input_CPD, 
#                                  test_target_CPD, 
#                                  allStates=Loss_On_AllState,
#                                  test_init=test_init_CPD,
#                                  randomInit=True,
#                                  changepoint=test_ChangePoint,
#                                  changeparameters=change_point_params,
#                                  first_dim_only=True)
