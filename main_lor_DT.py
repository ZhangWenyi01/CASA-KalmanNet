import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn
from Filters.EKF_test import EKFTest

from Simulations.Extended_sysmdl import SystemModel
from Pipelines.Pipeline_Unsupervised import Pipeline_Unsupervised as Pipeline_Unsupervised
from Simulations.utils import DataGen, Short_Traj_Split, DataGenCPD
import Simulations.config as config

from Pipelines.Pipeline_lor import Pipeline_EKF
from Pipelines.Pipeline_CPD import Pipeline_CPD

from datetime import datetime

from KNet.KalmanNet_nn import KalmanNetNN
from CPDNet.CPDNet_nn import CPDNetNN

from Simulations.Lorenz_Atractor.parameters import m1x_0, m2x_0, m, n,\
f, h, hRotate, H_Rotate, H_Rotate_inv, Q_structure, R_structure

# Plot 3D trajectories for comparison
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("Pipeline Start")
################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

###################
###  Settings   ###
###################
args = config.general_settings()
### dataset parameters
args.N_E = 1000
args.N_CV = 100
args.N_T = 200
args.T = 100
args.T_test = 100
### training parameters
args.use_cuda = True # use GPU or not
args.n_steps = 20000
args.n_batch = 30
args.lr = 1e-3
args.wd = 1e-3

if args.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")

offset = 0 # offset for the data
chop = False # whether to chop data sequences into shorter sequences
path_results = 'KNet/'
DatafolderName = 'Simulations/Lorenz_Atractor/data' + '/'
switch = 'partial' # 'full' or 'partial' or 'estH'
   
# noise q and r
r2 = torch.tensor([0.1]) # [100, 10, 1, 0.1, 0.01]
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10)
q2 = torch.mul(v,r2)

Q = q2[0] * Q_structure
R = r2[0] * R_structure

print("1/r2 [dB]: ", 10 * torch.log10(1/r2[0]))
print("1/q2 [dB]: ", 10 * torch.log10(1/q2[0]))

traj_resultName = ['traj_lorDT_rq1030_T100.pt']
dataFileName = ['data_lor_v20_rq1030_T100.pt']

#########################################
###  Generate and load data DT case   ###
#########################################

sys_model = SystemModel(f, Q, hRotate, R, args.T, args.T_test, m, n)# parameters for GT
sys_model.InitSequence(m1x_0, m2x_0)# x0 and P0

print("Start Data Gen")
DataGen(args, sys_model, DatafolderName + dataFileName[0])
print("Data Load")
print(dataFileName[0])
[train_input_long,train_target_long, cv_input, cv_target, test_input, test_target,_,_,_] =  torch.load(DatafolderName + dataFileName[0], map_location=device)  
if chop: 
   print("chop training data")    
   [train_target, train_input, train_init] = Short_Traj_Split(train_target_long, train_input_long, args.T)
   # [cv_target, cv_input] = Short_Traj_Split(cv_target, cv_input, args.T)
else:
   print("no chopping") 
   train_target = train_target_long[:,:,0:args.T]
   train_input = train_input_long[:,:,0:args.T] 
   # cv_target = cv_target[:,:,0:args.T]
   # cv_input = cv_input[:,:,0:args.T]  

print("trainset size:",train_target.size())
print("cvset size:",cv_target.size())
print("testset size:",test_target.size())


# Model with partial info
sys_model_partial = SystemModel(f, Q, h, R, args.T, args.T_test, m, n)
sys_model_partial.InitSequence(m1x_0, m2x_0)
# Model for 2nd pass
sys_model_pass2 = SystemModel(f, Q, h, R, args.T, args.T_test, m, n)# parameters for GT
sys_model_pass2.InitSequence(m1x_0, m2x_0)# x0 and P0

########################################
### Evaluate Observation Noise Floor ###
########################################
N_T = len(test_input)
loss_obs = nn.MSELoss(reduction='mean')
MSE_obs_linear_arr = torch.empty(N_T)# MSE [Linear]

for j in range(0, N_T): 
   reversed_target = torch.matmul(H_Rotate_inv.to(device), test_input[j])      
   MSE_obs_linear_arr[j] = loss_obs(reversed_target, test_target[j]).item()
MSE_obs_linear_avg = torch.mean(MSE_obs_linear_arr)
MSE_obs_dB_avg = 10 * torch.log10(MSE_obs_linear_avg)

# Standard deviation
MSE_obs_linear_std = torch.std(MSE_obs_linear_arr, unbiased=True)

# Confidence interval
obs_std_dB = 10 * torch.log10(MSE_obs_linear_std + MSE_obs_linear_avg) - MSE_obs_dB_avg

print("Observation Noise Floor(test dataset) - MSE LOSS:", MSE_obs_dB_avg, "[dB]")
print("Observation Noise Floor(test dataset) - STD:", obs_std_dB, "[dB]")


########################
### Evaluate Filters ###
########################
### Evaluate EKF true
print("Evaluate EKF true")
[MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(args, sys_model, test_input, test_target)
### Evaluate EKF partial
print("Evaluate EKF partial")
[MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, EKF_KG_array_partial, EKF_out_partial] = EKFTest(args, sys_model_partial, test_input, test_target)

### Save trajectories
trajfolderName = 'Filters' + '/'
DataResultName = traj_resultName[0]
EKF_sample = torch.reshape(EKF_out[0],[1,m,args.T_test])
target_sample = torch.reshape(test_target[0,:,:],[1,m,args.T_test])
input_sample = torch.reshape(test_input[0,:,:],[1,n,args.T_test])
torch.save({
            'EKF': EKF_sample,
            'ground_truth': target_sample,
            'observation': input_sample,
            }, trajfolderName+DataResultName)

#####################
### Evaluate KNet ###
#####################
if switch == 'full':
   ## KNet with full info ####################################################################################
   ################
   ## KNet full ###
   ################  
   ## Build Neural Network
   print("KNet with full model info")
   KNet_model = KalmanNetNN()
   KNet_model.NNBuild(sys_model, args)
   # ## Train Neural Network
   KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KNet")
   KNet_Pipeline.setssModel(sys_model)
   KNet_Pipeline.setModel(KNet_model)
   print("Number of trainable parameters for KNet:",sum(p.numel() for p in KNet_model.parameters() if p.requires_grad))
   KNet_Pipeline.setTrainingParams(args) 
   # if(chop):
   #    [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results,randomInit=True,train_init=train_init)
   # else:
   #    [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results)
   ## Test Neural Network
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,Knet_out,RunTime] = KNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)

####################################################################################
elif switch == 'partial':
   ## KNet with model mismatch ####################################################################################
   ###################
   ## KNet partial ###
   ####################
   ## Build Neural Network
   print("KNet with observation model mismatch")
   KNet_model = KalmanNetNN()
   KNet_model.NNBuild(sys_model_partial, args)
   ## Train Neural Network
   KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KNet")
   KNet_Pipeline.setssModel(sys_model_partial)
   KNet_Pipeline.setModel(KNet_model)
   KNet_Pipeline.setTrainingParams(args)
   # if(chop):
   #    [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model_partial, cv_input, cv_target, train_input, train_target, path_results,randomInit=True,train_init=train_init)
   # else:
   #    [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model_partial, cv_input, cv_target, train_input, train_target, path_results)
   # ## Test Neural Network
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,Knet_out,RunTime] = KNet_Pipeline.NNTest(sys_model_partial, test_input, test_target, path_results)

###################################################################################
elif switch == 'estH':
   print("True Observation matrix H:", H_Rotate)
   ### Least square estimation of H
   X = torch.squeeze(train_target[:,:,0])
   Y = torch.squeeze(train_input[:,:,0])
   for t in range(1,args.T):
      X_t = torch.squeeze(train_target[:,:,t])
      Y_t = torch.squeeze(train_input[:,:,t])
      X = torch.cat((X,X_t),0)
      Y = torch.cat((Y,Y_t),0)
   Y_1 = torch.unsqueeze(Y[:,0],1)
   Y_2 = torch.unsqueeze(Y[:,1],1)
   Y_3 = torch.unsqueeze(Y[:,2],1)
   H_row1 = torch.matmul(torch.matmul(torch.inverse(torch.matmul(X.T,X)),X.T),Y_1)
   H_row2 = torch.matmul(torch.matmul(torch.inverse(torch.matmul(X.T,X)),X.T),Y_2)
   H_row3 = torch.matmul(torch.matmul(torch.inverse(torch.matmul(X.T,X)),X.T),Y_3)
   H_hat = torch.cat((H_row1.T,H_row2.T,H_row3.T),0)
   print("Estimated Observation matrix H:", H_hat)

   def h_hat(x, jacobian=False):
    H = H_hat.reshape((1, n, m)).repeat(x.shape[0], 1, 1) # [batch_size, n, m] 
    y = torch.bmm(H,x)
    if jacobian:
        return y, H
    else:
        return y

   # Estimated model
   sys_model_esth = SystemModel(f, Q, h_hat, R, args.T, args.T_test, m, n)
   sys_model_esth.InitSequence(m1x_0, m2x_0)

   ################
   ## KNet estH ###
   ################
   print("KNet with estimated H")
   KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KNetEstH_"+ dataFileName[0])
   KNet_Pipeline.setssModel(sys_model_esth)
   KNet_model = KalmanNetNN()
   KNet_model.NNBuild(sys_model_esth, args)
   KNet_Pipeline.setModel(KNet_model)
   KNet_Pipeline.setTrainingParams(args)
   # [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model_esth, cv_input, cv_target, train_input, train_target, path_results)
   ## Test Neural Network
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,Knet_out,RunTime] = KNet_Pipeline.NNTest(sys_model_esth, test_input, test_target, path_results)
   
###################################################################################
else:
   print("Error in switch! Please try 'full' or 'partial' or 'estH'.")





######################################### CPDNet #########################################
# This data will be passed to a fully trained KalmanNet
# and used to generate CPD dataset for training CPDNet. Change point inclued in 
# the dataset.
CPDDatafolderName = 'CPDNet_lor/'
CPDDatafileName = 'data_lor_v20_rq1030_T100.pt'
path_results_CPD = 'CPDNet_lor/'

sys_model_CPD = SystemModel(f, Q, h, R, args.T, args.T_test, m, n, 
                           args.T_test,
                           Q_afterCPD=Q*500,  
                           f_afterCPD=f,    
                           h_afterCPD=h,
                           R_afterCPD=R*1.5,
                           param_to_change='Q'    
                           )
sys_model_CPD.InitSequence(m1x_0, m2x_0)# x0 and P0
# Build Neural Network
CPDNet_model = CPDNetNN(args.sample_interval, 1, 1, 1)
print("Number of trainable parameters for CPDNet pass 1:",sum(p.numel() for p in CPDNet_model.parameters() if p.requires_grad))
## Train Neural Network
CPD_Pipeline = Pipeline_CPD(strTime, "CPDNet_lor", "CPDNet")
CPD_Pipeline.setModel(CPDNet_model)
CPD_Pipeline.setssModel(sys_model_CPD)
CPD_Pipeline.setTrainingParams(args)

DataGenCPD(args, sys_model_CPD, CPDDatafolderName+CPDDatafileName)
[train_input_CPD, train_target_CPD, cv_input_CPD, cv_target_CPD, test_input_CPD, test_target_CPD,train_init_CPD,cv_init_CPD,test_init_CPD] = torch.load(CPDDatafolderName+CPDDatafileName, map_location=device)
print("Generate CPD dataset with Known Random Initial State")
## Test Neural Network
KNet_model = KalmanNetNN()
KNet_model.NNBuild(sys_model_partial, args)
## Train Neural Network
KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KNet")
KNet_Pipeline.setssModel(sys_model_partial)
KNet_Pipeline.setModel(KNet_model)
KNet_Pipeline.setTrainingParams(args)
KNet_Pipeline.CPD_Dataset(sys_model_partial, train_input_CPD, train_target_CPD,cv_input_CPD,cv_target_CPD,test_input_CPD,test_target_CPD, path_results,path_results_CPD,randomInit=True,train_init=train_init_CPD,test_init=test_init_CPD,cv_init=cv_init_CPD,scale_param=9)

# Load index_error data
index_error_data = torch.load(path_results_CPD+'/index_error.pt', map_location=device)

# Separate index and error
train_input = index_error_data['train_input']
train_target = index_error_data['train_target']
cv_input = index_error_data['cv_input']
cv_target = index_error_data['cv_target']
test_input = index_error_data['test_input']
test_target = index_error_data['test_target']

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

# # Create figure with three subplots (train, cv, test)
# fig = plt.figure(figsize=(20, 24))

# # Train data subplot
# ax1 = fig.add_subplot(321)
# batch_idx = 10  # Select a batch to plot

# # Plot CPD probabilities for train data
# ax1.plot(train_input[batch_idx, 0, :].cpu().detach().numpy(), label='train_input', color='blue')
# ax1.plot(train_target[batch_idx, 0, :].cpu().detach().numpy(), label='train_target', color='red')
# ax1.set_title('Train CPD Probabilities')
# ax1.set_xlabel('Time Steps')
# ax1.set_ylabel('Probability')
# ax1.legend()
# ax1.grid(True)

# # CV data subplot
# ax2 = fig.add_subplot(323)
# # Plot CPD probabilities for cv data
# ax2.plot(cv_input[batch_idx, 0, :].cpu().detach().numpy(), label='cv_input', color='blue')
# ax2.plot(cv_target[batch_idx, 0, :].cpu().detach().numpy(), label='cv_target', color='red')
# ax2.set_title('CV CPD Probabilities')
# ax2.set_xlabel('Time Steps')
# ax2.set_ylabel('Probability')
# ax2.legend()
# ax2.grid(True)

# # Test data subplot
# ax3 = fig.add_subplot(325)
# # Plot CPD probabilities for test data
# ax3.plot(test_input[batch_idx, 0, :].cpu().detach().numpy(), label='test_input', color='blue')
# ax3.plot(test_target[batch_idx, 0, :].cpu().detach().numpy(), label='test_target', color='red')
# ax3.set_title('Test CPD Probabilities')
# ax3.set_xlabel('Time Steps')
# ax3.set_ylabel('Probability')
# ax3.legend()
# ax3.grid(True)

# # Train 3D trajectories
# ax4 = fig.add_subplot(322, projection='3d')
# x_est_train = x_estimation_train[batch_idx].cpu().detach().numpy()
# x_true_train = x_ture_train[batch_idx].cpu().detach().numpy()
# y_est_train = y_estimation_train[batch_idx].cpu().detach().numpy()
# y_true_train = y_ture_train[batch_idx].cpu().detach().numpy()

# ax4.plot(x_est_train[0,:], x_est_train[1,:], x_est_train[2,:], 
#          label='Estimated State', color='red', linestyle='--')
# ax4.plot(x_true_train[0,:], x_true_train[1,:], x_true_train[2,:], 
#          label='True State', color='blue')
# ax4.plot(y_est_train[0,:], y_est_train[1,:], y_est_train[2,:], 
#          label='Estimated Observation', color='green', linestyle=':')
# ax4.plot(y_true_train[0,:], y_true_train[1,:], y_true_train[2,:], 
#          label='True Observation', color='purple', linestyle='-.')
# ax4.set_title('Train 3D Trajectories')
# ax4.set_xlabel('X')
# ax4.set_ylabel('Y')
# ax4.set_zlabel('Z')
# ax4.legend()

# # CV 3D trajectories
# ax5 = fig.add_subplot(324, projection='3d')
# x_est_cv = x_estimation_cv[batch_idx].cpu().detach().numpy()
# x_true_cv = x_ture_cv[batch_idx].cpu().detach().numpy()
# y_est_cv = y_estimation_cv[batch_idx].cpu().detach().numpy()
# y_true_cv = y_ture_cv[batch_idx].cpu().detach().numpy()

# ax5.plot(x_est_cv[0,:], x_est_cv[1,:], x_est_cv[2,:], 
#          label='Estimated State', color='red', linestyle='--')
# ax5.plot(x_true_cv[0,:], x_true_cv[1,:], x_true_cv[2,:], 
#          label='True State', color='blue')
# ax5.plot(y_est_cv[0,:], y_est_cv[1,:], y_est_cv[2,:], 
#          label='Estimated Observation', color='green', linestyle=':')
# ax5.plot(y_true_cv[0,:], y_true_cv[1,:], y_true_cv[2,:], 
#          label='True Observation', color='purple', linestyle='-.')
# ax5.set_title('CV 3D Trajectories')
# ax5.set_xlabel('X')
# ax5.set_ylabel('Y')
# ax5.set_zlabel('Z')
# ax5.legend()

# # Test 3D trajectories
# ax6 = fig.add_subplot(326, projection='3d')
# x_est_test = x_estimation_test[batch_idx].cpu().detach().numpy()
# x_true_test = x_ture_test[batch_idx].cpu().detach().numpy()
# y_est_test = y_estimation_test[batch_idx].cpu().detach().numpy()
# y_true_test = y_ture_test[batch_idx].cpu().detach().numpy()

# ax6.plot(x_est_test[0,:], x_est_test[1,:], x_est_test[2,:], 
#          label='Estimated State', color='red', linestyle='--')
# ax6.plot(x_true_test[0,:], x_true_test[1,:], x_true_test[2,:], 
#          label='True State', color='blue')
# ax6.plot(y_est_test[0,:], y_est_test[1,:], y_est_test[2,:], 
#          label='Estimated Observation', color='green', linestyle=':')
# ax6.plot(y_true_test[0,:], y_true_test[1,:], y_true_test[2,:], 
#          label='True Observation', color='purple', linestyle='-.')
# ax6.set_title('Test 3D Trajectories')
# ax6.set_xlabel('X')
# ax6.set_ylabel('Y')
# ax6.set_zlabel('Z')
# ax6.legend()

# plt.tight_layout()
# plt.show()


# [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = CPD_Pipeline.CPDNNTrain(sys_model_CPD,cv_input, cv_target, train_input, train_target, path_results_CPD, cv_init=cv_init_CPD)
# [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, x_out_test, t] = CPD_Pipeline.CPDNNTest(sys_model_CPD,test_input, test_target, path_results_CPD,x_estimation_test,x_ture_test,y_estimation_test,y_ture_test)


# Unsupervised stage initialization
# Load CPDNet model
sys_model_online = SystemModel(f, Q, h, R, args.T, args.T_test, m, n, 
                           args.T_test,
                           Q_afterCPD=Q*500,  
                           f_afterCPD=f,    
                           h_afterCPD=h,
                           R_afterCPD=R*1.5,
                           param_to_change='Q'    
                           )
sys_model_online.InitSequence(m1x_0, m2x_0)# x0 and P0
sys_model_KF = SystemModel(f, Q, h, R, args.T, args.T_test, m, n, 
                           args.T_test,
                           Q_afterCPD=Q*500,  
                           f_afterCPD=f,    
                           h_afterCPD=h,
                           R_afterCPD=R*1.5,
                           param_to_change='Q'    
                           )
sys_model_KF.InitSequence(m1x_0, m2x_0)# x0 and P0

unsupervised_pipeline = Pipeline_Unsupervised()
unsupervised_pipeline.setCPDNet('CPDNet_lor')
unsupervised_pipeline.setKNet_lor('KNet')
unsupervised_pipeline.setssModel(sys_model_online)
args.n_batch = 1
unsupervised_pipeline.setTrainingParams(args)
# Kalman Filter processing
EKFTest(args, sys_model_KF, y_ture_train, x_ture_train)


unsupervised_pipeline.NNTrain_lor(sys_model_online,y_ture_train,x_ture_train,train_init_CPD)