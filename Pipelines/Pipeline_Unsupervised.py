import os
import torch
import copy
import torch.nn as nn
import random
import time
from Plot import Plot_extended
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from Simulations.utils import cpd_dataset_process_lor,cpd_dataset_process
from tqdm import tqdm
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs')

class Pipeline_Unsupervised:
    # Initialize the pipeline with CPDNet path and KNet path
    def __init__(self):
        # self.cpdnet_path = cpdnet_path
        # self.knet_path = knet_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Setting CPDNet 
    def setCPDNet(self, path_results):
        model_path = os.path.join(path_results, 'best-model.pt')
        self.CPDmodel = torch.load(model_path, 
                                map_location=self.device,weights_only=False) 
        
    
    # Setting KNet
    def setKNet(self, path_results):
        model_path = os.path.join(path_results, 'best-model.pt')
        self.KNetmodel = torch.load(model_path, 
                        map_location=self.device,weights_only=False) 
    def setKNet_lor(self, path_results):
        model_path = os.path.join(path_results, 'lor-best-model.pt')
        self.KNetmodel = torch.load(model_path, 
                        map_location=self.device,weights_only=False) 
        
    def setssModel(self, ssModel):
        self.ssModel = ssModel
        
    def ResetOptimizer(self):
        # This method is now unused in the optimized version
        self.optimizer = torch.optim.Adam(self.KNetmodel.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        
    def setTrainingParams(self, args):
        self.args = args
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.N_steps = args.n_steps  # Number of Training Steps
        self.N_B = args.n_batch # Number of Samples in Batch
        self.learningRate = args.lr # Learning Rate
        self.weightDecay = args.wd # L2 Weight Regularization - Weight Decay
        self.alpha = args.alpha # Composition loss factor
        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.sample_interval = args.sample_interval
        self.optimizer = torch.optim.Adam(self.KNetmodel.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        

    def Unsupervised_CPD_Online(self, SysModel,y_observation,x_true,train_init):
        
        self.N_T = len(y_observation)  # Number of trajectories
        self.stride = 5
        
        # Initialize computation time tracking for each trajectory
        self.original_compute_times = torch.zeros(self.N_T)
        self.cpd_unsupervised_compute_times = torch.zeros(self.N_T)
        self.only_unsupervised_compute_times = torch.zeros(self.N_T)

        ###############################
        ### Initialize Models and Optimizers ###
        ###############################
        # Set model modes once at the beginning
        self.KNetmodel.eval()           # Baseline model - no training
        self.KNetmodel.batch_size = 1
        self.CPDmodel.eval()            # CPD detection model
        self.CPDmodel.batch_size = 1
        
        # Create independent model copies with proper configurations
        original_model = copy.deepcopy(self.KNetmodel)
        original_model.train()          # CPD-based online learning
        original_model.batch_size = 1
        
        unsupervised_model = copy.deepcopy(self.KNetmodel)
        unsupervised_model.train()      # Continuous online learning
        unsupervised_model.batch_size = 1
        
        # Ensure all three models use the provided (possibly mismatched) system dynamics
        self.KNetmodel.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)
        original_model.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)
        unsupervised_model.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)
        
        # Initialize optimizers once
        self.optimizer_original = torch.optim.Adam(original_model.parameters(), lr=2e-3, weight_decay=self.weightDecay)
        self.optimizer_unsupervised = torch.optim.Adam(unsupervised_model.parameters(), lr=2e-3, weight_decay=self.weightDecay)
        
        # Allocate result arrays
        self.observation_predictions = torch.empty((self.N_T, self.ssModel.n, SysModel.T))
        self.state_predictions = torch.empty((self.N_T, self.ssModel.m, SysModel.T))
        self.observation_original = torch.empty((self.N_T, self.ssModel.n, SysModel.T))
        self.state_original = torch.empty((self.N_T, self.ssModel.m, SysModel.T))
        self.observation_unsupervised = torch.empty((self.N_T, self.ssModel.n, SysModel.T))
        self.state_unsupervised = torch.empty((self.N_T, self.ssModel.m, SysModel.T))
        
        # MSE storage arrays for both overall and dimension-wise MSE
        # Overall MSE arrays
        self.MSE_Original_linear_arr = torch.empty((self.N_T,))
        self.MSE_Unsupervised_linear_arr = torch.empty((self.N_T,))
        self.MSE_Ours_linear_arr = torch.empty((self.N_T,))
        
        # Dimension-wise MSE arrays
        self.MSE_Original_linear_arr_by_dim = torch.empty((self.N_T, 3))  # 3 dimensions: pos, vel, acc
        self.MSE_Unsupervised_linear_arr_by_dim = torch.empty((self.N_T, 3))
        self.MSE_Ours_linear_arr_by_dim = torch.empty((self.N_T, 3))
        
        # Start looping over trajectories
        for trajectorie in range(self.N_T):
            print('Trajectory: ', trajectorie + 1, '/', self.N_T)

            # Initialize time counters for this trajectory
            trajectory_original_time = 0.0
            trajectory_cpd_unsupervised_time = 0.0
            trajectory_only_unsupervised_time = 0.0

            # Load the next trajectory
            y_training = torch.unsqueeze(y_observation[trajectorie, :, :],0).requires_grad_(True).to(self.device)
            
            # Initialize all models with the trajectory's initial state
            init_state = torch.unsqueeze(train_init[trajectorie,:,:],0).to(self.device)
            self.KNetmodel.InitSequence(init_state, self.args.T)
            original_model.InitSequence(init_state, self.args.T)
            unsupervised_model.InitSequence(init_state, self.args.T)
            
            # Initialize hidden states
            self.KNetmodel.init_hidden_KNet()
            original_model.init_hidden_KNet()
            unsupervised_model.init_hidden_KNet()
            
            # Calculate the number of strides required
            number_of_stride = int(self.ssModel.T / self.stride)
            cpd_input = torch.zeros(1, 1, SysModel.T).to(self.device)
            

            # Go through the whole trajectory step by step
            for stride in trange(number_of_stride):
                observation = y_training[:, :, (stride * self.stride):(stride * self.stride + self.stride)].to(self.device)
                
                # Allocate output tensors for this stride
                x_out_online = torch.empty(1, self.ssModel.m, self.stride).to(self.device)
                y_out_online = torch.zeros(1, self.ssModel.n, self.stride).to(self.device)
                x_out_original = torch.empty(1, self.ssModel.m, self.stride).to(self.device)
                y_out_original = torch.zeros(1, self.ssModel.n, self.stride).to(self.device)
                x_out_unsupervised = torch.empty(1, self.ssModel.m, self.stride).to(self.device)
                y_out_unsupervised = torch.zeros(1, self.ssModel.n, self.stride).to(self.device)
                
                # Update model states for this stride (no mode changes needed)
                self.KNetmodel.InitSequence(self.KNetmodel.m1x_posterior.clone().detach(), SysModel.T)
                original_model.InitSequence(original_model.m1x_posterior.clone().detach(), SysModel.T)
                unsupervised_model.InitSequence(unsupervised_model.m1x_posterior.clone().detach(), SysModel.T)
                self.KNetmodel.init_hidden_KNet()
                original_model.init_hidden_KNet()
                unsupervised_model.init_hidden_KNet()

                # Process each observation in the current stride
                for i in range(self.stride):
                    obs_i = torch.unsqueeze(observation[:, :, i], dim=2)
                    
                    # KNet baseline (no training)
                    start_time = time.time()
                    x_out_online[:, :, i] = torch.squeeze(self.KNetmodel(obs_i))
                    y_out_online[:, :, i] = torch.squeeze(self.KNetmodel.m1y)
                    trajectory_cpd_unsupervised_time += time.time() - start_time
                    
                    # Original model (CPD-based online learning)
                    start_time = time.time()
                    x_out_original[:, :, i] = torch.squeeze(original_model(obs_i))
                    y_out_original[:, :, i] = torch.squeeze(original_model.m1y)
                    trajectory_original_time += time.time() - start_time
                    
                    # Unsupervised model (continuous online learning)
                    start_time = time.time()
                    x_out_unsupervised[:, :, i] = torch.squeeze(unsupervised_model(obs_i))
                    y_out_unsupervised[:, :, i] = torch.squeeze(unsupervised_model.m1y)
                    trajectory_only_unsupervised_time += time.time() - start_time

                self.observation_predictions[trajectorie, :,
                (stride * self.stride):(stride * self.stride + self.stride)] = y_out_online.detach()
                self.state_predictions[trajectorie, :,
                (stride * self.stride):(stride * self.stride + self.stride)] = x_out_online.detach()
                self.observation_original[trajectorie, :,
                (stride * self.stride):(stride * self.stride + self.stride)] = y_out_original.detach()
                self.state_original[trajectorie, :,
                (stride * self.stride):(stride * self.stride + self.stride)] = x_out_original.detach()
                self.observation_unsupervised[trajectorie, :,
                (stride * self.stride):(stride * self.stride + self.stride)] = y_out_unsupervised.detach()
                self.state_unsupervised[trajectorie, :,
                (stride * self.stride):(stride * self.stride + self.stride)] = x_out_unsupervised.detach()
                
                if stride >0:
                    for t in range(self.stride):
                        cpd_input[:, :, (stride-1)*self.stride+t] = cpd_dataset_process(torch.unsqueeze(self.observation_predictions[trajectorie, :,(stride * self.stride-self.stride+t):(stride * self.stride+t)],dim=0).to(self.device),
                                                                    y_training[:, :, (stride * self.stride-self.stride+t):(stride * self.stride+t)])
                    cpd_out_tmp = self.CPDmodel(cpd_input[:, :, (stride-1)*self.stride:(stride-1)*self.stride+self.stride])
                    new_lr = (20*cpd_out_tmp*self.learningRate).item()
                    for param_group in self.optimizer_original.param_groups:
                        param_group['lr'] = new_lr

                    if cpd_out_tmp > 0.5:
                        LOSS_original = self.loss_fn(y_out_original,observation)
                        self.optimizer_original.zero_grad()
                        LOSS_original.backward()
                        self.optimizer_original.step()
                        
                        del LOSS_original
                
                LOSS_unsupervised = self.loss_fn(y_out_unsupervised,observation)
                self.optimizer_unsupervised.zero_grad()
                LOSS_unsupervised.backward()
                self.optimizer_unsupervised.step()
                    
                del observation,LOSS_unsupervised
            # Reset the optimizer for the next trajectory
            # self.ResetOptimizer()
            # self.optimizer_unsupervised = torch.optim.Adam(unsupervised_model.parameters(), lr=2e-3, weight_decay=self.weightDecay)
            # self.optimizer_original = torch.optim.Adam(original_model.parameters(), lr=2e-3, weight_decay=self.weightDecay)
            
            # Save computation times for this trajectory
            self.original_compute_times[trajectorie] = trajectory_original_time
            self.cpd_unsupervised_compute_times[trajectorie] = trajectory_cpd_unsupervised_time
            self.only_unsupervised_compute_times[trajectorie] = trajectory_only_unsupervised_time
            
            loss_fn = torch.nn.MSELoss(reduction='mean')  # Fix: Use consistent calculation method as KF
            
            # Calculate MSE for Original model
            # Overall MSE
            self.MSE_Original_linear_arr[trajectorie] = loss_fn(self.state_original[trajectorie,:,:].cpu().detach(), x_true[trajectorie,:,:].cpu().detach()).item()
            
            # Dimension-wise MSE
            # Position (dimension 0)
            self.MSE_Original_linear_arr_by_dim[trajectorie,0] = loss_fn(self.state_original[trajectorie,0:1,:].cpu().detach(), x_true[trajectorie,0:1,:].cpu().detach()).item()
            
            # Velocity (dimension 1) if available
            if self.ssModel.m > 1:
                self.MSE_Original_linear_arr_by_dim[trajectorie,1] = loss_fn(self.state_original[trajectorie,1:2,:].cpu().detach(), x_true[trajectorie,1:2,:].cpu().detach()).item()
            
            # Acceleration (dimension 2) if available
            if self.ssModel.m > 2:
                self.MSE_Original_linear_arr_by_dim[trajectorie,2] = loss_fn(self.state_original[trajectorie,2:3,:].cpu().detach(), x_true[trajectorie,2:3,:].cpu().detach()).item()
            
            # Calculate MSE for CPD-Unsupervised model
            # Overall MSE
            self.MSE_Ours_linear_arr[trajectorie] = loss_fn(self.state_predictions[trajectorie,:,:].cpu().detach(), x_true[trajectorie,:,:].cpu().detach()).item()
            
            # Dimension-wise MSE
            # Position (dimension 0)
            self.MSE_Ours_linear_arr_by_dim[trajectorie,0] = loss_fn(self.state_predictions[trajectorie,0:1,:].cpu().detach(), x_true[trajectorie,0:1,:].cpu().detach()).item()
            
            # Velocity (dimension 1) if available
            if self.ssModel.m > 1:
                self.MSE_Ours_linear_arr_by_dim[trajectorie,1] = loss_fn(self.state_predictions[trajectorie,1:2,:].cpu().detach(), x_true[trajectorie,1:2,:].cpu().detach()).item()
            
            # Acceleration (dimension 2) if available
            if self.ssModel.m > 2:
                self.MSE_Ours_linear_arr_by_dim[trajectorie,2] = loss_fn(self.state_predictions[trajectorie,2:3,:].cpu().detach(), x_true[trajectorie,2:3,:].cpu().detach()).item()
            
            # Calculate MSE for Pure Unsupervised model
            # Overall MSE
            self.MSE_Unsupervised_linear_arr[trajectorie] = loss_fn(self.state_unsupervised[trajectorie,:,:].cpu().detach(), x_true[trajectorie,:,:].cpu().detach()).item()
            
            # Dimension-wise MSE
            # Position (dimension 0)
            self.MSE_Unsupervised_linear_arr_by_dim[trajectorie,0] = loss_fn(self.state_unsupervised[trajectorie,0:1,:].cpu().detach(), x_true[trajectorie,0:1,:].cpu().detach()).item()
            
            # Velocity (dimension 1) if available
            if self.ssModel.m > 1:
                self.MSE_Unsupervised_linear_arr_by_dim[trajectorie,1] = loss_fn(self.state_unsupervised[trajectorie,1:2,:].cpu().detach(), x_true[trajectorie,1:2,:].cpu().detach()).item()
            
            # Acceleration (dimension 2) if available
            if self.ssModel.m > 2:
                self.MSE_Unsupervised_linear_arr_by_dim[trajectorie,2] = loss_fn(self.state_unsupervised[trajectorie,2:3,:].cpu().detach(), x_true[trajectorie,2:3,:].cpu().detach()).item()
            
            
        # Calculate time statistics
        original_mean_time = self.original_compute_times.mean().item()
        original_std_time = self.original_compute_times.std().item()
        cpd_mean_time = self.cpd_unsupervised_compute_times.mean().item()
        cpd_std_time = self.cpd_unsupervised_compute_times.std().item()
        unsupervised_mean_time = self.only_unsupervised_compute_times.mean().item()
        unsupervised_std_time = self.only_unsupervised_compute_times.std().item()
        
        # Calculate averages and dB values for both overall and dimension-wise MSE
        
        # Overall MSE
        MSE_Original_linear_avg = torch.mean(self.MSE_Original_linear_arr)
        MSE_Original_dB_avg = 10 * torch.log10(MSE_Original_linear_avg)
        MSE_Original_linear_std = torch.std(self.MSE_Original_linear_arr, unbiased=True)
        Original_std_dB = 10 * torch.log10(MSE_Original_linear_std + MSE_Original_linear_avg) - MSE_Original_dB_avg
        
        MSE_Ours_linear_avg = torch.mean(self.MSE_Ours_linear_arr)
        MSE_Ours_dB_avg = 10 * torch.log10(MSE_Ours_linear_avg)
        MSE_Ours_linear_std = torch.std(self.MSE_Ours_linear_arr, unbiased=True)
        Ours_std_dB = 10 * torch.log10(MSE_Ours_linear_std + MSE_Ours_linear_avg) - MSE_Ours_dB_avg
        
        MSE_Unsupervised_linear_avg = torch.mean(self.MSE_Unsupervised_linear_arr)
        MSE_Unsupervised_dB_avg = 10 * torch.log10(MSE_Unsupervised_linear_avg)
        MSE_Unsupervised_linear_std = torch.std(self.MSE_Unsupervised_linear_arr, unbiased=True)
        Unsupervised_std_dB = 10 * torch.log10(MSE_Unsupervised_linear_std + MSE_Unsupervised_linear_avg) - MSE_Unsupervised_dB_avg
        
        # Dimension-wise MSE
        MSE_Original_linear_avg_by_dim = torch.mean(self.MSE_Original_linear_arr_by_dim, dim=0)
        MSE_Original_dB_avg_by_dim = 10 * torch.log10(MSE_Original_linear_avg_by_dim)
        MSE_Original_linear_std_by_dim = torch.std(self.MSE_Original_linear_arr_by_dim, dim=0, unbiased=True)
        Original_std_dB_by_dim = 10 * torch.log10(MSE_Original_linear_std_by_dim + MSE_Original_linear_avg_by_dim) - MSE_Original_dB_avg_by_dim
        
        MSE_Ours_linear_avg_by_dim = torch.mean(self.MSE_Ours_linear_arr_by_dim, dim=0)
        MSE_Ours_dB_avg_by_dim = 10 * torch.log10(MSE_Ours_linear_avg_by_dim)
        MSE_Ours_linear_std_by_dim = torch.std(self.MSE_Ours_linear_arr_by_dim, dim=0, unbiased=True)
        Ours_std_dB_by_dim = 10 * torch.log10(MSE_Ours_linear_std_by_dim + MSE_Ours_linear_avg_by_dim) - MSE_Ours_dB_avg_by_dim
        
        MSE_Unsupervised_linear_avg_by_dim = torch.mean(self.MSE_Unsupervised_linear_arr_by_dim, dim=0)
        MSE_Unsupervised_dB_avg_by_dim = 10 * torch.log10(MSE_Unsupervised_linear_avg_by_dim)
        MSE_Unsupervised_linear_std_by_dim = torch.std(self.MSE_Unsupervised_linear_arr_by_dim, dim=0, unbiased=True)
        Unsupervised_std_dB_by_dim = 10 * torch.log10(MSE_Unsupervised_linear_std_by_dim + MSE_Unsupervised_linear_avg_by_dim) - MSE_Unsupervised_dB_avg_by_dim
        
        # Display comprehensive performance and computation time statistics
        print('\n' + '='*90)
        print('                    Algorithm Performance and Computation Time Statistics                    ')
        print('='*90)
        
        # Calculate efficiency rating
        fastest_time = min(original_mean_time, cpd_mean_time, unsupervised_mean_time)
        original_efficiency = fastest_time/original_mean_time
        cpd_efficiency = fastest_time/cpd_mean_time
        unsupervised_efficiency = fastest_time/unsupervised_mean_time
        
        # Overall MSE results
        print(f'{"Algorithm Name":<20} {"Overall MSE (dB)":<15} {"STD (dB)":<12} {"Avg Time (s)":<12} {"Time Std":<12} {"Efficiency":<12}')
        print('-'*90)
        print(f'{"Original KalmanNet":<20} {MSE_Original_dB_avg.item():<15.4f} {Original_std_dB.item():<12.4f} {original_mean_time:<12.4f} {original_std_time:<12.4f} {original_efficiency:<12.2f}')
        print(f'{"CPDNet-Unsupervised":<20} {MSE_Ours_dB_avg.item():<15.4f} {Ours_std_dB.item():<12.4f} {cpd_mean_time:<12.4f} {cpd_std_time:<12.4f} {cpd_efficiency:<12.2f}')
        print(f'{"Pure Unsupervised":<20} {MSE_Unsupervised_dB_avg.item():<15.4f} {Unsupervised_std_dB.item():<12.4f} {unsupervised_mean_time:<12.4f} {unsupervised_std_time:<12.4f} {unsupervised_efficiency:<12.2f}')
        
        # Dimension-wise MSE results
        print('\n' + '='*90)
        print('                    Algorithm Performance by Dimension                    ')
        print('='*90)
        print(f'{"Algorithm Name":<20} {"Dimension":<12} {"MSE (dB)":<12} {"STD (dB)":<12}')
        print('-'*90)
        
        dimensions = ['Position', 'Velocity', 'Acceleration']
        for dim_idx in range(min(3, self.ssModel.m)):  # Only show available dimensions
            dim_name = dimensions[dim_idx]
            print(f'{"Original KalmanNet":<20} {dim_name:<12} {MSE_Original_dB_avg_by_dim[dim_idx].item():<12.4f} {Original_std_dB_by_dim[dim_idx].item():<12.4f}')
            print(f'{"CPDNet-Unsupervised":<20} {dim_name:<12} {MSE_Ours_dB_avg_by_dim[dim_idx].item():<12.4f} {Ours_std_dB_by_dim[dim_idx].item():<12.4f}')
            print(f'{"Pure Unsupervised":<20} {dim_name:<12} {MSE_Unsupervised_dB_avg_by_dim[dim_idx].item():<12.4f} {Unsupervised_std_dB_by_dim[dim_idx].item():<12.4f}')
            if dim_idx < min(3, self.ssModel.m) - 1:
                print()  # Add blank line between dimensions
        
        print('='*90)


            
            
            # plt.figure(figsize=(10, 6))
            # plt.plot(self.state_predictions[trajectorie, 0, :].cpu().detach().numpy(), label='y_out_online')
            # plt.plot(self.state_original[trajectorie, 0, :].cpu().detach().numpy(), label='y_out_original')
            # plt.plot(x_true[trajectorie, 0, :].cpu().detach().numpy(), label='x_true')
            # plt.plot(self.state_unsupervised[trajectorie, 0, :].cpu().detach().numpy(), label='y_out_unsupervised')
            # plt.title('y_out_online,y_out_original,y_out_unsupervised, x_true')
            # plt.legend()
            # plt.show()
    
    def NNTrain_lor(self, SysModel,y_observation,x_true,train_init):
        
        self.N_T = len(y_observation)  # Number of trajectories
        self.stride = 5
        

        ###############################
        ### Initialize models and optimizers (align with Unsupervised_CPD_Online) ###
        ###############################
        # Baseline model: inference only, no training
        self.KNetmodel.eval()
        self.KNetmodel.batch_size = 1
        # CPDNet: detection model, inference only
        self.CPDmodel.eval()
        self.CPDmodel.batch_size = 1

        # Create independent copies for CPD-triggered training and continuous self-supervised training
        original_model = copy.deepcopy(self.KNetmodel)
        original_model.train()
        original_model.batch_size = 1

        unsupervised_model = copy.deepcopy(self.KNetmodel)
        unsupervised_model.train()
        unsupervised_model.batch_size = 1

        # Ensure all three models use the provided system dynamics (consistent with Unsupervised_CPD_Online)
        self.KNetmodel.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)
        original_model.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)
        unsupervised_model.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)

        # Create separate optimizers for the two trainable branches
        self.optimizer_original = torch.optim.Adam(original_model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        self.optimizer_unsupervised = torch.optim.Adam(unsupervised_model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

        # Result buffers (same shapes as in Unsupervised_CPD_Online)
        self.observation_predictions = torch.empty((self.N_T, self.ssModel.n, SysModel.T))
        self.state_predictions = torch.empty((self.N_T, self.ssModel.m, SysModel.T))
        self.observation_original = torch.empty((self.N_T, self.ssModel.n, SysModel.T))
        self.state_original = torch.empty((self.N_T, self.ssModel.m, SysModel.T))
        self.observation_unsupervised = torch.empty((self.N_T, self.ssModel.n, SysModel.T))
        self.state_unsupervised = torch.empty((self.N_T, self.ssModel.m, SysModel.T))
        
        # MSE storage arrays for final averaging
        self.MSE_CPD_linear_arr = torch.empty((self.N_T,))
        self.MSE_Original_KNet_linear_arr = torch.empty((self.N_T,))
        self.MSE_Unsupervised_linear_arr = torch.empty((self.N_T,))
        
        # Start looping over trajectories
        for trajectorie in range(self.N_T):
            print('Trajectory: ', trajectorie + 1, '/', self.N_T)

            # Load the next trajectory
            y_training = torch.unsqueeze(y_observation[trajectorie, :, :],0).requires_grad_(True).to(self.device)

            # Initialize state
            init_state = torch.unsqueeze(train_init[trajectorie,:,:],0).to(self.device)
            self.KNetmodel.InitSequence(init_state, self.args.T)
            original_model.InitSequence(init_state, self.args.T)
            unsupervised_model.InitSequence(init_state, self.args.T)
            # Init hidden
            self.KNetmodel.init_hidden_KNet()
            original_model.init_hidden_KNet()
            unsupervised_model.init_hidden_KNet()
            # Calculate the number of strides required
            number_of_stride = int(self.ssModel.T / self.stride)
            
            cpd_input = torch.zeros(1, 1, SysModel.T).to(self.device)
            counter = 0
            

            # Go through the whole trajectory step by step
            for stride in trange(number_of_stride):
                observation = y_training[:, :, (stride * self.stride):(stride * self.stride + self.stride)].to(self.device)
                
                x_out_online = torch.empty(1, self.ssModel.m, self.stride).to(self.device)
                y_out_online = torch.zeros(1, self.ssModel.n, self.stride).to(self.device)
                
                x_out_original = torch.empty(1, self.ssModel.m, self.stride).to(self.device)
                y_out_original = torch.zeros(1, self.ssModel.n, self.stride).to(self.device)
                
                x_out_unsupervised = torch.empty(1, self.ssModel.m, self.stride).to(self.device)
                y_out_unsupervised = torch.zeros(1, self.ssModel.n, self.stride).to(self.device)
                # Consistent with Unsupervised_CPD_Online: use previous window posterior as next window initial state
                self.KNetmodel.InitSequence(self.KNetmodel.m1x_posterior.clone().detach(), SysModel.T)
                original_model.InitSequence(original_model.m1x_posterior.clone().detach(), SysModel.T)
                unsupervised_model.InitSequence(unsupervised_model.m1x_posterior.clone().detach(), SysModel.T)
                self.KNetmodel.init_hidden_KNet()
                original_model.init_hidden_KNet()
                unsupervised_model.init_hidden_KNet()

                # Forward pass (baseline not trained; other two branches will be trained)
                for i in range(self.stride):
                    x_out_online[:, :, i] = torch.squeeze(self.KNetmodel(torch.unsqueeze(observation[:, :, i], dim=2)))
                    y_out_online[:, :, i] = torch.squeeze(self.KNetmodel.m1y)
                    x_out_original[:, :, i] = torch.squeeze(original_model(torch.unsqueeze(observation[:, :, i], dim=2)))
                    y_out_original[:, :, i] = torch.squeeze(original_model.m1y)
                    x_out_unsupervised[:, :, i] = torch.squeeze(unsupervised_model(torch.unsqueeze(observation[:, :, i], dim=2)))
                    y_out_unsupervised[:, :, i] = torch.squeeze(unsupervised_model.m1y)

                self.observation_predictions[trajectorie, :,
                (stride * self.stride):(stride * self.stride + self.stride)] = y_out_online.detach()
                self.state_predictions[trajectorie, :,
                (stride * self.stride):(stride * self.stride + self.stride)] = x_out_online.detach()
                self.observation_original[trajectorie, :,
                (stride * self.stride):(stride * self.stride + self.stride)] = y_out_original.detach()
                self.state_original[trajectorie, :,
                (stride * self.stride):(stride * self.stride + self.stride)] = x_out_original.detach()
                self.observation_unsupervised[trajectorie, :,
                (stride * self.stride):(stride * self.stride + self.stride)] = y_out_unsupervised.detach()
                self.state_unsupervised[trajectorie, :,
                (stride * self.stride):(stride * self.stride + self.stride)] = x_out_unsupervised.detach()
                
                if stride > 0:
                    for t in range(self.stride):
                        cpd_input[:, :, (stride-1)*self.stride+t] = cpd_dataset_process_lor(torch.unsqueeze(self.observation_predictions[trajectorie, :,(stride * self.stride-self.stride+t):(stride * self.stride+t)],dim=0).to(self.device),
                                                                    y_training[:, :, (stride * self.stride-self.stride+t):(stride * self.stride+t)])
                    cpd_out_tmp = self.CPDmodel(cpd_input[:, :, (stride-1)*self.stride:(stride-1)*self.stride+self.stride])
                    new_lr = (3 * cpd_out_tmp * self.learningRate).item()
                    for param_group in self.optimizer_original.param_groups:
                        param_group['lr'] = new_lr

                # Supervised branch training triggered by CPD
                if stride > 0 and (cpd_out_tmp > 0.5):
                    LOSS_original = self.loss_fn(y_out_original, observation)
                    self.optimizer_original.zero_grad()
                    LOSS_original.backward()
                    self.optimizer_original.step()

                # Continuous self-supervised branch training
                LOSS_unsupervised = self.loss_fn(y_out_unsupervised, observation)
                self.optimizer_unsupervised.zero_grad()
                LOSS_unsupervised.backward()
                self.optimizer_unsupervised.step()
                
                #  Print statistics every 10% of a trajectory
                # counter += 1
                # if counter % max(int(number_of_stride/10),1) == 0:
                #     print('Training itt:', stride + 1, '/', number_of_stride, ',OBS MSE:',
                #           10 * torch.log10(LOSS).item(), '[dB]')
                    
                del y_out_online, observation, LOSS_unsupervised
            
            # Calculate MSE for this trajectory and store for later averaging
            loss_fn = torch.nn.MSELoss(reduction='mean')
            self.MSE_CPD_linear_arr[trajectorie] = loss_fn(self.state_original[trajectorie,:,:].cpu().detach(), x_true[trajectorie,:,:].cpu().detach()).item()
            self.MSE_Original_KNet_linear_arr[trajectorie] = loss_fn(self.state_predictions[trajectorie,:,:].cpu().detach(), x_true[trajectorie,:,:].cpu().detach()).item()
            self.MSE_Unsupervised_linear_arr[trajectorie] = loss_fn(self.state_unsupervised[trajectorie,:,:].cpu().detach(), x_true[trajectorie,:,:].cpu().detach()).item()
        
        # Calculate average MSE across all trajectories (following lines 298-299 pattern)
        MSE_CPD_linear_avg = torch.mean(self.MSE_CPD_linear_arr)
        MSE_CPD_dB_avg = 10 * torch.log10(MSE_CPD_linear_avg)
        MSE_Original_KNet_linear_avg = torch.mean(self.MSE_Original_KNet_linear_arr)
        MSE_Original_KNet_dB_avg = 10 * torch.log10(MSE_Original_KNet_linear_avg)
        MSE_Unsupervised_linear_avg = torch.mean(self.MSE_Unsupervised_linear_arr)
        MSE_Unsupervised_dB_avg = 10 * torch.log10(MSE_Unsupervised_linear_avg)
        
        # Print average MSE results
        print('Average MSE CPDUnsupervised:', MSE_CPD_dB_avg.item(), '[dB]')
        print('Average MSE Original KNet:', MSE_Original_KNet_dB_avg.item(), '[dB]')
        print('Average MSE Unsupervised:', MSE_Unsupervised_dB_avg.item(), '[dB]')
        
        # Randomly select one trajectory for plotting
        selected_traj = random.randint(0, self.N_T - 1)
        print(f'Plotting trajectory {selected_traj + 1}/{self.N_T}')
        
        # Create a single 3D subplot showing Original (baseline), Ours (CPD-triggered), and True trajectories
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Original (baseline, no updates)
        ax.plot(self.state_predictions[selected_traj, 0, :].cpu().detach().numpy(),
                self.state_predictions[selected_traj, 1, :].cpu().detach().numpy(),
                self.state_predictions[selected_traj, 2, :].cpu().detach().numpy(),
                label='original KNet')
        # Ours (CPD-triggered branch)
        ax.plot(self.state_original[selected_traj, 0, :].cpu().detach().numpy(),
                self.state_original[selected_traj, 1, :].cpu().detach().numpy(),
                self.state_original[selected_traj, 2, :].cpu().detach().numpy(),
                label='ours')
        # Unsupervised
        ax.plot(self.state_unsupervised[selected_traj, 0, :].cpu().detach().numpy(),
                self.state_unsupervised[selected_traj, 1, :].cpu().detach().numpy(),
                self.state_unsupervised[selected_traj, 2, :].cpu().detach().numpy(),
                label='unsupervised')
        # True
        ax.plot(x_true[selected_traj, 0, :].cpu().detach().numpy(),
                x_true[selected_traj, 1, :].cpu().detach().numpy(),
                x_true[selected_traj, 2, :].cpu().detach().numpy(),
                label='x_true')
        ax.set_title(f'3D Trajectories (Trajectory {selected_traj + 1}): Original vs Ours vs Unsupervised vs True')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        
        plt.tight_layout()
        plt.show()


            # # X-axis comparison (unsupervised vs online vs true)
            # fig_x = plt.figure(figsize=(12, 4))
            # plt.plot(self.state_predictions[trajectorie, 0, :].cpu().detach().numpy(), label='y_out_online')
            # plt.plot(self.state_unsupervised[trajectorie, 0, :].cpu().detach().numpy(), label='y_out_unsupervised')
            # plt.plot(x_true[trajectorie, 0, :].cpu().detach().numpy(), label='x_true')
            # plt.title('X-axis Comparison')
            # plt.xlabel('Time')
            # plt.ylabel('X')
            # plt.legend()
            # plt.show()

            # # Y-axis comparison (unsupervised vs online vs true)
            # fig_y = plt.figure(figsize=(12, 4))
            # plt.plot(self.state_predictions[trajectorie, 1, :].cpu().detach().numpy(), label='y_out_online')
            # plt.plot(self.state_unsupervised[trajectorie, 1, :].cpu().detach().numpy(), label='y_out_unsupervised')
            # plt.plot(x_true[trajectorie, 1, :].cpu().detach().numpy(), label='x_true')
            # plt.title('Y-axis Comparison')
            # plt.xlabel('Time')
            # plt.ylabel('Y')
            # plt.legend()
            # plt.show()

            # # Z-axis comparison (unsupervised vs online vs true)
            # fig_z = plt.figure(figsize=(12, 4))
            # plt.plot(self.state_predictions[trajectorie, 2, :].cpu().detach().numpy(), label='y_out_online')
            # plt.plot(self.state_unsupervised[trajectorie, 2, :].cpu().detach().numpy(), label='y_out_unsupervised')
            # plt.plot(x_true[trajectorie, 2, :].cpu().detach().numpy(), label='x_true')
            # plt.title('Z-axis Comparison')
            # plt.xlabel('Time')
            # plt.ylabel('Z')
            # plt.legend()
            # plt.show()

