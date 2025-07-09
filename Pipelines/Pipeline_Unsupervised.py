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
        ### Training Sequence Batch ###
        ###############################
        self.optimizer.zero_grad()
        # KNet Training Mode
        self.KNetmodel.train()
        self.KNetmodel.batch_size = 1
        # CPDNet Test Mode
        self.CPDmodel.eval()
        self.CPDmodel.batch_size = 1
        # Init Hidden State
        self.KNetmodel.init_hidden_KNet()
        # Allocate estimate array
        self.observation_predictions = torch.empty((self.N_T, self.ssModel.n, SysModel.T))
        self.state_predictions = torch.empty((self.N_T, self.ssModel.m, SysModel.T))
        self.observation_original = torch.empty((self.N_T, self.ssModel.n, SysModel.T))
        self.state_original = torch.empty((self.N_T, self.ssModel.m, SysModel.T))
        self.observation_unsupervised = torch.empty((self.N_T, self.ssModel.n, SysModel.T))
        self.state_unsupervised = torch.empty((self.N_T, self.ssModel.m, SysModel.T))
        
        self.MSE_Original_dB = torch.empty((self.N_T, 1, SysModel.T))
        self.STD_Original_dB = torch.empty((self.N_T, 1, SysModel.T))
        self.MSE_Unsupervised_dB = torch.empty((self.N_T, 1, SysModel.T))
        self.STD_Unsupervised_dB = torch.empty((self.N_T, 1, SysModel.T))
        self.MSE_Ours_dB = torch.empty((self.N_T, 1, SysModel.T))
        self.STD_Ours_dB = torch.empty((self.N_T, 1, SysModel.T))

        # Copy to restore the NN to its original state for each trajectory
        original_model = copy.deepcopy(self.KNetmodel)
        original_model.eval()
        original_model.batch_size = 1
        original_model.init_hidden_KNet()
        
        unsupervised_model = copy.deepcopy(self.KNetmodel)
        unsupervised_model.train()
        unsupervised_model.batch_size = 1
        unsupervised_model.init_hidden_KNet()
        self.optimizer_unsupervised = torch.optim.Adam(unsupervised_model.parameters(), lr=2e-3, weight_decay=self.weightDecay)
        
        # Start looping over trajectories
        for trajectorie in range(self.N_T):
            print('Trajectory: ', trajectorie + 1, '/', self.N_T)

            # Initialize time counters for this trajectory
            trajectory_original_time = 0.0
            trajectory_cpd_unsupervised_time = 0.0
            trajectory_only_unsupervised_time = 0.0

            ###############################
            ### Training Sequence Batch ###
            ###############################
            self.reTraining = False
            # Reset optimizer
            self.ResetOptimizer()
            # Training Mode
            self.KNetmodel.train()

            # Load the next trajectory
            y_training = torch.unsqueeze(y_observation[trajectorie, :, :],0).requires_grad_(True).to(self.device)

            # Initialize state
            self.KNetmodel.InitSequence(torch.unsqueeze(train_init[trajectorie,:,:],0).to(self.device),self.args.T)
            original_model.InitSequence(torch.unsqueeze(train_init[trajectorie,:,:],0).to(self.device),self.args.T)
            unsupervised_model.InitSequence(torch.unsqueeze(train_init[trajectorie,:,:],0).to(self.device),self.args.T)
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
                # Initialize training mode
                self.KNetmodel.train()
                original_model.eval()
                unsupervised_model.train()
                # Initialize the informations
                self.KNetmodel.InitSequence(self.KNetmodel.m1x_posterior.clone().detach(),SysModel.T)
                original_model.InitSequence(original_model.m1x_posterior.clone().detach(),SysModel.T)
                unsupervised_model.InitSequence(unsupervised_model.m1x_posterior.clone().detach(),SysModel.T)
                self.KNetmodel.init_hidden_KNet()
                original_model.init_hidden_KNet()
                unsupervised_model.init_hidden_KNet()

                # Start training
                for i in range(self.stride):
                    # CPDUnsupervised algorithm computation time
                    start_time = time.time()
                    x_out_online[:, :, i] = torch.squeeze(self.KNetmodel(torch.unsqueeze(observation[:, :, i],dim=2)))
                    y_out_online[:, :, i] = torch.squeeze(self.KNetmodel.m1y)
                    trajectory_cpd_unsupervised_time += time.time() - start_time
                    
                    # Original algorithm computation time
                    start_time = time.time()
                    x_out_original[:, :, i] = torch.squeeze(original_model(torch.unsqueeze(observation[:, :, i],dim=2)))
                    y_out_original[:, :, i] = torch.squeeze(original_model.m1y)
                    trajectory_original_time += time.time() - start_time
                    
                    # Only Unsupervised algorithm computation time
                    start_time = time.time()
                    x_out_unsupervised[:, :, i] = torch.squeeze(unsupervised_model(torch.unsqueeze(observation[:, :, i],dim=2)))
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
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr

                    if cpd_out_tmp > 0.65:
                        LOSS = self.loss_fn(y_out_online,observation)
                        self.optimizer.zero_grad()
                        LOSS.backward()
                        self.optimizer.step()
                        
                        del LOSS
                
                LOSS_unsupervised = self.loss_fn(y_out_unsupervised,observation)
                self.optimizer_unsupervised.zero_grad()
                LOSS_unsupervised.backward()
                self.optimizer_unsupervised.step()
                    
                del y_out_online,observation,LOSS_unsupervised
            # Reset the optimizer for the next trajectory
            self.ResetOptimizer()
            
            # Save computation times for this trajectory
            self.original_compute_times[trajectorie] = trajectory_original_time
            self.cpd_unsupervised_compute_times[trajectorie] = trajectory_cpd_unsupervised_time
            self.only_unsupervised_compute_times[trajectorie] = trajectory_only_unsupervised_time
            
            loss_fn = torch.nn.MSELoss(reduction='none')
            MSE_Original_Linear = loss_fn(self.state_original[trajectorie,:,:].cpu().detach(), x_true[trajectorie,:,:].cpu().detach())
            MSE_Original_dB = 10 * torch.log10(MSE_Original_Linear)
            STD_Original = torch.std(MSE_Original_dB, unbiased=True)
            STD_Original_dB = 10*torch.log10(STD_Original)
            self.MSE_Original_dB[trajectorie,:,:] = MSE_Original_dB.mean().item()
            self.STD_Original_dB[trajectorie,:,:] = STD_Original_dB.mean().item()
            
            MSE_CPDUnsupervised_Linear = loss_fn(self.state_predictions[trajectorie,:,:].cpu().detach(), x_true[trajectorie,:,:].cpu().detach())
            MSE_CPDUnsupervised_dB = 10 * torch.log10(MSE_CPDUnsupervised_Linear)
            STD_CPDUnsupervised = torch.std(MSE_CPDUnsupervised_dB, unbiased=True)
            STD_CPDUnsupervised_dB = 10*torch.log10(STD_CPDUnsupervised)
            self.MSE_Ours_dB[trajectorie,:,:] = MSE_CPDUnsupervised_dB.mean().item()
            self.STD_Ours_dB[trajectorie,:,:] = STD_CPDUnsupervised_dB.mean().item()
            
            MSE_Unsupervised_Linear = loss_fn(self.state_unsupervised[trajectorie,:,:].cpu().detach(), x_true[trajectorie,:,:].cpu().detach())
            MSE_Unsupervised_dB = 10 * torch.log10(MSE_Unsupervised_Linear)
            STD_Unsupervised = torch.std(MSE_Unsupervised_dB, unbiased=True)
            STD_Unsupervised_dB = 10*torch.log10(STD_Unsupervised)
            self.MSE_Unsupervised_dB[trajectorie,:,:] = MSE_Unsupervised_dB.mean().item()
            self.STD_Unsupervised_dB[trajectorie,:,:] = STD_Unsupervised_dB.mean().item()
            
            
        # Calculate time statistics
        original_mean_time = self.original_compute_times.mean().item()
        original_std_time = self.original_compute_times.std().item()
        cpd_mean_time = self.cpd_unsupervised_compute_times.mean().item()
        cpd_std_time = self.cpd_unsupervised_compute_times.std().item()
        unsupervised_mean_time = self.only_unsupervised_compute_times.mean().item()
        unsupervised_std_time = self.only_unsupervised_compute_times.std().item()
        
        # Display comprehensive performance and computation time statistics
        print('\n' + '='*90)
        print('                        算法性能与计算时间综合统计结果                        ')
        print('='*90)
        print(f'{"算法名称":<20} {"MSE (dB)":<12} {"STD (dB)":<12} {"平均时间 (秒)":<12} {"时间标准差":<12} {"效率评级":<12}')
        print('-'*90)
        
        # 计算效率评级
        fastest_time = min(original_mean_time, cpd_mean_time, unsupervised_mean_time)
        original_efficiency = fastest_time/original_mean_time
        cpd_efficiency = fastest_time/cpd_mean_time
        unsupervised_efficiency = fastest_time/unsupervised_mean_time
        
        print(f'{"原始 KalmanNet":<20} {self.MSE_Original_dB.mean().item():<12.4f} {self.STD_Original_dB.mean().item():<12.4f} {original_mean_time:<12.4f} {original_std_time:<12.4f} {original_efficiency:<12.2f}')
        print(f'{"CPDNet-无监督":<20} {self.MSE_Ours_dB.mean().item():<12.4f} {self.STD_Ours_dB.mean().item():<12.4f} {cpd_mean_time:<12.4f} {cpd_std_time:<12.4f} {cpd_efficiency:<12.2f}')
        print(f'{"纯无监督":<20} {self.MSE_Unsupervised_dB.mean().item():<12.4f} {self.STD_Unsupervised_dB.mean().item():<12.4f} {unsupervised_mean_time:<12.4f} {unsupervised_std_time:<12.4f} {unsupervised_efficiency:<12.2f}')
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
        ### Training Sequence Batch ###
        ###############################
        self.optimizer.zero_grad()
        # KNet Training Mode
        self.KNetmodel.train()
        self.KNetmodel.batch_size = 1
        # CPDNet Test Mode
        self.CPDmodel.eval()
        self.CPDmodel.batch_size = 1
        # Init Hidden State
        self.KNetmodel.init_hidden_KNet()
        # Allocate estimate array
        self.observation_predictions = torch.empty((self.N_T, self.ssModel.n, SysModel.T))
        self.state_predictions = torch.empty((self.N_T, self.ssModel.m, SysModel.T))
        self.observation_original = torch.empty((self.N_T, self.ssModel.n, SysModel.T))
        self.state_original = torch.empty((self.N_T, self.ssModel.m, SysModel.T))
        self.observation_unsupervised = torch.empty((self.N_T, self.ssModel.n, SysModel.T))
        self.state_unsupervised = torch.empty((self.N_T, self.ssModel.m, SysModel.T))

        # Copy to restore the NN to its original state for each trajectory
        original_model = copy.deepcopy(self.KNetmodel)
        original_model.eval()
        original_model.batch_size = 1
        original_model.init_hidden_KNet()
        
        unsupervised_model = copy.deepcopy(self.KNetmodel)
        unsupervised_model.train()
        unsupervised_model.batch_size = 1
        unsupervised_model.init_hidden_KNet()
        self.optimizer_unsupervised = torch.optim.Adam(unsupervised_model.parameters(), lr=1e-3, weight_decay=self.weightDecay)
        
        # Start looping over trajectories
        for trajectorie in range(self.N_T):
            print('Trajectory: ', trajectorie + 1, '/', self.N_T)

            ###############################
            ### Training Sequence Batch ###
            ###############################
            self.reTraining = False
            # Reset optimizer
            self.ResetOptimizer()
            # Training Mode
            self.KNetmodel.train()

            # Load the next trajectory
            y_training = torch.unsqueeze(y_observation[trajectorie, :, :],0).requires_grad_(True).to(self.device)

            # Initialize state
            self.KNetmodel.InitSequence(torch.unsqueeze(train_init[trajectorie,:,:],0).to(self.device),self.args.T)
            original_model.InitSequence(torch.unsqueeze(train_init[trajectorie,:,:],0).to(self.device),self.args.T)
            unsupervised_model.InitSequence(torch.unsqueeze(train_init[trajectorie,:,:],0).to(self.device),self.args.T)
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
                # Initialize training mode
                self.KNetmodel.train()
                original_model.eval()
                unsupervised_model.train()
                # Initialize the informations
                self.KNetmodel.InitSequence(self.KNetmodel.m1x_posterior.clone().detach(),SysModel.T)
                original_model.InitSequence(original_model.m1x_posterior.clone().detach(),SysModel.T)
                unsupervised_model.InitSequence(unsupervised_model.m1x_posterior.clone().detach(),SysModel.T)
                self.KNetmodel.init_hidden_KNet()
                original_model.init_hidden_KNet()
                unsupervised_model.init_hidden_KNet()

                # Start training
                for i in range(self.stride):
                    x_out_online[:, :, i] = torch.squeeze(self.KNetmodel(torch.unsqueeze(observation[:, :, i],dim=2)))
                    y_out_online[:, :, i] = torch.squeeze(self.KNetmodel.m1y)
                    x_out_original[:, :, i] = torch.squeeze(original_model(torch.unsqueeze(observation[:, :, i],dim=2)))
                    y_out_original[:, :, i] = torch.squeeze(original_model.m1y)
                    x_out_unsupervised[:, :, i] = torch.squeeze(unsupervised_model(torch.unsqueeze(observation[:, :, i],dim=2)))
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
                
                if stride >0:
                    for t in range(self.stride):
                        cpd_input[:, :, (stride-1)*self.stride+t] = cpd_dataset_process_lor(torch.unsqueeze(self.observation_predictions[trajectorie, :,(stride * self.stride-self.stride+t):(stride * self.stride+t)],dim=0).to(self.device),
                                                                    y_training[:, :, (stride * self.stride-self.stride+t):(stride * self.stride+t)])
                    cpd_out_tmp = self.CPDmodel(cpd_input[:, :, (stride-1)*self.stride:(stride-1)*self.stride+self.stride])
                    new_lr = (3*cpd_out_tmp*self.learningRate).item()
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr


                LOSS = self.loss_fn(y_out_online,observation)
                self.optimizer.zero_grad()
                LOSS.backward()
                self.optimizer.step()
                
                LOSS_unsupervised = self.loss_fn(y_out_unsupervised,observation)
                self.optimizer_unsupervised.zero_grad()
                LOSS_unsupervised.backward()
                self.optimizer_unsupervised.step()
                
                #  Print statistics every 10% of a trajectory
                # counter += 1
                # if counter % max(int(number_of_stride/10),1) == 0:
                #     print('Training itt:', stride + 1, '/', number_of_stride, ',OBS MSE:',
                #           10 * torch.log10(LOSS).item(), '[dB]')
                    
                del LOSS,y_out_online,observation,LOSS_unsupervised
            # Reset the optimizer for the next trajectory
            self.ResetOptimizer()
            self.optimizer_unsupervised = torch.optim.Adam(unsupervised_model.parameters(), lr=2e-3, weight_decay=self.weightDecay)
        
            
            loss_fn = torch.nn.MSELoss(reduction='none')
            MSE_Original_Linear = loss_fn(self.state_original[trajectorie,:,:].cpu().detach(), x_true[trajectorie,:,:].cpu().detach())
            MSE_Original_dB = 10 * torch.log10(MSE_Original_Linear)
            MSE_CPDUnsupervised_Linear = loss_fn(self.state_predictions[trajectorie,:,:].cpu().detach(), x_true[trajectorie,:,:].cpu().detach())
            MSE_CPDUnsupervised_dB = 10 * torch.log10(MSE_CPDUnsupervised_Linear)
            MSE_Unsupervised_Linear = loss_fn(self.state_unsupervised[trajectorie,:,:].cpu().detach(), x_true[trajectorie,:,:].cpu().detach())
            MSE_Unsupervised_dB = 10 * torch.log10(MSE_Unsupervised_Linear)
            print('MSE Original:', MSE_Original_dB.mean().item(), '[dB]')
            print('MSE CPDUnsupervised:', MSE_CPDUnsupervised_dB.mean().item(), '[dB]')
            print('MSE Unsupervised:', MSE_Unsupervised_dB.mean().item(), '[dB]')
            # Create tensorboard writer
            
            
            # # Log MSE values for each time step
            # writer.add_scalars('MSE Comparison', {
            #     'Original MSE': MSE_Original_dB.mean().item(),
            #     'CPDUnsupervised MSE': MSE_CPDUnsupervised_dB.mean().item(),
            #     'Unsupervised MSE': MSE_Unsupervised_dB.mean().item()
            # }, trajectorie)
            
            # writer.close()
            
            
            # Create a figure with 3 subplots
            fig = plt.figure(figsize=(15, 5))
            
            # Online trajectory subplot
            ax1 = fig.add_subplot(131, projection='3d')
            ax1.plot(self.state_predictions[trajectorie, 0, :].cpu().detach().numpy(),
                    self.state_predictions[trajectorie, 1, :].cpu().detach().numpy(),
                    self.state_predictions[trajectorie, 2, :].cpu().detach().numpy(),
                    label='y_out_online')
            ax1.set_title('Online Trajectory')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.legend()
            
            # Original trajectory subplot
            ax2 = fig.add_subplot(132, projection='3d')
            ax2.plot(self.state_original[trajectorie, 0, :].cpu().detach().numpy(),
                    self.state_original[trajectorie, 1, :].cpu().detach().numpy(),
                    self.state_original[trajectorie, 2, :].cpu().detach().numpy(),
                    label='y_out_original')
            ax2.set_title('Original Trajectory')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.legend()
            
            # True trajectory subplot
            ax3 = fig.add_subplot(133, projection='3d')
            ax3.plot(x_true[trajectorie, 0, :].cpu().detach().numpy(),
                    x_true[trajectorie, 1, :].cpu().detach().numpy(),
                    x_true[trajectorie, 2, :].cpu().detach().numpy(),
                    label='x_true')
            ax3.set_title('True Trajectory')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            ax3.legend()
            
            plt.tight_layout()
            plt.show()

            # # X-axis comparison
            # fig_x = plt.figure(figsize=(12, 4))
            # plt.plot(self.state_predictions[trajectorie, 0, :].cpu().detach().numpy(), label='y_out_online')
            # plt.plot(self.state_original[trajectorie, 0, :].cpu().detach().numpy(), label='y_out_original')
            # plt.plot(x_true[trajectorie, 0, :].cpu().detach().numpy(), label='x_true')
            # plt.title('X-axis Comparison')
            # plt.xlabel('Time')
            # plt.ylabel('X')
            # plt.legend()
            # plt.show()

            # # Y-axis comparison
            # fig_y = plt.figure(figsize=(12, 4))
            # plt.plot(self.state_predictions[trajectorie, 1, :].cpu().detach().numpy(), label='y_out_online')
            # plt.plot(self.state_original[trajectorie, 1, :].cpu().detach().numpy(), label='y_out_original')
            # plt.plot(x_true[trajectorie, 1, :].cpu().detach().numpy(), label='x_true')
            # plt.title('Y-axis Comparison')
            # plt.xlabel('Time')
            # plt.ylabel('Y')
            # plt.legend()
            # plt.show()

            # # Z-axis comparison
            # fig_z = plt.figure(figsize=(12, 4))
            # plt.plot(self.state_predictions[trajectorie, 2, :].cpu().detach().numpy(), label='y_out_online')
            # plt.plot(self.state_original[trajectorie, 2, :].cpu().detach().numpy(), label='y_out_original')
            # plt.plot(x_true[trajectorie, 2, :].cpu().detach().numpy(), label='x_true')
            # plt.title('Z-axis Comparison')
            # plt.xlabel('Time')
            # plt.ylabel('Z')
            # plt.legend()
            # plt.show()

