import os
import torch
import copy
import torch.nn as nn
import random
import time
from Plot import Plot_extended
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from Simulations.utils import cpd_dataset_process,cpd_dataset_process_single  

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
        
    def setssModel(self, ssModel):
        self.ssModel = ssModel
        
    def ResetOptimizer(self):
        self.optimizer = torch.optim.Adam(self.KNetmodel.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

    def setKNetLearningRate(self, learningRate):
        self.learningRate = learningRate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learningRate
        
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
        

    def NNTrain(self, SysModel,y_observation,x_true,train_init,cpd_input_for_plot,cpt_target_for_plot):
        
        self.N_E = len(y_observation)  # Number of trajectories
        # self.MSE_cv_linear_epoch = torch.zeros([self.N_steps])
        # self.MSE_cv_dB_epoch = torch.zeros([self.N_steps])
        # self.MSE_train_linear_epoch = torch.zeros([self.N_steps])
        # self.MSE_train_dB_epoch = torch.zeros([self.N_steps])
        
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
        self.observation_predictions = torch.empty((self.N_E, 1, SysModel.T))
        self.state_predictions = torch.empty((self.N_E, 3, SysModel.T))
        self.observation_original = torch.empty((self.N_E, 1, SysModel.T))
        self.state_original = torch.empty((self.N_E, 3, SysModel.T))

        # Copy to restore the NN to its original state for each trajectory
        original_model = copy.deepcopy(self.KNetmodel)
        original_model.eval()
        original_model.batch_size = 1
        original_model.init_hidden_KNet()

        
        
        # Start looping over trajectories
        for trajectorie in range(self.N_E):
            print('Trajectory: ', trajectorie + 1, '/', self.N_E)

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
            # Calculate the number of strides required
            number_of_stride = int(self.ssModel.T / self.stride)
            
            cpd_output = torch.zeros(1, SysModel.n, SysModel.T-self.sample_interval+1).to(self.device)
            # cpd_input = torch.zeros(1, SysModel.n, SysModel.T-self.sample_interval+1).to(self.device)
            cpd_input = torch.zeros(1, SysModel.n, SysModel.T).to(self.device)
            counter = 0
            

            # Go through the whole trajectory step by step
            for stride in range(number_of_stride):
                observation = y_training[:, :, (stride * self.stride):(stride * self.stride + self.stride)].to(self.device)
                x_out_online = torch.empty(1, self.ssModel.m, self.stride).to(self.device)
                y_out_online = torch.zeros(1, self.ssModel.n, self.stride).to(self.device)
                x_out_original = torch.empty(1, self.ssModel.m, self.stride).to(self.device)
                y_out_original = torch.zeros(1, self.ssModel.n, self.stride).to(self.device)
                # Initialize training mode
                self.KNetmodel.train()
                original_model.eval()
                # Initialize the informations
                self.KNetmodel.InitSequence(self.KNetmodel.m1x_posterior.clone().detach(),SysModel.T)
                original_model.InitSequence(original_model.m1x_posterior.clone().detach(),SysModel.T)
                self.KNetmodel.init_hidden_KNet()
                original_model.init_hidden_KNet()

                # Start training
                for i in range(self.stride):
                    x_out_online[:, :, i] = torch.squeeze(self.KNetmodel(torch.unsqueeze(observation[:, :, i],dim=2)))
                    y_out_online[:, :, i] = torch.squeeze(self.KNetmodel.m1y)
                    x_out_original[:, :, i] = torch.squeeze(original_model(torch.unsqueeze(observation[:, :, i],dim=2)))
                    y_out_original[:, :, i] = torch.squeeze(original_model.m1y)

                self.observation_predictions[trajectorie, :,
                (stride * self.stride):(stride * self.stride + self.stride)] = y_out_online.detach()
                self.state_predictions[trajectorie, :,
                (stride * self.stride):(stride * self.stride + self.stride)] = x_out_online.detach()
                self.observation_original[trajectorie, :,
                (stride * self.stride):(stride * self.stride + self.stride)] = y_out_original.detach()
                self.state_original[trajectorie, :,
                (stride * self.stride):(stride * self.stride + self.stride)] = x_out_original.detach()
                if stride >0:
                    for t in range(self.stride):
                        cpd_input[:, :, (stride-1)*self.stride+t] = cpd_dataset_process(torch.unsqueeze(self.observation_predictions[trajectorie, :,(stride * self.stride-self.stride+t):(stride * self.stride+t)],dim=0).to(self.device),
                                                                    y_training[:, :, (stride * self.stride-self.stride+t):(stride * self.stride+t)])
                    cpd_out_tmp = self.CPDmodel(cpd_input[:, :, (stride-1)*self.stride:(stride-1)*self.stride+self.stride])
                    new_lr = (20*cpd_out_tmp*self.learningRate).item()
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr


                LOSS = self.loss_fn(y_out_online,observation)
                self.optimizer.zero_grad()
                # optimize
                LOSS.backward()
                
                self.optimizer.step()
                
                #  Print statistics every 10% of a trajectory
                counter += 1
                if counter % max(int(number_of_stride/10),1) == 0:
                    print('Training itt:', stride + 1, '/', number_of_stride, ',OBS MSE:',
                          10 * torch.log10(LOSS).item(), '[dB]')
                    
                del LOSS,y_out_online,observation
            # Reset the optimizer for the next trajectory
            self.ResetOptimizer()
            
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.observation_predictions[trajectorie, 0, :].cpu().detach().numpy(), label='y_out_online')
            plt.plot(self.observation_original[trajectorie, 0, :].cpu().detach().numpy(), label='y_out_observation')
            plt.plot(x_true[trajectorie, 0, :].cpu().detach().numpy(), label='x_true')
            plt.title('y_out_online, y_training, x_out_online, x_true')
            plt.legend()
            plt.show()
            
            
            # plt.figure(figsize=(10, 6))
            # plt.subplot(1, 2, 1)
            # plt.plot(self.observation_predictions[trajectorie, 0, :].cpu().detach().numpy(), label='y_out_online_total')
            # plt.plot(y_training[0, 0, :].cpu().detach().numpy(), label='y_training')
            # plt.plot(x_true[trajectorie, 0, :].cpu().detach().numpy(), label='x_true')
            # plt.title('y_out_online_total, y_training, x_out_online_total, x_true')
            # plt.legend()
            
            # plt.subplot(1, 2, 2)
            # plt.plot(cpt_target_for_plot[trajectorie, 0, :].cpu().detach().numpy(), label='cpt_target_for_plot')
            # plt.plot(cpd_output[0, 0, :].cpu().detach().numpy(), label='cpt_output')
            # plt.title('cpd_output_total')
            # plt.legend()
            # plt.tight_layout()
            # plt.show()

        loss_fn = torch.nn.MSELoss(reduction='none')

        # self.MSE_state_arr = loss_fn(training_dataset.target,self.state_predictions)
        # self.MSE_observation_arr = loss_fn(training_dataset.input,self.output_predictions)

        self.MSE_states_over_time = 10 * torch.log10(torch.mean(self.MSE_state_arr,axis = (0,1)))
        self.MSE_observation_over_time = 10 * torch.log10(torch.mean(self.MSE_observation_arr,axis = (0,1)))

        self.MSE_states_over_trajectories = 10 * torch.log10(torch.mean(self.MSE_state_arr,axis = (1,2)))
        self.MSE_observation_over_trajectories = 10 * torch.log10(torch.mean(self.MSE_observation_arr,axis = (1,2)))


        self.MSE_states_before_training = 10 * torch.log10(torch.mean(self.MSE_state_arr[:,:,:self.training_start])).item()
        self.MSE_states_after_training = 10 * torch.log10(torch.mean(self.MSE_state_arr[:,:,self.training_start:])).item()

        if not self.training_start==0:
            print('MSE before training start:',self.MSE_states_before_training,'[dB]')
        print('MSE after training start:', self.MSE_states_after_training,'[dB]')

