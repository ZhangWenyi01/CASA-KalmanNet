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
        self.stride = 1
        self.MSE_cv_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_cv_dB_epoch = torch.zeros([self.N_steps])
        self.MSE_train_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_train_dB_epoch = torch.zeros([self.N_steps])
        

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
        self.output_predictions = torch.empty((self.N_E, 1, SysModel.T), requires_grad=False)
        self.state_predictions = torch.empty((self.N_E, 3, SysModel.T), requires_grad=False)

        # Copy to restore the NN to its original state for each trajectory
        original_model = copy.deepcopy(self.KNetmodel)

        # For printing out useful information
        counter = 0

        # Start looping over trajectories
        for trajectorie in range(self.N_E):
            print('Trajectory: ', trajectorie + 1, '/', self.N_E)

            ###############################
            ### Training Sequence Batch ###
            ###############################

            # Reset the model
            self.KNetmodel = copy.deepcopy(original_model)
            self.reTraining = False

            # Reset optimizer
            self.ResetOptimizer()

            # Training Mode
            self.KNetmodel.train()

            # Load the next trajectory
            y_training = torch.unsqueeze(y_observation[trajectorie, :, :],0).to(self.device)

            # Initialize state
            self.KNetmodel.InitSequence(torch.unsqueeze(train_init[trajectorie,:,:],0).to(self.device),self.args.T)

            # Calculate the number of strides required
            number_of_stride = int(SysModel.T / self.stride)
            # Calculate the remainder
            remainder = int(SysModel.T % self.stride)
            
            x_out_online_total = torch.empty(1, SysModel.m, SysModel.T).to(self.device)
            y_out_online_total = torch.zeros(1, SysModel.n, SysModel.T).to(self.device)
            cpd_output_total = torch.zeros(1, SysModel.n, SysModel.T).to(self.device)
            cpd_input_total = torch.zeros(1, SysModel.n, SysModel.T).to(self.device)

            # Go through the whole trajectory stride by stride, updating the NN parameters after every stride-
            # time steps
            for stride in range(number_of_stride):

                # Initialize training mode
                self.KNetmodel.train()

                # Set the initial posterior to the previous posterior and detaching it from the gradient calculation
                self.KNetmodel.InitSequence(self.KNetmodel.m1x_posterior.detach(),SysModel.T)

                # Get next observations
                observations = y_training[0,:, (stride * self.stride):(stride * self.stride + self.stride)]
                observations = observations.reshape(1,SysModel.n,self.stride)

                # Initialize hidden state of GRU
                # self.KNetmodel.init_hidden()

                # Allocate estimate arrays
                x_KNet_posterior = torch.empty(1, SysModel.m, self.stride, requires_grad=True).to(self.device)
                y_KNet_posterior = torch.zeros(1, SysModel.n, self.stride, requires_grad=True).to(self.device)
                m1y_KNet_posterior = torch.zeros(1, SysModel.n, self.stride).to(self.device)
                cpd_input = torch.zeros(1, SysModel.n, self.stride).to(self.device)
                cpd_output = torch.zeros(1, SysModel.n, self.stride).to(self.device)

                # Loop trough a single stride
                for t in range(self.stride):
                    # Take time step in NN
                    x_KNet_posterior[:, :, t] = torch.squeeze(self.KNetmodel(torch.unsqueeze(observations[:, :, t], 0)).T,2)
                    # Get the output estimate from the NN
                    y_KNet_posterior[:, :, t] = self.KNetmodel.m1y.squeeze().T.clone().detach()
                    m1y_KNet_posterior[:, :, t] = self.KNetmodel.m1y.squeeze().T.clone().detach()
                # Plot x_KNet_posterior, y_KNet_posterior, and cpd_output
                cpd_input[:, :, :] = cpd_dataset_process(observations,m1y_KNet_posterior)
                
                x_out_online_total[:, :, (stride * self.stride):(stride * self.stride + self.stride)] = x_KNet_posterior
                y_out_online_total[:, :, (stride * self.stride):(stride * self.stride + self.stride)] = y_KNet_posterior
                cpd_output_total[:, :, (stride * self.stride):(stride * self.stride + self.stride)] = cpd_output
                cpd_input_total[:, :, (stride * self.stride):(stride * self.stride + self.stride)] = cpd_input
                
                # # Plot cpd_input_total and cpd_input_for_plot
                # plt.figure(figsize=(10, 6))
                # plt.plot(cpd_input_total[0, 0, :].cpu().detach().numpy(), label='cpd_input_total')
                # plt.plot(cpd_input_for_plot[0, 0, :].cpu().detach().numpy(), label='cpd_input_for_plot')
                # plt.title('cpd_input_total vs cpd_input_for_plot')
                # plt.legend()
                # plt.show()
                    

                # Count the number of exceeding threshold values in the CPD output
                num_exceeding_threshold = torch.sum(cpd_output > self.args.threshold).item()
                if num_exceeding_threshold > self.stride/2:
                    self.reTraining = True
                
                # Plug obtained values into the allocated arrays
                self.output_predictions[trajectorie, :,
                (stride * self.stride):(stride * self.stride + self.stride)] = y_KNet_posterior.detach()
                self.state_predictions[trajectorie, :,
                (stride * self.stride):(stride * self.stride + self.stride)] = x_KNet_posterior.detach()

                # Calculate Loss
                LOSS = self.loss_fn(y_KNet_posterior, observations)

                # Print statistics every 10% of a trajectory
                counter += 1
                if counter % max(int(number_of_stride/10),1) == 0:
                    print('Training itt:', stride + 1, '/', number_of_stride, ',OBS MSE:',
                          10 * torch.log10(LOSS).item(), '[dB]')

                # optimize if reTraining is True
                if self.reTraining is True:
                    # Zero Gradient
                    self.optimizer.zero_grad()
                    # optimize
                    LOSS.backward()
                    self.optimizer.step()

                # Clear variables to save memory
                del observations, y_KNet_posterior, LOSS, x_KNet_posterior
                
            # Calculate the final time steps
            if not remainder == 0:

                # Initialize the posterior
                self.KNetmodel.InitSequence(self.KNetmodel.m1x_posterior.detach())

                # Get Observations
                observations = y_training[0, :, -remainder:].reshape(1, SysModel.n, remainder).detach()

                # Initialize hidden state of GRU
                self.KNetmodel.init_hidden()

                # Allocate estimates
                x_KNet_posterior = torch.empty(1, SysModel.m, remainder)
                y_KNet_posterior = torch.empty(1, SysModel.n, remainder)

                # Loop through the remaining time steps
                for t in range(remainder):
                    # Take time step in NN
                    x_KNet_posterior[:, :, t] = self.KNetmodel(observations[:, :, t]).T
                    # Get the output of the NN
                    y_KNet_posterior[:, :, t] = self.KNetmodel.m1y.squeeze().T

                # Plug obtained values into the allocated arrays
                self.output_predictions[trajectorie, :, -remainder:] = y_KNet_posterior
                self.state_predictions[trajectorie, :, -remainder:] = x_KNet_posterior

            # Reset the optimizer for the next trajectory
            self.ResetOptimizer()
            
            plt.figure(figsize=(10, 6))
            plt.subplot(2, 1, 1)
            plt.plot(y_out_online_total[0, 0, :].cpu().detach().numpy(), label='y_out_online_total')
            plt.plot(y_observation[0, 0, :].cpu().detach().numpy(), label='y_observation')
            plt.plot(x_out_online_total[0, 0, :].cpu().detach().numpy(), label='x_out_online_total')
            plt.plot(x_true[trajectorie, 0, :].cpu().detach().numpy(), label='x_true')
            plt.title('y_out_online_total, y_observation, x_out_online_total, x_true')
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(cpd_output_total[0, 0, :].cpu().detach().numpy(), label='cpd_output_total')
            plt.plot(cpt_target_for_plot[0, 0, :].cpu().detach().numpy(), label='cpt_target_for_plot')
            plt.title('cpd_output_total, cpt_target_for_plot')
            plt.legend()
            plt.tight_layout()
            plt.show()

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

