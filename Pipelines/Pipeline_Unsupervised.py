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
        
    def NNTrain(self, SysModel,y_observation,x_true,\
        randomInit=False,train_init=None,train_lengthMask=None):
        
        self.N_E = len(y_observation)
        self.MSE_cv_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_cv_dB_epoch = torch.zeros([self.N_steps])

        self.MSE_train_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_train_dB_epoch = torch.zeros([self.N_steps])


        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for ti in range(0, self.N_steps):

            ###############################
            ### Training Sequence Batch ###
            ###############################
            self.optimizer.zero_grad()
            # KNet Training Mode
            self.KNetmodel.train()
            self.KNetmodel.batch_size = self.N_B
            # CPDNet Test Mode
            self.CPDmodel.eval()
            self.CPDmodel.batch_size = self.N_B
            # Init Hidden State
            self.KNetmodel.init_hidden_KNet()

            # Init Training Batch tensors
            y_training_knet = torch.zeros([self.N_B, SysModel.n, SysModel.T]).to(self.device)
            x_training_knet = torch.zeros([self.N_B, SysModel.m, SysModel.T]).to(self.device)
            x_out_training_knet = torch.zeros([self.N_B, 1, SysModel.T]).to(self.device)
            y_out_training_knet = torch.zeros([self.N_B, 1, SysModel.T]).to(self.device)
            probability_cpd = torch.zeros([self.N_B, 1, SysModel.T-self.sample_interval+1]).to(self.device)
            cpd_train_input_batch = torch.zeros([self.N_B,1, SysModel.T-self.sample_interval+1]).to(self.device)

            # Randomly select N_B training sequences
            assert self.N_B <= self.N_E # N_B must be smaller than N_E
            n_e = random.sample(range(self.N_E), k=self.N_B)
            ii = 0
            for index in n_e:
                if self.args.randomLength:
                    y_training_knet[ii,:,train_lengthMask[index,:]] = y_observation[index,:,train_lengthMask[index,:]]
                    x_true[ii,:,train_lengthMask[index,:]] = x_true[index,:,train_lengthMask[index,:]]
                else:
                    y_training_knet[ii,:,:] = y_observation[index]
                    x_training_knet[ii,:,:] = x_true[index]
                ii += 1
            
            # Init Sequence
            if(randomInit):
                train_init_batch = torch.empty([self.N_B, SysModel.m,1]).to(self.device)
                ii = 0
                for index in n_e:
                    train_init_batch[ii,:,0] = torch.squeeze(train_init[index])
                    ii += 1
                self.KNetmodel.InitSequence(train_init_batch, SysModel.T)
            else:
                self.KNetmodel.InitSequence(\
                SysModel.m1x_0.reshape(1,SysModel.m,1).repeat(self.N_B,1,1), SysModel.T)
            
            # Forward Computation
            window_size = 20
            detect_trigger = False
            trigger_iteration = 0
            cpd_probability_store = torch.zeros([self.N_B, window_size]).to(self.device)
            for t in range(0, SysModel.T):
                # Detecting stage:
                x_out_training_knet[:, :, t] = torch.squeeze(self.KNetmodel(torch.unsqueeze(y_training_knet[:, :, t],2)))[:,0:1]
                y_out_training_knet[:, :, t] = torch.unsqueeze(self.KNetmodel.m1y.squeeze(),dim=1)
                cpd_train_input_batch = cpd_dataset_process_single(y_out_training_knet[:, :, t],
                                                            y_training_knet[:, :, t]
                                                            ).to(self.device)
                probability_cpd[:,:,t] = self.CPDmodel(torch.unsqueeze(cpd_train_input_batch,2))
                
                # If probability larger than threshold, store following values for detection and retraining
                if (torch.mean(probability_cpd[:,:,t],dim=0) > self.args.threshold) or detect_trigger:
                    if detect_trigger == False:
                        detect_trigger = True
                        trigger_iteration = t
                    cpd_probability_store[:,t%window_size] = probability_cpd[:,:,t]
                    if t >= window_size:
                        # Calculate the mean of the last window_size probabilities
                        mean_prob = torch.mean(cpd_probability_store, dim=1)
                        # Check if the mean probability is above the threshold
                        if mean_prob > self.args.threshold:
                            # Perform retraining or any other action here
                            print(f"Retraining triggered at time step {t}")
                
                # # Plot x_out_training_knet and y_out_training_knet in one figure,
                # # plot probability_cpd in another figure
                # _, ax = plt.subplots(1, 2, figsize=(12, 6))
                # ax[0].plot(x_out_training_knet[0, 0, :].cpu().detach().numpy(), label='x_out')
                # ax[0].plot(y_out_training_knet[0, 0, :].cpu().detach().numpy(), label='y_estimation')
                # ax[0].plot(x_training_knet[0, 0, :].cpu().detach().numpy(), label='x_true')
                # ax[0].plot(y_training_knet[0, 0, :].cpu().detach().numpy(), label='y_true')
                # ax[0].set_title('x_out and y_out')
                # ax[0].legend()
                # ax[1].plot(probability_cpd[0, 0, :].cpu().detach().numpy(), label='probability_cpd')
                # ax[1].set_title('probability_cpd')
                # ax[1].legend()
                # plt.show()

                # dB Loss
                MSE_trainbatch_linear_LOSS = self.loss_fn(y_out_training_knet, y_training_knet)
                self.MSE_train_linear_epoch[ti] = MSE_trainbatch_linear_LOSS.item()
                self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

                MSE_trainbatch_linear_LOSS.backward(retain_graph=True)
                self.optimizer.step()
                
                # Set the learning rate to a specific value
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = torch.mean(probability_cpd)  # Example: set learning rate to 0.001

            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti],
                  "[dB]")
                      
            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

        # return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]


