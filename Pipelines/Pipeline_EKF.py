"""
This file contains the class Pipeline_EKF, 
which is used to train and test KalmanNet.
"""

import torch
import torch.nn as nn
import random
import time
from Plot import Plot_extended
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

class Pipeline_EKF:

    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"

    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, model):
        self.model = model

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

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

    def NNTrain(self, SysModel, cv_input, cv_target, train_input, train_target, path_results, \
        MaskOnState=False, randomInit=False,cv_init=None,train_init=None,\
        train_lengthMask=None,cv_lengthMask=None):

        self.writer = SummaryWriter(os.path.join(self.folderName, 'runs'))
        
        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        self.MSE_cv_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_cv_dB_epoch = torch.zeros([self.N_steps])

        self.MSE_train_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_train_dB_epoch = torch.zeros([self.N_steps])
        
        if MaskOnState:
            mask = torch.tensor([True,False,False])
            if SysModel.m == 2: 
                mask = torch.tensor([True,False])

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
            # Training Mode
            self.model.train()
            self.model.batch_size = self.N_B
            # Init Hidden State
            self.model.init_hidden_KNet()

            # Init Training Batch tensors
            y_training_batch = torch.zeros([self.N_B, SysModel.n, SysModel.T]).to(self.device)
            train_target_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T]).to(self.device)
            x_out_training_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T]).to(self.device)
            if self.args.randomLength:
                MSE_train_linear_LOSS = torch.zeros([self.N_B])
                MSE_cv_linear_LOSS = torch.zeros([self.N_CV])

            # Randomly select N_B training sequences
            assert self.N_B <= self.N_E # N_B must be smaller than N_E
            n_e = random.sample(range(self.N_E), k=self.N_B)
            ii = 0
            for index in n_e:
                if self.args.randomLength:
                    y_training_batch[ii,:,train_lengthMask[index,:]] = train_input[index,:,train_lengthMask[index,:]]
                    train_target_batch[ii,:,train_lengthMask[index,:]] = train_target[index,:,train_lengthMask[index,:]]
                else:
                    y_training_batch[ii,:,:] = train_input[index]
                    train_target_batch[ii,:,:] = train_target[index]
                ii += 1
            
            # Init Sequence
            if(randomInit):
                train_init_batch = torch.empty([self.N_B, SysModel.m,1]).to(self.device)
                ii = 0
                for index in n_e:
                    train_init_batch[ii,:,0] = torch.squeeze(train_init[index])
                    ii += 1
                self.model.InitSequence(train_init_batch, SysModel.T)
            else:
                self.model.InitSequence(\
                SysModel.m1x_0.reshape(1,SysModel.m,1).repeat(self.N_B,1,1), SysModel.T)
            
            # Forward Computation
            for t in range(0, SysModel.T):
                x_out_training_batch[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(y_training_batch[:, :, t],2)))
            
            # Compute Training Loss
            MSE_trainbatch_linear_LOSS = 0
            if (self.args.CompositionLoss):
                y_hat = torch.zeros([self.N_B, SysModel.n, SysModel.T])
                for t in range(SysModel.T):
                    y_hat[:,:,t] = torch.squeeze(SysModel.h(torch.unsqueeze(x_out_training_batch[:,:,t])))

                if(MaskOnState):### FIXME: composition loss, y_hat may have different mask with x
                    if self.args.randomLength:
                        jj = 0
                        for index in n_e:# mask out the padded part when computing loss
                            MSE_train_linear_LOSS[jj] = self.alpha * self.loss_fn(x_out_training_batch[jj,mask,train_lengthMask[index]], train_target_batch[jj,mask,train_lengthMask[index]])+(1-self.alpha)*self.loss_fn(y_hat[jj,mask,train_lengthMask[index]], y_training_batch[jj,mask,train_lengthMask[index]])
                            jj += 1
                        MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
                    else:                     
                        MSE_trainbatch_linear_LOSS = self.alpha * self.loss_fn(x_out_training_batch[:,mask,:], train_target_batch[:,mask,:])+(1-self.alpha)*self.loss_fn(y_hat[:,mask,:], y_training_batch[:,mask,:])
                else:# no mask on state
                    if self.args.randomLength:
                        jj = 0
                        for index in n_e:# mask out the padded part when computing loss
                            MSE_train_linear_LOSS[jj] = self.alpha * self.loss_fn(x_out_training_batch[jj,:,train_lengthMask[index]], train_target_batch[jj,:,train_lengthMask[index]])+(1-self.alpha)*self.loss_fn(y_hat[jj,:,train_lengthMask[index]], y_training_batch[jj,:,train_lengthMask[index]])
                            jj += 1
                        MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
                    else:                
                        MSE_trainbatch_linear_LOSS = self.alpha * self.loss_fn(x_out_training_batch, train_target_batch)+(1-self.alpha)*self.loss_fn(y_hat, y_training_batch)
            
            else:# no composition loss
                if(MaskOnState):
                    if self.args.randomLength:
                        jj = 0
                        for index in n_e:# mask out the padded part when computing loss
                            MSE_train_linear_LOSS[jj] = self.loss_fn(x_out_training_batch[jj,mask,train_lengthMask[index]], train_target_batch[jj,mask,train_lengthMask[index]])
                            jj += 1
                        MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
                    else:
                        MSE_trainbatch_linear_LOSS = self.loss_fn(x_out_training_batch[:,mask,:], train_target_batch[:,mask,:])
                else: # no mask on state
                    if self.args.randomLength:
                        jj = 0
                        for index in n_e:# mask out the padded part when computing loss
                            MSE_train_linear_LOSS[jj] = self.loss_fn(x_out_training_batch[jj,:,train_lengthMask[index]], train_target_batch[jj,:,train_lengthMask[index]])
                            jj += 1
                        MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
                    else: 
                        MSE_trainbatch_linear_LOSS = self.loss_fn(x_out_training_batch, train_target_batch)

            # dB Loss
            self.MSE_train_linear_epoch[ti] = MSE_trainbatch_linear_LOSS.item()
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])


            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            MSE_trainbatch_linear_LOSS.backward(retain_graph=True)

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()
            # self.scheduler.step(self.MSE_cv_dB_epoch[ti])

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()
            self.model.batch_size = self.N_CV
            # Init Hidden State
            self.model.init_hidden_KNet()
            with torch.no_grad():

                SysModel.T_test = cv_input.size()[-1] # T_test is the maximum length of the CV sequences

                x_out_cv_batch = torch.empty([self.N_CV, SysModel.m, SysModel.T_test]).to(self.device)
                
                # Init Sequence
                if(randomInit):
                    if(cv_init==None):
                        self.model.InitSequence(\
                        SysModel.m1x_0.reshape(1,SysModel.m,1).repeat(self.N_CV,1,1), SysModel.T_test)
                    else:
                        self.model.InitSequence(cv_init, SysModel.T_test)                       
                else:
                    self.model.InitSequence(\
                        SysModel.m1x_0.reshape(1,SysModel.m,1).repeat(self.N_CV,1,1), SysModel.T_test)

                for t in range(0, SysModel.T_test):
                    x_out_cv_batch[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(cv_input[:, :, t],2)))
                
                # Compute CV Loss
                MSE_cvbatch_linear_LOSS = 0
                if(MaskOnState):
                    if self.args.randomLength:
                        for index in range(self.N_CV):
                            MSE_cv_linear_LOSS[index] = self.loss_fn(x_out_cv_batch[index,mask,cv_lengthMask[index]], cv_target[index,mask,cv_lengthMask[index]])
                        MSE_cvbatch_linear_LOSS = torch.mean(MSE_cv_linear_LOSS)
                    else:          
                        MSE_cvbatch_linear_LOSS = self.loss_fn(x_out_cv_batch[:,mask,:], cv_target[:,mask,:])
                else:
                    if self.args.randomLength:
                        for index in range(self.N_CV):
                            MSE_cv_linear_LOSS[index] = self.loss_fn(x_out_cv_batch[index,:,cv_lengthMask[index]], cv_target[index,:,cv_lengthMask[index]])
                        MSE_cvbatch_linear_LOSS = torch.mean(MSE_cv_linear_LOSS)
                    else:
                        MSE_cvbatch_linear_LOSS = self.loss_fn(x_out_cv_batch, cv_target)

                # dB Loss
                self.MSE_cv_linear_epoch[ti] = MSE_cvbatch_linear_LOSS.item()
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])
                
                # Add validation loss to TensorBoard
                self.writer.add_scalar('Loss/validation_dB', self.MSE_cv_dB_epoch[ti], ti)
                self.writer.add_scalar('Loss/train_dB', self.MSE_train_dB_epoch[ti], ti)    
                
                if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti
                    
                    torch.save(self.model, path_results + 'best-model.pt')

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

        # Close TensorBoard writer
        self.writer.close()

        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]

    def NNTest(self, SysModel, test_input, test_target, path_results, MaskOnState=False,\
     randomInit=False,test_init=None,load_model=False,load_model_path=None,\
        test_lengthMask=None):
        # Load model
        if load_model:
            self.model = torch.load(load_model_path, map_location=self.device,weights_only=False) 
        else:
            self.model = torch.load(path_results+'best-model.pt', map_location=self.device,weights_only=False) 

        self.N_T = test_input.shape[0]
        SysModel.T_test = test_input.size()[-1]
        self.MSE_test_linear_arr = torch.zeros([self.N_T])
        x_out_test = torch.zeros([self.N_T, SysModel.m,SysModel.T_test]).to(self.device)

        if MaskOnState:
            mask = torch.tensor([True,False,False])
            if SysModel.m == 2: 
                mask = torch.tensor([True,False])

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        # Test mode
        self.model.eval()
        self.model.batch_size = self.N_T
        # Init Hidden State
        self.model.init_hidden_KNet()
        torch.no_grad()

        start = time.time()

        if (randomInit):
            self.model.InitSequence(test_init, SysModel.T_test)               
        else:
            self.model.InitSequence(SysModel.m1x_0.reshape(1,SysModel.m,1).repeat(self.N_T,1,1), SysModel.T_test)         
        
        for t in range(0, SysModel.T_test):
            x_out_test[:,:, t] = torch.squeeze(self.model(torch.unsqueeze(test_input[:,:, t],2)))
        
        end = time.time()
        t = end - start

        # MSE loss
        for j in range(self.N_T):# cannot use batch due to different length and std computation  
            if(MaskOnState):
                if self.args.randomLength:
                    self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j,mask,test_lengthMask[j]], test_target[j,mask,test_lengthMask[j]]).item()
                else:
                    self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j,mask,:], test_target[j,mask,:]).item()
            else:
                if self.args.randomLength:
                    self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j,:,test_lengthMask[j]], test_target[j,:,test_lengthMask[j]]).item()
                else:
                    self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j,:,:], test_target[j,:,:]).item()
        
        
        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # Standard deviation
        self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)

        # Confidence interval
        self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg

        # Print MSE and std
        str = self.modelName + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        str = self.modelName + "-" + "STD Test:"
        print(str, self.test_std_dB, "[dB]")
        # Print Run Time
        print("Inference Time:", t)

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_test, t]

    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot_extended(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.N_steps, MSE_KF_dB_avg,
                                self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)
    
    def CPD_Dataset(self, SysModel,train_input, train_target, cv_input, cv_target,test_input,test_target, path_results, path_dataset, MaskOnState=False,\
        randomInit=False, test_init=None, load_model=False, load_model_path=None,\
            test_lengthMask=None,cv_init=None):
        # Load model
        if load_model:
            self.model = torch.load(load_model_path, map_location=self.device,weights_only=False) 
        else:
            self.model = torch.load(path_results+'best-model.pt', map_location=self.device,weights_only=False) 

        self.N_T = train_input.shape[0]
        self.N_CV = cv_input.shape[0]
        SysModel.T_test = train_input.size()[-1]
        SysModel.T_cv = cv_input.size()[-1]
        
        x_out_train = torch.zeros([self.N_T, 1, SysModel.T_test]).to(self.device)
        x_out_train_prior = torch.zeros([self.N_T, 1, SysModel.T_test]).to(self.device)
        
        x_out_cv = torch.zeros([self.N_CV, 1, SysModel.T_cv]).to(self.device)
        x_out_cv_prior = torch.zeros([self.N_CV, 1, SysModel.T_cv]).to(self.device)
        
        x_out_test = torch.zeros([self.N_T, 1, SysModel.T_test]).to(self.device)
        x_out_test_prior = torch.zeros([self.N_T, 1, SysModel.T_test]).to(self.device)
        
        y_train_estimation = torch.zeros([self.N_T, SysModel.n, SysModel.T_test]).to(self.device)
        
        y_cv_estimation = torch.zeros([self.N_CV, SysModel.n, SysModel.T_cv]).to(self.device)
        
        y_test_estimation = torch.zeros([self.N_T, SysModel.n, SysModel.T_test]).to(self.device)

        if MaskOnState:
            mask = torch.tensor([True,False,False])
            if SysModel.m == 2: 
                mask = torch.tensor([True,False])

        # Test mode
        self.model.eval()
        torch.no_grad()

        # Process test data
        self.model.batch_size = self.N_T
        self.model.init_hidden_KNet()  # Reset hidden state
        if (randomInit):
            self.model.InitSequence(test_init, SysModel.T_test)               
        else:
            self.model.InitSequence(SysModel.m1x_0.reshape(1,SysModel.m,1).repeat(self.N_T,1,1), SysModel.T_test)
        
        for t in range(0, SysModel.T_test):
            output, prior, y = self.model(torch.unsqueeze(train_input[:,:, t],2))
            x_out_train[:,:, t] = output[:,0,:]
            x_out_train_prior[:,:, t] = prior[:,0,:]
            y_train_estimation[:,:, t] = torch.squeeze(y, dim=2)
            
        # # Plot 2D curves for x_out_train_prior, x_out_train, train_input, and train_target
        # time_steps = torch.arange(SysModel.T_test).cpu().detach().numpy()
        # i=20
        # plt.figure(figsize=(10, 6))
        # plt.plot(time_steps, x_out_train_prior[i, 0, :].cpu().detach().numpy(),
        #         label='x_out_train_prior', linestyle='--', color='orange')
        # plt.plot(time_steps, x_out_train[i, 0, :].cpu().detach().numpy(),
        #         label='x_out_train', color='red')
        # plt.plot(time_steps, train_target[i, 0, :].cpu().detach().numpy(),
        #         label='train_target', color='blue')
        # plt.plot(time_steps, train_input[i, 0, :].cpu().detach().numpy(),
        #         label='train_input', color='green')
        # plt.title(f'2D Curve for Batch {i}')
        # plt.xlabel('Time Step')
        # plt.ylabel('Value (Dimension 1)')
        # plt.legend()
        # plt.grid()
        # plt.show()
        train_input_plt = train_input
        train_target_plt = train_target
        cv_input_plt = cv_input
        cv_target_plt = cv_target
        
        # Process cv data
        self.model.batch_size = self.N_CV
        self.model.init_hidden_KNet()  # Reset hidden state
        if (randomInit):
            self.model.InitSequence(cv_init, SysModel.T_cv)               
        else:
            self.model.InitSequence(SysModel.m1x_0.reshape(1,SysModel.m,1).repeat(self.N_CV,1,1), SysModel.T_cv)
        for t in range(0, SysModel.T_cv):
            output, prior,y= self.model(torch.unsqueeze(cv_input[:,:, t],2))
            x_out_cv[:,:, t] = output[:,0,:]
            x_out_cv_prior[:,:, t] = prior[:,0,:]
            y_cv_estimation[:,:, t] = torch.squeeze(y,dim=2)
            
        # Process test data
        self.model.batch_size = self.N_T
        self.model.init_hidden_KNet()  # Reset hidden state
        if (randomInit):
            self.model.InitSequence(test_init, SysModel.T_test)
        else:
            self.model.InitSequence(SysModel.m1x_0.reshape(1,SysModel.m,1).repeat(self.N_T,1,1), SysModel.T_test)
        for t in range(0, SysModel.T_test):
            output, prior,y= self.model(torch.unsqueeze(test_input[:,:, t],2))
            x_out_test[:,:, t] = output[:,0,:]
            x_out_test_prior[:,:, t] = prior[:,0,:]
            y_test_estimation[:,:, t] = torch.squeeze(y,dim=2)


        # Calculate absolute errors
        train_target = torch.abs(x_out_train[:,0:1,:] - train_target[:,0:1,:])
        train_input = torch.abs(y_train_estimation - train_input)
        cv_target = torch.abs(x_out_cv[:,0:1,:] - cv_target[:,0:1,:])
        cv_input = torch.abs(y_cv_estimation - cv_input)
        test_target = torch.abs(x_out_test[:,0:1,:] - test_target[:,0:1,:])
        test_input = torch.abs(y_test_estimation - test_input)

        # # Normalize errors to range [0, 4]
        # def normalize_error(error):
        #     min_val = torch.min(error, dim=2, keepdim=True)[0]
        #     max_val = torch.max(error, dim=2, keepdim=True)[0]
        #     return (error - min_val) / (max_val - min_val) * 2

        train_target = torch.sigmoid(train_target)-0.5
        train_input = torch.sigmoid(train_input)-0.5
        cv_target = torch.sigmoid(cv_target)-0.5
        cv_input = torch.sigmoid(cv_input)-0.5
        test_target = torch.sigmoid(test_target)-0.5
        test_input = torch.sigmoid(test_input)-0.5
        

        # Apply tanh transformation
        train_target = torch.tanh(15*train_target)
        train_input = torch.tanh(15*train_input)
        cv_target = torch.tanh(15*cv_target)
        cv_input = torch.tanh(15*cv_input)   
        test_target = torch.tanh(15*test_target)
        test_input = torch.tanh(15*test_input)
        
        # Calculate MSE over sample intervals
        def calculate_mse_over_intervals(tanh_index, sample_interval):
            num_intervals = tanh_index.size(2) - sample_interval + 1
            mse_result = torch.zeros((tanh_index.size(0), tanh_index.size(1), num_intervals))
            for i in range(num_intervals):
                mse_result[:, :, i] = torch.mean((tanh_index[:, :, i:i + sample_interval]) ** 2, dim=2)
            return mse_result

        train_target = calculate_mse_over_intervals(train_target, self.sample_interval)
        cv_target = calculate_mse_over_intervals(cv_target, self.sample_interval)
        train_input = calculate_mse_over_intervals(train_input, self.sample_interval)
        cv_input = calculate_mse_over_intervals(cv_input, self.sample_interval)
        test_target = calculate_mse_over_intervals(test_target, self.sample_interval)
        test_input = calculate_mse_over_intervals(test_input, self.sample_interval)
        
        # time_steps = torch.arange(SysModel.T_test).cpu().detach().numpy()
        # for i in range(train_input.size(0)):
        #     plt.figure(figsize=(10, 4))
        #     plt.subplot(1, 2, 1)
        #     plt.plot(train_input[i, 0, :].cpu().detach().numpy(), label="Train Input")
        #     plt.plot(train_target[i, 0, :].cpu().detach().numpy(), label="Train Target")
        #     plt.title(f"Batch {i}")
        #     plt.xlabel("Length")
        #     plt.ylabel("Value")
        #     plt.legend()
            
        #     plt.subplot(1, 2, 2)
        #     plt.plot(time_steps, x_out_train[i, 0, :].cpu().detach().numpy(),
        #         label='x_out_train', color='red')
        #     plt.plot(time_steps, train_target_plt[i, 0, :].cpu().detach().numpy(),
        #         label='train_target', color='blue')
        #     plt.plot(time_steps, y_train_estimation[i, 0, :].cpu().detach().numpy(),
        #         label='y_train_estimation', color='black')
        #     plt.plot(time_steps, train_input_plt[i, 0, :].cpu().detach().numpy(),
        #         label='y_train_input')
        #     plt.title('2D Curve: x_out_train, x_out_train_prior,y_train_estimation, train_target (Random Batch)')
        #     plt.xlabel('Time Step')
        #     plt.ylabel('Value (Dimension 1)')
        #     plt.legend()
        #     plt.show()

        # Save results
        torch.save({
            'train_input': train_input,
            'train_target': train_target,
            'cv_input': cv_input,
            'cv_target': cv_target,
            'test_input': test_input,
            'test_target': test_target,
            'x_estimation_cv': x_out_cv,
            'x_ture_cv': cv_target_plt,
            'y_estimation_cv': y_cv_estimation,
            'y_ture_cv': cv_input_plt
        }, path_dataset + 'index_error.pt')
 


