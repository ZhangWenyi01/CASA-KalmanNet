"""
This file contains the class Pipeline_EKF, 
which is used to train and test KalmanNet.
"""

import torch
import torch.nn as nn
import random
import time
from Plot import Plot_extended
import matplotlib.pyplot as plt

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
            train_target_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T-self.sample_interval+1]).to(self.device)
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

            # Add training loss to TensorBoard
            self.writer.add_scalar('Loss/train', self.MSE_train_linear_epoch[ti], ti)
            self.writer.add_scalar('Loss/train_dB', self.MSE_train_dB_epoch[ti], ti)

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
                self.writer.add_scalar('Loss/validation', self.MSE_cv_linear_epoch[ti], ti)
                self.writer.add_scalar('Loss/validation_dB', self.MSE_cv_dB_epoch[ti], ti)
                
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
                
                # Add loss differences to TensorBoard
                self.writer.add_scalar('Loss/train_diff', d_train, ti)
                self.writer.add_scalar('Loss/validation_diff', d_cv, ti)

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
    
    def CPD_Dataset(self, SysModel,train_input, train_target,train_priorX, cv_input, cv_target,cv_priorX, path_results, path_dataset, MaskOnState=False,\
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
        
        x_out_train = torch.zeros([self.N_T, SysModel.m, SysModel.T_test]).to(self.device)
        x_out_cv = torch.zeros([self.N_CV, SysModel.m, SysModel.T_cv]).to(self.device)

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
            x_out_train[:,:, t] = torch.squeeze(self.model(torch.unsqueeze(train_input[:,:, t],2)))

        # Process cv data
        self.model.batch_size = self.N_CV
        self.model.init_hidden_KNet()  # Reset hidden state
        if (randomInit):
            self.model.InitSequence(cv_init, SysModel.T_cv)               
        else:
            self.model.InitSequence(SysModel.m1x_0.reshape(1,SysModel.m,1).repeat(self.N_CV,1,1), SysModel.T_cv)
        
        for t in range(0, SysModel.T_cv):
            x_out_cv[:,:, t] = torch.squeeze(self.model(torch.unsqueeze(cv_input[:,:, t],2)))

        # Calculate error and index
        error_train = torch.mean(torch.abs(x_out_train - train_target), dim=1, keepdim=True)
        error_cv = torch.mean(torch.abs(x_out_cv - cv_target), dim=1, keepdim=True)
        error_train_prior = torch.mean(torch.abs(x_out_train - train_priorX), dim=1, keepdim=True)
        error_cv_prior = torch.mean(torch.abs(x_out_cv - cv_priorX), dim=1, keepdim=True)

        # Normalize along the 3rd dimension with max=3 and min=0
        # Normalize along the 3rd dimension with max=3 and min=0
        error_train = (error_train - torch.min(error_train, dim=2, keepdim=True)[0]) / \
             (torch.max(error_train, dim=2, keepdim=True)[0] - torch.min(error_train, dim=2, keepdim=True)[0]) * 4
        error_cv = (error_cv - torch.min(error_cv, dim=2, keepdim=True)[0]) / \
               (torch.max(error_cv, dim=2, keepdim=True)[0] - torch.min(error_cv, dim=2, keepdim=True)[0]) * 4
        tanh_index_test = torch.tanh(error_train)
        tanh_index_cv = torch.tanh(error_cv)
        
        error_train_prior = (error_train_prior - torch.min(error_train_prior, dim=2, keepdim=True)[0]) / \
             (torch.max(error_train_prior, dim=2, keepdim=True)[0] - torch.min(error_train_prior, dim=2, keepdim=True)[0]) * 4
        error_cv_prior = (error_cv_prior - torch.min(error_cv_prior, dim=2, keepdim=True)[0]) / \
               (torch.max(error_cv_prior, dim=2, keepdim=True)[0] - torch.min(error_cv_prior, dim=2, keepdim=True)[0]) * 4
        

        # # Select a batch from the normalized error_test
        # batch_index = random.randint(0, 99)  # Select a random batch index within the range [0, 99]
        # selected_batch = error_test[batch_index, 0, :].cpu().detach().numpy()

        # # Create a figure with two subplots side by side
        # fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # # Plot the selected batch as a line plot on the left
        # axes[0].plot(selected_batch, alpha=0.7, color='blue')
        # axes[0].set_title(f'Line Plot of Batch {batch_index}')
        # axes[0].set_xlabel('Time')
        # axes[0].set_ylabel('Value')
        # axes[0].grid(True)

        # # Plot the Tanh Index of the selected batch on the right
        # axes[1].plot(tanh_index_test[batch_index, 0, :].cpu().detach().numpy())
        # axes[1].set_title(f'Tanh Index of Batch {batch_index}')
        # axes[1].set_xlabel('Time')
        # axes[1].set_ylabel('Value')
        # axes[1].grid(True)

        # # Adjust layout and show the figure
        # plt.tight_layout()
        # plt.show()
        
        # Calculate MSE
        sample_interval = self.sample_interval
        tanh_index_sample_test = torch.zeros((tanh_index_test.size(0), tanh_index_test.size(1), tanh_index_test.size(2) - sample_interval+1))
        tanh_index_sample_cv = torch.zeros((tanh_index_cv.size(0), tanh_index_cv.size(1), tanh_index_cv.size(2) - sample_interval+1))
        
        for i in range(tanh_index_test.size(2) - sample_interval+1):
            mse_test = torch.mean((tanh_index_test[:, :, i:i+sample_interval]) ** 2, dim=2)
            tanh_index_sample_test[:, :, i] = mse_test
        
        for i in range(tanh_index_cv.size(2) - sample_interval+1):
            mse_cv = torch.mean((tanh_index_cv[:, :, i:i+sample_interval]) ** 2, dim=2)
            tanh_index_sample_cv[:, :, i] = mse_cv
        
        # Save results
        torch.save({
            'index_train': tanh_index_sample_test, 
            'error_train': error_train_prior, 
            'index_cv': tanh_index_sample_cv, 
            'error_cv': error_cv_prior
        }, path_dataset + 'index_error.pt')
 


