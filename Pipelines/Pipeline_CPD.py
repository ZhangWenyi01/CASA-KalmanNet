"""
This file contains the class Pipeline_CPD, 
which is used to train and test CPDNetwork.
"""

import torch
import torch.nn as nn
import random
import time
from Plot import Plot_extended
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

class Pipeline_CPD:

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
        self.loss_fn = nn.BCELoss(reduction='mean')
        self.sample_interval = args.sample_interval

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

    def CPDNNTest(self, SysModel, test_input, test_target, path_results,x_estimation_cv,x_ture_cv,
                  y_estimation_cv,y_ture_cv,threshold=0.5,MaskOnState=False,\
     randomInit=False,test_init=None,load_model=False,load_model_path=None,\
        test_lengthMask=None):
        # Load model
        if load_model:
            self.model = torch.load(load_model_path, map_location=self.device,weights_only=False) 
        else:
            self.model = torch.load(path_results+'best-model.pt', map_location=self.device,weights_only=False) 

        changepoint = SysModel.changepoint
        self.N_T = test_input.shape[0]
        SysModel.T_test = test_input.size()[-1]
        self.MSE_test_linear_arr = torch.zeros([self.N_T])
        x_out_test = torch.zeros([self.N_T, 1,SysModel.T_test]).to(self.device)

        if MaskOnState:
            mask = torch.tensor([True,False,False])
            if SysModel.m == 2: 
                mask = torch.tensor([True,False])

        # BCE LOSS Function
        loss_fn = nn.BCELoss(reduction='mean')

        # Test mode
        self.model.eval()
        self.model.batch_size = self.N_T
        torch.no_grad()

        start = time.time()       
        
        for t in range(0, SysModel.T_test-self.sample_interval+1):
            x_out_test[:,:, t] = self.model(test_input[:,:, t:t+self.sample_interval])
        
        # Randomly select a batch
        # batch_idx = random.randint(0, self.N_T - 1)
        batch_idx = 20

        # Plot the predicted trajectory vs actual trajectory
        time_steps = torch.arange(SysModel.T_test).cpu().detach().numpy()
        plt.subplot(1, 2, 1)
        plt.plot(x_out_test[batch_idx, 0, :].detach().cpu().numpy(), label="Predicted CPD probability", linestyle='--')
        plt.plot(test_target[batch_idx, 0, :].detach().cpu().numpy(), label="Actual CPD probability", linestyle='-')
        plt.axvline(x=changepoint, color='green', linestyle='--', label=f'Changepoint ({changepoint})')
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.title(f"Trajectory Comparison for Batch {batch_idx}")
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(x_estimation_cv[batch_idx, 0, :].cpu().detach().numpy(),
            label='estimation state', color='red')
        plt.plot(x_ture_cv[batch_idx, 0, :].cpu().detach().numpy(),
            label='true state', color='blue')
        plt.plot(y_estimation_cv[batch_idx, 0, :].cpu().detach().numpy(),
            label='estimation y', color='black')
        plt.plot(y_ture_cv[batch_idx, 0, :].cpu().detach().numpy(),
            label='true y')
        plt.axvline(x=changepoint, color='green', linestyle='--', label=f'Changepoint ({changepoint})')
        plt.title('2D Curve: x_out_train, x_out_train_prior,y_train_estimation, train_target (Random Batch)')
        plt.xlabel('Time Step')
        plt.ylabel('Value (Dimension 1)')
        plt.legend()
        plt.show()
        
        # 将x_ouot_test中每一个点的数值与threshold对比，超过threshold的点设为1，否则设为0
        x_out_test_compared = (x_out_test > threshold).float()
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

    def CPDNNTrain(self, SysModel,cv_input, cv_target, train_input, train_target, path_results, \
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
        
        # Load checkpoint if exists
        batch = -1
        resume_train = False
        if resume_train is True:
            checkpoint_path = path_results + 'checkpoint.pt'
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.MSE_train_linear_epoch[:len(checkpoint['MSE_train_linear_epoch'])] = checkpoint['MSE_train_linear_epoch']
            self.MSE_cv_linear_epoch[:len(checkpoint['MSE_cv_linear_epoch'])] = checkpoint['MSE_cv_linear_epoch']
            self.MSE_train_dB_epoch[:len(checkpoint['MSE_train_dB_epoch'])] = checkpoint['MSE_train_dB_epoch']
            self.MSE_cv_dB_epoch[:len(checkpoint['MSE_cv_dB_epoch'])] = checkpoint['MSE_cv_dB_epoch']
            self.MSE_cv_dB_opt = checkpoint['MSE_cv_dB_opt']
            self.MSE_cv_idx_opt = checkpoint['MSE_cv_idx_opt']
            batch = checkpoint['Batch']

        # Training phase. In the training phase, 5 sequences are taken from the 
        # train_input dataset each time and fed into the model for training. 
        # Then shift one position to the right and take another 5 sequences 
        # for training.
        for ti in range(batch+1, self.N_steps):

            ###############################
            ### Training Sequence Batch ###
            ###############################
            self.optimizer.zero_grad()
            # Training Mode
            self.model.train()
            self.model.batch_size = self.N_B

            # Init Training Batch tensors
            y_training_batch = torch.zeros([self.N_B, SysModel.n, SysModel.T-self.sample_interval+1]).to(self.device)
            train_target_batch = torch.zeros([self.N_B, 1, SysModel.T-self.sample_interval+1]).to(self.device)
            x_out_training_batch = torch.zeros([self.N_B, 1, SysModel.T-self.sample_interval+1]).to(self.device)
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
            
            # Forward Computation
            for t in range(0, SysModel.T-self.sample_interval+1):
                x_out_training_batch[:, :, t] = self.model(y_training_batch[:, :, t:t+self.sample_interval])
            
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

            ########################
            ### Validation Sequence Batch ###
            ########################

            # Cross Validation Mode
            self.model.eval()
            self.model.batch_size = self.N_CV
            # # Init Hidden State
            # self.model.init_hidden_KNet()
            with torch.no_grad():

                # SysModel.T_test = cv_input.size()[-1] # T_test is the maximum length of the CV sequences

                x_out_cv_batch = torch.empty([self.N_CV, 1, SysModel.T_test-self.sample_interval+1]).to(self.device)

                for t in range(0, SysModel.T_test-self.sample_interval+1):
                    x_out_cv_batch[:, :, t] = self.model(cv_input[:, :, t:t+self.sample_interval])
                
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
                
                if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti
                    
                    torch.save(self.model, path_results + 'best-model.pt')

            ### Training Summary ###
            ########################        
            print(ti, "MSE Training :", self.MSE_train_linear_epoch[ti], "MSE Validation :", self.MSE_cv_linear_epoch[ti])
                      
            if (ti > 1):
                d_train = self.MSE_train_linear_epoch[ti] - self.MSE_train_linear_epoch[ti - 1]
                d_cv = self.MSE_cv_linear_epoch[ti] - self.MSE_cv_linear_epoch[ti - 1]
                print("diff MSE Training :", d_train,  "diff MSE Validation :", d_cv)

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt)
            
            ########################
            #### Save Checkpoint ###
            ########################
            # Save checkpoint
            checkpoint = {
                'epoch': ti,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'MSE_train_linear_epoch': self.MSE_train_linear_epoch[:ti+1],
                'MSE_cv_linear_epoch': self.MSE_cv_linear_epoch[:ti+1],
                'MSE_train_dB_epoch': self.MSE_train_dB_epoch[:ti+1],
                'MSE_cv_dB_epoch': self.MSE_cv_dB_epoch[:ti+1],
                'MSE_cv_dB_opt': self.MSE_cv_dB_opt,
                'MSE_cv_idx_opt': self.MSE_cv_idx_opt,
                'Batch': ti
            }
            torch.save(checkpoint, path_results + 'checkpoint.pt')

            
            ########################
            ###      Summary     ###
            ########################
            self.writer.add_scalar('Loss/train', self.MSE_train_linear_epoch[ti], ti)
            self.writer.add_scalar('Loss/validation', self.MSE_cv_linear_epoch[ti], ti)

        self.writer.close()

        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]

