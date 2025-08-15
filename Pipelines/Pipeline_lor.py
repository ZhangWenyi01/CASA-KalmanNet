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
import csv
from Simulations.utils import cpd_dataset_process_lor

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

        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]

    def NNTest(self, SysModel, test_input, test_target, path_results, MaskOnState=False,\
     randomInit=False,test_init=None,load_model=False,load_model_path=None,\
        test_lengthMask=None):
        # Load model
        if load_model:
            self.model = torch.load(load_model_path, map_location=self.device) 
        else:
            self.model = torch.load(path_results+'lor-best-model.pt', map_location=self.device) 

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
        
        # Figure plot
        # Randomly select a batch
        random_batch_index = random.randint(0, self.N_T - 1)

        # Extract 3D state components for x_out_test and test_target
        x_out_test_3d = x_out_test[random_batch_index, :, :].cpu().detach().numpy()
        test_target_3d = test_target[random_batch_index, :, :].cpu().detach().numpy()

        # # Create a single 3D plot
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111, projection='3d')

        # # Plot x_out_test 3D trajectory
        # ax.plot(x_out_test_3d[0, :], x_out_test_3d[1, :], x_out_test_3d[2, :], label='x_out_test', color='red')

        # # Plot test_target 3D trajectory
        # ax.plot(test_target_3d[0, :], test_target_3d[1, :], test_target_3d[2, :], label='test_target', color='blue')

        # ax.set_title('3D Trajectory: x_out_test vs test_target')
        # ax.set_xlabel('Dimension 1')
        # ax.set_ylabel('Dimension 2')
        # ax.set_zlabel('Dimension 3')
        # ax.legend()

        # # Show the plots
        # plt.tight_layout()
        # plt.show()
        
        
        # # Plot 2D comparison for the first dimension
        # plt.figure(figsize=(8, 6))

        # # Plot x_out_test for the first dimension
        # plt.plot(x_out_test[random_batch_index, 0, :].cpu().detach().numpy(), label='x_out_test', color='red')

        # # Plot test_target for the first dimension
        # plt.plot(test_target[random_batch_index, 0, :].cpu().detach().numpy(), label='test_target', color='blue')

        # plt.title('2D Comparison: x_out_test vs test_target (X Dimension)')
        # plt.xlabel('Time Step')
        # plt.ylabel('Value')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        
        # # Plot 2D comparison for the first dimension
        # plt.figure(figsize=(8, 6))

        # # Plot x_out_test for the first dimension
        # plt.plot(x_out_test[random_batch_index, 1, :].cpu().detach().numpy(), label='x_out_test', color='red')

        # # Plot test_target for the first dimension
        # plt.plot(test_target[random_batch_index, 1, :].cpu().detach().numpy(), label='test_target', color='blue')

        # plt.title('2D Comparison: x_out_test vs test_target (Y Dimension)')
        # plt.xlabel('Time Step')
        # plt.ylabel('Value')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        
        # # Plot 2D comparison for the first dimension
        # plt.figure(figsize=(8, 6))

        # # Plot x_out_test for the first dimension
        # plt.plot(x_out_test[random_batch_index, 2, :].cpu().detach().numpy(), label='x_out_test', color='red')

        # # Plot test_target for the first dimension
        # plt.plot(test_target[random_batch_index, 2, :].cpu().detach().numpy(), label='test_target', color='blue')

        # plt.title('2D Comparison: x_out_test vs test_target (Z Dimension)')
        # plt.xlabel('Time Step')
        # plt.ylabel('Value')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        
        
        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)
        
        # Loss on single dimension
        MSE_test_dB_avg_x = 10 * torch.log10(torch.mean((x_out_test[:, 0, :] - test_target[:, 0, :]) ** 2))
        MSE_test_dB_avg_y = 10 * torch.log10(torch.mean((x_out_test[:, 1, :] - test_target[:, 1, :]) ** 2))
        MSE_test_dB_avg_z = 10 * torch.log10(torch.mean((x_out_test[:, 2, :] - test_target[:, 2, :]) ** 2))

        # Standard deviation
        self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)

        # Confidence interval
        self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg

        # Update MSE.csv with new data
        mse_file_path = os.path.join(self.folderName, "MSE.csv")
        if not os.path.exists(mse_file_path):
            # Create the file with headers if it doesn't exist
            with open(mse_file_path, mode='w', newline='') as mse_file:
                mse_writer = csv.writer(mse_file)
                mse_writer.writerow(["Model Name", "Metric", "Value", "Unit"])

        # Read existing data
        existing_data = {}
        if os.path.exists(mse_file_path):
            with open(mse_file_path, mode='r') as mse_file:
                mse_reader = csv.reader(mse_file)
                for row in mse_reader:
                    if len(row) >= 2:
                        key = (row[0], row[1])  # (Model Name, Metric)
                        existing_data[key] = row

        # Update or append new data
        new_data = {
            (self.modelName, "MSE Test X"): [self.modelName, "MSE Test X", MSE_test_dB_avg_x.item()],
            (self.modelName, "MSE Test Y"): [self.modelName, "MSE Test Y", MSE_test_dB_avg_y.item()],
            (self.modelName, "MSE Test Z"): [self.modelName, "MSE Test Z", MSE_test_dB_avg_z.item()]
        }

        for key, value in new_data.items():
            if key in existing_data:
                existing_data[key].extend(value[2:])  # Append new values
            else:
                existing_data[key] = value

        # Write updated data back to the file
        with open(mse_file_path, mode='w', newline='') as mse_file:
            mse_writer = csv.writer(mse_file)
            for row in existing_data.values():
                mse_writer.writerow(row)

        # Print MSE and std
        str = self.modelName + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        str = self.modelName + "-" + "STD Test:"
        print(str, self.test_std_dB, "[dB]")
        # Print MSE on single dimension
        str = self.modelName + "-" + "MSE Test X:"
        print(str, MSE_test_dB_avg_x, "[dB]")
        str = self.modelName + "-" + "MSE Test Y:"
        print(str, MSE_test_dB_avg_y, "[dB]")
        str = self.modelName + "-" + "MSE Test Z:"
        print(str, MSE_test_dB_avg_z, "[dB]")
        # Print Run Time
        print("Inference Time:", t)

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_test, t]

    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot_extended(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.N_steps, MSE_KF_dB_avg,
                                self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)

    def CPD_Dataset(self, SysModel,train_input, train_target, cv_input, cv_target,test_input,test_target, path_results, path_dataset, MaskOnState=False,\
        randomInit=False,train_init=None, test_init=None, load_model=False, load_model_path=None,\
            test_lengthMask=None,cv_init=None,scale_param = 8):
        # Load model
        if load_model:
            self.model = torch.load(load_model_path, map_location=self.device,weights_only=False) 
        else:
            self.model = torch.load(path_results+'lor-best-model.pt', map_location=self.device,weights_only=False) 

        self.N_T = train_input.shape[0]
        self.N_CV = cv_input.shape[0]
        self.N_Test = test_input.shape[0]
        SysModel.T_test = test_input.size()[-1]
        SysModel.T_cv = cv_input.size()[-1]
        
        x_out_train = torch.zeros([self.N_T, SysModel.m, SysModel.T]).to(self.device)
        x_out_train_prior = torch.zeros([self.N_T, SysModel.m, SysModel.T]).to(self.device)
        
        x_out_cv = torch.zeros([self.N_CV, SysModel.m, SysModel.T_cv]).to(self.device)
        x_out_cv_prior = torch.zeros([self.N_CV, SysModel.m, SysModel.T_cv]).to(self.device)
        
        x_out_test = torch.zeros([self.N_Test, SysModel.m, SysModel.T_test]).to(self.device)
        x_out_test_prior = torch.zeros([self.N_Test, SysModel.m, SysModel.T_test]).to(self.device)
        
        y_train_estimation = torch.zeros([self.N_T, SysModel.n, SysModel.T]).to(self.device)
        
        y_cv_estimation = torch.zeros([self.N_CV, SysModel.n, SysModel.T_cv]).to(self.device)
        
        y_test_estimation = torch.zeros([self.N_Test, SysModel.n, SysModel.T_test]).to(self.device)


        # Test mode
        self.model.eval()
        torch.no_grad()
        
        train_input_plt = train_input
        train_target_plt = train_target
        cv_input_plt = cv_input
        cv_target_plt = cv_target
        test_input_plt = test_input
        test_target_plt = test_target

        # Process train data
        self.model.batch_size = self.N_T
        self.model.init_hidden_KNet()  # Reset hidden state
        if (randomInit):
            self.model.InitSequence(train_init, SysModel.T)               
        else:
            self.model.InitSequence(SysModel.m1x_0.reshape(1,SysModel.m,1).repeat(self.N_T,1,1), SysModel.T)
        
        for t in range(0, SysModel.T):
            output = self.model(torch.unsqueeze(train_input[:,:, t],2))
            x_out_train[:,:, t] = torch.squeeze(output, dim=2)
            x_out_train_prior[:,:, t] = torch.squeeze(self.model.m1x_prior,dim=2)
            y_train_estimation[:,:, t] = torch.squeeze(self.model.m1y, dim=2)

        
        # Process cv data
        self.model.batch_size = self.N_CV
        self.model.init_hidden_KNet()  # Reset hidden state
        if (randomInit):
            self.model.InitSequence(cv_init, SysModel.T_cv)               
        else:
            self.model.InitSequence(SysModel.m1x_0.reshape(1,SysModel.m,1).repeat(self.N_CV,1,1), SysModel.T_cv)
        for t in range(0, SysModel.T_cv):
            output = self.model(torch.unsqueeze(cv_input[:,:, t],2))
            x_out_cv[:,:, t] = torch.squeeze(output,dim=2)
            x_out_cv_prior[:,:, t] = torch.squeeze(self.model.m1x_prior,dim=2)
            y_cv_estimation[:,:, t] = torch.squeeze(self.model.m1y, dim=2)
            
        # Process test data
        self.model.batch_size = self.N_Test
        self.model.init_hidden_KNet()  # Reset hidden state
        if (randomInit):
            self.model.InitSequence(test_init, SysModel.T_test)
        else:
            self.model.InitSequence(SysModel.m1x_0.reshape(1,SysModel.m,1).repeat(self.N_T,1,1), SysModel.T_test)
        for t in range(0, SysModel.T_test):
            output = self.model(torch.unsqueeze(test_input[:,:, t],2))
            x_out_test[:,:, t] = torch.squeeze(output,dim=2)
            x_out_test_prior[:,:, t] = torch.squeeze(self.model.m1x_prior,dim=2)
            y_test_estimation[:,:, t] = torch.squeeze(self.model.m1y, dim=2)

        
        train_target = cpd_dataset_process_lor(x_out_train,
                                                train_target,
                                                sample_interval=self.sample_interval,
                                                scale_param=scale_param)
        train_input = cpd_dataset_process_lor(y_train_estimation,
                                                train_input,
                                                sample_interval=self.sample_interval,
                                                scale_param=scale_param)
        cv_target = cpd_dataset_process_lor(x_out_cv,
                                                cv_target,
                                                sample_interval=self.sample_interval,
                                                scale_param=scale_param)
        cv_input = cpd_dataset_process_lor(y_cv_estimation,
                                                cv_input,
                                                sample_interval=self.sample_interval,
                                                scale_param=scale_param)
        test_target = cpd_dataset_process_lor(x_out_test,
                                                test_target,
                                                sample_interval=self.sample_interval,
                                                scale_param=scale_param)
        test_input = cpd_dataset_process_lor(y_test_estimation,
                                                test_input,
                                                sample_interval=self.sample_interval,
                                                scale_param=scale_param)
       
        # Save results
        torch.save({
            # Datasets used to train CPDNet
            'train_input': train_input,
            'train_target': train_target,
            'cv_input': cv_input,
            'cv_target': cv_target,
            'test_input': test_input,
            'test_target': test_target,
            # Test datasets used from KalmanNet
            'x_estimation_test': x_out_test,
            'x_ture_test': test_target_plt,
            'y_estimation_test': y_test_estimation,
            'y_ture_test': test_input_plt,
            # Train datasets used from KalmanNet
            'x_estimation_train': x_out_train,
            'x_ture_train': train_target_plt,
            'y_estimation_train': y_train_estimation,
            'y_ture_train': train_input_plt,
            # Cross Validation datasets used from KalmanNet
            'x_estimation_cv': x_out_cv,
            'x_ture_cv': cv_target_plt,
            'y_estimation_cv': y_cv_estimation,
            'y_ture_cv': cv_input_plt
        }, path_dataset + 'index_error.pt')