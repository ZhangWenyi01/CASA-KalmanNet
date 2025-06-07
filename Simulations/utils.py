"""
The file contains utility functions for the simulations.
"""

import torch
import matplotlib.pyplot as plt

def DataGen(args, SysModel_data, fileName):

    ##################################
    ### Generate Training Sequence ###
    ##################################
    SysModel_data.GenerateBatch(args, args.N_E, args.T, randomInit=args.randomInit_train)
    train_input = SysModel_data.Input
    train_target = SysModel_data.Target
    ### init conditions ###
    train_init = SysModel_data.m1x_0_batch #size: N_E x m x 1
    ### length mask ###
    if args.randomLength:
        train_lengthMask = SysModel_data.lengthMask

    ####################################
    ### Generate Validation Sequence ###
    ####################################
    SysModel_data.GenerateBatch(args, args.N_CV, args.T, randomInit=args.randomInit_cv)
    cv_input = SysModel_data.Input
    cv_target = SysModel_data.Target
    cv_init = SysModel_data.m1x_0_batch #size: N_CV x m x 1
    ### length mask ###
    if args.randomLength:
        cv_lengthMask = SysModel_data.lengthMask

    ##############################
    ### Generate Test Sequence ###
    ##############################
    SysModel_data.GenerateBatch(args, args.N_T, args.T_test, randomInit=args.randomInit_test)
    test_input = SysModel_data.Input
    test_target = SysModel_data.Target
    test_init = SysModel_data.m1x_0_batch #size: N_T x m x 1
    ### length mask ###
    if args.randomLength:
        test_lengthMask = SysModel_data.lengthMask

    #################
    ### Save Data ###
    #################
    if(args.randomLength):
        torch.save([train_input, train_target, cv_input, cv_target, test_input, test_target,train_init, cv_init, test_init, train_lengthMask,cv_lengthMask,test_lengthMask], fileName)
    else:
        torch.save([train_input, train_target, cv_input, cv_target, test_input, test_target,train_init, cv_init, test_init], fileName)
    
def DataGenCPD(args, SysModel_data, fileName):

    ##################################
    ### Generate Training Sequence ###
    ##################################
    SysModel_data.GenerateBatchCPD(args, args.N_E, args.T, randomInit=args.randomInit_train)
    train_input = SysModel_data.Input
    train_target = SysModel_data.Target
    ### init conditions ###
    train_init = SysModel_data.m1x_0_batch #size: N_E x m x 1
    train_ChangePoint = SysModel_data.changepoint
    ### length mask ###
    if args.randomLength:
        train_lengthMask = SysModel_data.lengthMask
        

    ####################################
    ### Generate Validation Sequence ###
    ####################################
    SysModel_data.GenerateBatchCPD(args, args.N_CV, args.T, randomInit=args.randomInit_cv)
    cv_input = SysModel_data.Input
    cv_target = SysModel_data.Target
    cv_init = SysModel_data.m1x_0_batch #size: N_CV x m x 1
    cv_ChangePoint = SysModel_data.changepoint
    ### length mask ###
    if args.randomLength:
        cv_lengthMask = SysModel_data.lengthMask

    ##############################
    ### Generate Test Sequence ###
    ##############################
    SysModel_data.GenerateBatchCPD(args, args.N_T, args.T_test, randomInit=args.randomInit_test)
    test_input = SysModel_data.Input
    test_target = SysModel_data.Target
    test_ChangePoint = SysModel_data.changepoint
    test_init = SysModel_data.m1x_0_batch #size: N_T x m x 1
    # import matplotlib.pyplot as plt
    # import numpy as np
    
    # # Randomly select a batch
    # i = np.random.randint(0, test_target.shape[0])
    
    # # Get changepoint index
    # change_index = SysModel_data.changepoint
    
    # # Create figure for plotting
    # plt.figure(figsize=(10, 6))
    
    # # Plot target trajectory
    # x_target = test_target[i, 0, :].cpu().numpy()
    
    # # Plot target trajectory with different styles before and after changepoint
    # plt.plot(x_target, 
    #         color='blue', alpha=0.5, label='Target (before CP)')
    
    # # Plot input observations
    # x_input = test_input[i, 0, :].cpu().detach().numpy()
    
    # # Plot input trajectory with different styles before and after changepoint
    # plt.plot(x_input, 
    #         color='green', alpha=0.3, label='Input ')
    
    # plt.title(f'Test Data Trajectories (Batch {i})')
    # plt.xlabel('X Position')
    # plt.ylabel('Y Position')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
    
    # # Plot 3D trajectories for test data
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # import numpy as np
    
    # # Create figure for plotting
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    
    # # Randomly select a batch
    # i = np.random.randint(0, test_target.shape[0])
    
    # # Get changepoint index
    # change_index = SysModel_data.changepoint
    
    # # Plot target trajectories
    # x = test_target[i, 0, :].cpu().numpy()
    # y = test_target[i, 1, :].cpu().numpy()
    # z = test_target[i, 2, :].cpu().numpy()
    
    # # Plot target trajectory with different styles before and after changepoint
    # ax.plot(x[:change_index], y[:change_index], z[:change_index], 
    #         color='blue', alpha=0.5, label='Target (before CP)')
    # ax.plot(x[change_index-1:], y[change_index-1:], z[change_index-1:], 
    #         color='blue', alpha=0.5, linestyle='--', label='Target (after CP)')
    
    # # Plot input observations
    # x_in = test_input[i, 0, :].cpu().numpy()
    # y_in = test_input[i, 1, :].cpu().numpy()
    # z_in = test_input[i, 2, :].cpu().numpy()
    
    # # Plot input trajectory with different styles before and after changepoint
    # ax.plot(x_in[:change_index], y_in[:change_index], z_in[:change_index], 
    #         color='green', alpha=0.3, label='Input (before CP)')
    # ax.plot(x_in[change_index-1:], y_in[change_index-1:], z_in[change_index-1:], 
    #         color='green', alpha=0.3, linestyle='--', label='Input (after CP)')
    
    # ax.set_title(f'Test Data Trajectories (Batch {i})')
    # ax.legend()
    # plt.tight_layout()
    # plt.show()
    # ### length mask ###
    if args.randomLength:
        test_lengthMask = SysModel_data.lengthMask

    #################
    ### Save Data ###
    #################
    if(args.randomLength):
        torch.save([train_input, train_target, cv_input, cv_target, test_input, test_target,train_init, cv_init, test_init, train_lengthMask,cv_lengthMask,test_lengthMask,train_ChangePoint,cv_ChangePoint,test_ChangePoint], fileName)
    else:
        torch.save([train_input, train_target, cv_input, cv_target, test_input, test_target,train_init, cv_init, test_init,train_ChangePoint,cv_ChangePoint,test_ChangePoint], fileName)
        

def DecimateData(all_tensors, t_gen,t_mod, offset=0):
    
    # ratio: defines the relation between the sampling time of the true process and of the model (has to be an integer)
    ratio = round(t_mod/t_gen)

    i = 0
    all_tensors_out = all_tensors
    for tensor in all_tensors:
        tensor = tensor[:,(0+offset)::ratio]
        if(i==0):
            all_tensors_out = torch.cat([tensor], dim=0).view(1,all_tensors.size()[1],-1)
        else:
            all_tensors_out = torch.cat([all_tensors_out,tensor.view(1,all_tensors.size()[1],-1)], dim=0)
        i += 1

    return all_tensors_out

def Decimate_and_perturbate_Data(true_process, delta_t, delta_t_mod, N_examples, h, lambda_r, offset=0):
    
    # Decimate high resolution process
    decimated_process = DecimateData(true_process, delta_t, delta_t_mod, offset)

    noise_free_obs = getObs(decimated_process,h)

    # Replicate for computation purposes
    decimated_process = torch.cat(int(N_examples)*[decimated_process])
    noise_free_obs = torch.cat(int(N_examples)*[noise_free_obs])


    # Observations; additive Gaussian Noise
    observations = noise_free_obs + torch.randn_like(decimated_process) * lambda_r

    return [decimated_process, observations]

def getObs(sequences, h):
    i = 0
    sequences_out = torch.zeros_like(sequences)
    # sequences_out = torch.zeros_like(sequences)
    for sequence in sequences:
        for t in range(sequence.size()[1]):
            sequences_out[i,:,t] = h(sequence[:,t])
    i = i+1

    return sequences_out

def Short_Traj_Split(data_target, data_input, T):### Random Init is automatically incorporated
    data_target = list(torch.split(data_target,T+1,2)) # +1 to reserve for init
    data_input = list(torch.split(data_input,T+1,2)) # +1 to reserve for init

    data_target.pop()# Remove the last one which may not fullfill length T
    data_input.pop()# Remove the last one which may not fullfill length T

    data_target = torch.squeeze(torch.cat(list(data_target), dim=0))#Back to tensor and concat together
    data_input = torch.squeeze(torch.cat(list(data_input), dim=0))#Back to tensor and concat together
    # Split out init
    target = data_target[:,:,1:]
    input = data_input[:,:,1:]
    init = data_target[:,:,0]
    return [target, input, init]

# Calculate MSE over sample intervals
def calculate_mse_over_intervals(tanh_index, sample_interval):
    num_intervals = tanh_index.size(2) - sample_interval + 1
    mse_result = torch.zeros((tanh_index.size(0), tanh_index.size(1), 
                              num_intervals))
    for i in range(num_intervals):
        mse_result[:, :, i] = torch.mean(
            (tanh_index[:, :, i:i + sample_interval]) ** 2, dim=2
            )
    return mse_result


def cpd_dataset_process(input_one:torch.Tensor,input_two:torch.Tensor,
                        sample_interval:int = 5,
                        scale_param:int = 19)->torch.Tensor:
    # Calculate absolute errors
    abs_errors = torch.abs(input_one[:,0:1,:] - input_two[:,0:1,:])
    # Apply sigmoid transformation, minus 0.5 to center around 0
    transformed_errors = torch.sigmoid(abs_errors) - 0.5
    # Apply tanh transformation, scale_param is a hyperparameter to 
    # control the range
    transformed_errors =  torch.tanh(scale_param*transformed_errors)
    
    # Calculate MSE over sample intervals
    mse_result = calculate_mse_over_intervals(transformed_errors, 
                                              sample_interval)
    return mse_result

def cpd_dataset_process_single(input_one:torch.Tensor,input_two:torch.Tensor,
                        sample_interval:int = 5,
                        scale_param:int = 19)->torch.Tensor:
    # Calculate absolute errors
    abs_errors = torch.abs(input_one[:,0:1] - input_two[:,0:1])
    # Apply sigmoid transformation, minus 0.5 to center around 0
    transformed_errors = torch.sigmoid(abs_errors) - 0.5
    # Apply tanh transformation, scale_param is a hyperparameter to 
    # control the range
    transformed_errors =  torch.tanh(scale_param*transformed_errors)**2

    return transformed_errors

def cpd_dataset_process_lor(input_one:torch.Tensor,input_two:torch.Tensor,
                        sample_interval:int = 5,
                        scale_param:int = 19)->torch.Tensor:
    # Calculate absolute errors
    abs_errors = torch.mean(torch.abs(input_one - input_two), dim=1, keepdim=True)
    # Apply sigmoid transformation, minus 0.5 to center around 0
    transformed_errors = torch.sigmoid(abs_errors) - 0.5
    # Apply tanh transformation, scale_param is a hyperparameter to 
    # control the range
    transformed_errors =  torch.tanh(scale_param*transformed_errors)
    
    # Calculate MSE over sample intervals
    mse_result = calculate_mse_over_intervals(transformed_errors, 
                                              sample_interval)
    return mse_result
