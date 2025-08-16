import torch.nn as nn
import torch
import time
from Filters.EKF import ExtendedKalmanFilter


def EKFTest(args, SysModel, test_input, test_target, allStates=True,\
     randomInit = False,test_init=None, test_lengthMask=None,
     changepoint = None, changeparameters: dict = None):
    
    # Check changepoint parameters
    if (changepoint is None and changeparameters is not None) or (changepoint is not None and changeparameters is None):
        raise ValueError("changepoint and changeparameters must be provided together or not at all")
    elif (changepoint is not None and changeparameters is not None):
        change_happened = True
    else:
        change_happened = False
    
    # Extract change point parameters
    if change_happened:
        Q_after = changeparameters['Q']
        R_after = changeparameters['R']
        F_after = changeparameters['F']
        H_after = changeparameters['H']
        changed_param = changeparameters['changed_param']
    
    # Number of test samples
    N_T = test_target.size()[0]
    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')  
    # MSE [Linear]
    MSE_EKF_linear_arr = torch.zeros(N_T)
    # Allocate empty tensor for output
    EKF_out = torch.zeros([N_T, SysModel.m, test_input.size()[2]]) # N_T x m x T
    KG_array = torch.zeros([N_T, SysModel.m, SysModel.n, test_input.size()[2]]) # N_T x m x n x T
    
    if not allStates:
        loc = torch.tensor([True,False,False]) # for position only
        if SysModel.m == 2: 
            loc = torch.tensor([True,False]) # for position only

    start = time.time()
    EKF = ExtendedKalmanFilter(SysModel, args)
    
    # Init and Forward Computation   
    if(randomInit):
        EKF.Init_batched_sequence(test_init, SysModel.m2x_0.view(1,SysModel.m,SysModel.m).expand(N_T,-1,-1))        
    else:
        EKF.Init_batched_sequence(SysModel.m1x_0.view(1,SysModel.m,1).expand(N_T,-1,-1), SysModel.m2x_0.view(1,SysModel.m,SysModel.m).expand(N_T,-1,-1))           
    
    # Handle changepoint processing
    if change_happened:
        # Process data before changepoint
        EKF.GenerateBatch(test_input[:, :, :changepoint])
        X_before = EKF.x
        
        # Save state and covariance at change point to ensure continuity
        final_state = EKF.m1x_posterior.clone()
        final_covariance = EKF.m2x_posterior.clone()
        
        # Update parameters at change point
        if changed_param == 'Q':
            EKF.Q = Q_after.to(EKF.device)
        elif changed_param == 'R':
            EKF.R = R_after.to(EKF.device)
        
        # Re-initialize EKF using state at change point as new initial condition
        EKF.Init_batched_sequence(final_state, final_covariance)
        
        # Continue processing data after change point
        EKF.GenerateBatch(test_input[:, :, changepoint:])
        EKF_out = torch.cat([X_before, EKF.x], dim=2)
        
        # Note: KG_array from second part will overwrite, this is acceptable for EKF
        KG_array = EKF.KG_array
    else:
        EKF.GenerateBatch(test_input)
        EKF_out = EKF.x
        KG_array = EKF.KG_array
     
    end = time.time()
    t = end - start

    # MSE loss
    for j in range(N_T):# cannot use batch due to different length and std computation   
        if(allStates):
            if args.randomLength:
                MSE_EKF_linear_arr[j] = loss_fn(EKF_out[j,:,test_lengthMask[j]], test_target[j,:,test_lengthMask[j]]).item()
            else:      
                MSE_EKF_linear_arr[j] = loss_fn(EKF_out[j,:,:], test_target[j,:,:]).item()
        else: # mask on state
            if args.randomLength:
                MSE_EKF_linear_arr[j] = loss_fn(EKF_out[j,loc,test_lengthMask[j]], test_target[j,loc,test_lengthMask[j]]).item()
            else:           
                MSE_EKF_linear_arr[j] = loss_fn(EKF_out[j,loc,:], test_target[j,loc,:]).item()

    MSE_EKF_linear_avg = torch.mean(MSE_EKF_linear_arr)
    MSE_EKF_dB_avg = 10 * torch.log10(MSE_EKF_linear_avg)

    # Standard deviation
    MSE_EKF_linear_std = torch.std(MSE_EKF_linear_arr, unbiased=True)

    # Confidence interval
    EKF_std_dB = 10 * torch.log10(MSE_EKF_linear_std + MSE_EKF_linear_avg) - MSE_EKF_dB_avg
    
    print("Extended Kalman Filter - MSE LOSS:", MSE_EKF_dB_avg, "[dB]")
    print("Extended Kalman Filter - STD:", EKF_std_dB, "[dB]")
    # Print Run Time
    print("Inference Time:", t)

    return [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out]


