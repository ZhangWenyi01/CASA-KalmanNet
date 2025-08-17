"""
Test function for Unscented Kalman Filter
"""

import torch
import torch.nn as nn
import time
from Filters.UKF import UnscentedKalmanFilter


def UKFTest(args, SysModel, test_input, test_target, allStates=True,
            randomInit=False, test_init=None, test_lengthMask=None,
            changepoint=None, changeparameters: dict = None):
    """
    Test the Unscented Kalman Filter
    
    Args:
        args: configuration arguments
        SysModel: system model
        test_input: test observations
        test_target: ground truth states
        allStates: whether to evaluate all states or position only
        randomInit: whether to use random initialization
        test_init: initial states for random init
        test_lengthMask: mask for variable length sequences
        changepoint: time index where parameter change occurs
        changeparameters: dictionary containing changed parameters
    """
    
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
    MSE_linear_arr = torch.zeros(N_T)
    # Allocate empty tensor for output
    UKF_out = torch.zeros([N_T, SysModel.m, test_input.size()[2]]) # N_T x m x T
    KG_array = torch.zeros([N_T, SysModel.m, SysModel.n, test_input.size()[2]]) # N_T x m x n x T
    
    if not allStates:
        loc = torch.tensor([True,False,False]) # for position only
        if SysModel.m == 2: 
            loc = torch.tensor([True,False]) # for position only

    # Initialize UKF
    UKF = UnscentedKalmanFilter(SysModel, args)

    start = time.time()
    
    # Init and Forward Computation   
    if(randomInit):
        UKF.Init_batched_sequence(test_init, SysModel.m2x_0.view(1,SysModel.m,SysModel.m).expand(N_T,-1,-1))        
    else:
        UKF.Init_batched_sequence(SysModel.m1x_0.view(1,SysModel.m,1).expand(N_T,-1,-1), SysModel.m2x_0.view(1,SysModel.m,SysModel.m).expand(N_T,-1,-1))           
    
    # Handle changepoint processing
    if change_happened:
        # Process data before changepoint
        UKF.GenerateBatch(test_input[:, :, :changepoint])
        X_before = UKF.x
        
        # Save state and covariance at change point to ensure continuity
        # UKF doesn't store posterior state as attribute, so we need to extract from the last time step
        final_state = UKF.x[:, :, -1:].clone()  # Last state from before changepoint
        final_covariance = UKF.m2x_0_batch.clone()  # Use initial covariance as approximation
        
        # Update parameters at change point
        if changed_param == 'Q':
            UKF.Q = Q_after.to(UKF.device)
        elif changed_param == 'R':
            UKF.R = R_after.to(UKF.device)
        elif changed_param == 'F':
            # For UKF, F is a function, update the system model function
            UKF.f = F_after
        elif changed_param == 'H':
            # For UKF, H is a function, update the system model function
            UKF.h = H_after
        
        # Re-initialize UKF using state at change point as new initial condition
        UKF.Init_batched_sequence(final_state, final_covariance)
        
        # Continue processing data after change point
        UKF.GenerateBatch(test_input[:, :, changepoint:])
        UKF_out = torch.cat([X_before, UKF.x], dim=2)
        KG_array = UKF.KG_array
    else:
        UKF.GenerateBatch(test_input)
        UKF_out = UKF.x
        KG_array = UKF.KG_array
     
    end = time.time()
    t = end - start

    # MSE loss
    for j in range(N_T):# cannot use batch due to different length and std computation   
        if(allStates):
            if args.randomLength:
                MSE_linear_arr[j] = loss_fn(UKF_out[j,:,test_lengthMask[j]], test_target[j,:,test_lengthMask[j]]).item()
            else:      
                MSE_linear_arr[j] = loss_fn(UKF_out[j,:,:], test_target[j,:,:]).item()
        else: # mask on state
            if args.randomLength:
                MSE_linear_arr[j] = loss_fn(UKF_out[j,loc,test_lengthMask[j]], test_target[j,loc,test_lengthMask[j]]).item()
            else:           
                MSE_linear_arr[j] = loss_fn(UKF_out[j,loc,:], test_target[j,loc,:]).item()

    MSE_linear_avg = torch.mean(MSE_linear_arr)
    MSE_dB_avg = 10 * torch.log10(MSE_linear_avg)

    # Standard deviation
    MSE_linear_std = torch.std(MSE_linear_arr, unbiased=True)

    # Confidence interval
    std_dB = 10 * torch.log10(MSE_linear_std + MSE_linear_avg) - MSE_dB_avg
    
    print("Unscented Kalman Filter - MSE LOSS:", MSE_dB_avg, "[dB]")
    print("Unscented Kalman Filter - STD:", std_dB, "[dB]")
    # Print Run Time
    print("Inference Time:", t)

    return [MSE_linear_arr, MSE_linear_avg, MSE_dB_avg, KG_array, UKF_out]
