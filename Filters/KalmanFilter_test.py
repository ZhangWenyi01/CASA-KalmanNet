import torch
import torch.nn as nn
import time
from Filters.Linear_KF import KalmanFilter
    

def KFTest(args, SysModel, test_input, test_target, allStates=True,\
     randomInit = False, test_init=None, test_lengthMask=None,
     changepoint = None,changeparameters:dict = None):
    
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

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_KF_linear_arr = torch.zeros(args.N_T)
    # allocate memory for KF output
    KF_out = torch.zeros(args.N_T, SysModel.m, args.T_test)
    if not allStates:
        loc = torch.tensor([True,False,False]) # for position only
        if SysModel.m == 2: 
            loc = torch.tensor([True,False]) # for position only

    start = time.time()

    KF = KalmanFilter(SysModel, args)
    # Init and Forward Computation 
    if(randomInit):
        KF.Init_batched_sequence(test_init, SysModel.m2x_0.view(1,SysModel.m,SysModel.m).expand(args.N_T,-1,-1))        
    else:
        KF.Init_batched_sequence(SysModel.m1x_0.view(1,SysModel.m,1).expand(args.N_T,-1,-1), SysModel.m2x_0.view(1,SysModel.m,SysModel.m).expand(args.N_T,-1,-1))           
    
    # 修改GenerateBatch过程以处理参数变化
    if change_happened:
        # 首先处理到变化点之前的数据
        KF.GenerateBatch(test_input[:, :, :changepoint])
        X_before = KF.x
        # # Calculate MSE for the first part of the trajectory
        # loss_before = torch.mean(loss_fn(X_before,test_target[:,:,:changepoint]))
        # MSE_KF_before_dB = 10 * torch.log10(loss_before)
        # print("Kalman Filter (before change) - MSE LOSS:", MSE_KF_before_dB, "[dB]")
        
        # 在变化点处更新参数
        if changed_param == 'Q':
            KF.Q = Q_after.to(KF.device)
        elif changed_param == 'R':
            KF.R = R_after.to(KF.device)
        elif changed_param == 'F':
            KF.F = F_after
        elif changed_param == 'H':
            KF.H = H_after
            
        # 继续处理变化点之后的数据
        KF.GenerateBatch(test_input[:, :, changepoint:])
        x_after = torch.cat([X_before, KF.x], dim=2)
        KF_out = x_after
        
    else:
        KF.GenerateBatch(test_input)
        KF_out = KF.x
    
    end = time.time()
    t = end - start
    
    # MSE loss
    for j in range(args.N_T):# cannot use batch due to different length and std computation   
        if(allStates):
            if args.randomLength:
                MSE_KF_linear_arr[j] = loss_fn(KF_out[j,:,test_lengthMask[j]], test_target[j,:,test_lengthMask[j]]).item()
            else:      
                MSE_KF_linear_arr[j] = loss_fn(KF_out[j,:,:], test_target[j,:,:]).item()
        else: # mask on state
            if args.randomLength:
                MSE_KF_linear_arr[j] = loss_fn(KF_out[j,loc,test_lengthMask[j]], test_target[j,loc,test_lengthMask[j]]).item()
            else:           
                MSE_KF_linear_arr[j] = loss_fn(KF_out[j,loc,:], test_target[j,loc,:]).item()

    MSE_KF_linear_avg = torch.mean(MSE_KF_linear_arr)
    MSE_KF_dB_avg = 10 * torch.log10(MSE_KF_linear_avg)

    # Standard deviation
    MSE_KF_linear_std = torch.std(MSE_KF_linear_arr, unbiased=True)

    # Confidence interval
    KF_std_dB = 10 * torch.log10(MSE_KF_linear_std + MSE_KF_linear_avg) - MSE_KF_dB_avg

    print("Kalman Filter - MSE LOSS:", MSE_KF_dB_avg, "[dB]")
    print("Kalman Filter - STD:", KF_std_dB, "[dB]")
    # Print Run Time
    print("Inference Time:", t)
    return [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out]