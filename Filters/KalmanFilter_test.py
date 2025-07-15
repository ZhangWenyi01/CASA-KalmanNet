import torch
import torch.nn as nn
import time
from Filters.Linear_KF import KalmanFilter
    

def KFTest(args, SysModel, test_input, test_target, allStates=True,\
     randomInit = False, test_init=None, test_lengthMask=None,
     changepoint = None,changeparameters:dict = None, first_dim_only=False):
    
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

    # MSE [Linear] - modify to support dimension-wise MSE when first_dim_only=True
    if first_dim_only:
        # Store MSE for position, velocity, acceleration separately
        MSE_KF_linear_arr = torch.zeros(args.N_T, 3)  # 3 dimensions: pos, vel, acc
        MSE_KF_linear_avg = torch.zeros(3)
        MSE_KF_dB_avg = torch.zeros(3)
    else:
        MSE_KF_linear_arr = torch.zeros(args.N_T)
        
    # allocate memory for KF output
    KF_out = torch.zeros(args.N_T, SysModel.m, args.T_test)
    
    # 确定要计算误差的维度
    if not first_dim_only and not allStates:
        # 原来的逻辑：计算位置误差
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
        
        # 🔥 保存变点处的状态和协方差，确保连续性
        final_state = KF.m1x_posterior.clone()  # 变点前的最终状态
        final_covariance = KF.m2x_posterior.clone()  # 变点前的最终协方差
        
        # 在变化点处更新参数
        if changed_param == 'Q':
            KF.Q = Q_after.to(KF.device)
        elif changed_param == 'R':
            KF.R = R_after.to(KF.device)
        elif changed_param == 'F':
            KF.F = F_after
        elif changed_param == 'H':
            KF.H = H_after
        
        # 🔥 重新初始化KF，使用变点处的状态作为新的初始条件
        KF.Init_batched_sequence(final_state, final_covariance)
            
        # 继续处理变化点之后的数据
        KF.GenerateBatch(test_input[:, :, changepoint:])
        x_after = torch.cat([X_before, KF.x], dim=2)
        KF_out = x_after
        
    else:
        KF.GenerateBatch(test_input)
        KF_out = KF.x
    
    end = time.time()
    t = end - start
    
    # MSE loss calculation
    for j in range(args.N_T):# cannot use batch due to different length and std computation   
        if first_dim_only:
            # Calculate MSE for each dimension separately: position, velocity, acceleration
            # Position (dimension 0)
            if args.randomLength:
                MSE_KF_linear_arr[j,0] = loss_fn(KF_out[j,0:1,test_lengthMask[j]], test_target[j,0:1,test_lengthMask[j]]).item()
            else:      
                MSE_KF_linear_arr[j,0] = loss_fn(KF_out[j,0:1,:], test_target[j,0:1,:]).item()
            
            # Velocity (dimension 1) if available
            if SysModel.m > 1:
                if args.randomLength:
                    MSE_KF_linear_arr[j,1] = loss_fn(KF_out[j,1:2,test_lengthMask[j]], test_target[j,1:2,test_lengthMask[j]]).item()
                else:      
                    MSE_KF_linear_arr[j,1] = loss_fn(KF_out[j,1:2,:], test_target[j,1:2,:]).item()
            
            # Acceleration (dimension 2) if available
            if SysModel.m > 2:
                if args.randomLength:
                    MSE_KF_linear_arr[j,2] = loss_fn(KF_out[j,2:3,test_lengthMask[j]], test_target[j,2:3,test_lengthMask[j]]).item()
                else:      
                    MSE_KF_linear_arr[j,2] = loss_fn(KF_out[j,2:3,:], test_target[j,2:3,:]).item()
                    
        elif allStates:
            # 计算所有状态的误差
            if args.randomLength:
                MSE_KF_linear_arr[j] = loss_fn(KF_out[j,:,test_lengthMask[j]], test_target[j,:,test_lengthMask[j]]).item()
            else:      
                MSE_KF_linear_arr[j] = loss_fn(KF_out[j,:,:], test_target[j,:,:]).item()
        else: # mask on state (原来的逻辑)
            if args.randomLength:
                MSE_KF_linear_arr[j] = loss_fn(KF_out[j,loc,test_lengthMask[j]], test_target[j,loc,test_lengthMask[j]]).item()
            else:           
                MSE_KF_linear_arr[j] = loss_fn(KF_out[j,loc,:], test_target[j,loc,:]).item()

    # Calculate averages and dB values
    if first_dim_only:
        MSE_KF_linear_avg = torch.mean(MSE_KF_linear_arr, dim=0)  # Average over trajectories for each dimension
        MSE_KF_dB_avg = 10 * torch.log10(MSE_KF_linear_avg)
        
        # Standard deviation for each dimension
        MSE_KF_linear_std = torch.std(MSE_KF_linear_arr, dim=0, unbiased=True)
        KF_std_dB = 10 * torch.log10(MSE_KF_linear_std + MSE_KF_linear_avg) - MSE_KF_dB_avg
        
        # Output results for each dimension
        print("KF MSE by Dimension:")
        dimensions = ['Position', 'Velocity', 'Acceleration']
        for dim_idx in range(min(3, SysModel.m)):
            dim_name = dimensions[dim_idx]
            print(f"  {dim_name}: {MSE_KF_dB_avg[dim_idx]:.6f} dB")
    else:
        MSE_KF_linear_avg = torch.mean(MSE_KF_linear_arr)
        MSE_KF_dB_avg = 10 * torch.log10(MSE_KF_linear_avg)
        
        # Standard deviation
        MSE_KF_linear_std = torch.std(MSE_KF_linear_arr, unbiased=True)
        KF_std_dB = 10 * torch.log10(MSE_KF_linear_std + MSE_KF_linear_avg) - MSE_KF_dB_avg
        
        print(f"KF MSE: {MSE_KF_dB_avg:.6f} dB")
    
    # Print Run Time
    print("Inference Time:", t)
    
    # Return results - include dimension-wise results if first_dim_only=True
    if first_dim_only:
        detailed_results = {
            'MSE_position_dB': MSE_KF_dB_avg[0].item(),
            'MSE_velocity_dB': MSE_KF_dB_avg[1].item() if SysModel.m > 1 else None,
            'MSE_acceleration_dB': MSE_KF_dB_avg[2].item() if SysModel.m > 2 else None,
            'MSE_linear_arr_by_dim': MSE_KF_linear_arr,  # [N_T, 3] array
            'MSE_linear_avg_by_dim': MSE_KF_linear_avg,  # [3] array
            'MSE_dB_avg_by_dim': MSE_KF_dB_avg  # [3] array
        }
        return [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out]
    else:
        return [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out]