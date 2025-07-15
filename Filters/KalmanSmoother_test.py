import torch
import torch.nn as nn
import time
import numpy as np


class KalmanSmoother:
    """
    Kalman Smoother (RTS Smoother)
    Uses forward-backward algorithm for optimal state estimation
    """
    
    def __init__(self, SystemModel, args):
        # Device
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        self.F = SystemModel.F
        self.m = SystemModel.m
        self.Q = SystemModel.Q.to(self.device)

        self.H = SystemModel.H
        self.n = SystemModel.n
        self.R = SystemModel.R.to(self.device)

        self.T = SystemModel.T
        self.T_test = SystemModel.T_test

    def Forward(self, y):
        """Forward pass: Standard Kalman Filter"""
        y = y.to(self.device)
        self.batch_size = y.shape[0]
        T = y.shape[2]

        # Batched F and H
        self.batched_F = self.F.view(1,self.m,self.m).expand(self.batch_size,-1,-1).to(self.device)
        self.batched_F_T = torch.transpose(self.batched_F, 1, 2).to(self.device)
        self.batched_H = self.H.view(1,self.n,self.m).expand(self.batch_size,-1,-1).to(self.device)
        self.batched_H_T = torch.transpose(self.batched_H, 1, 2).to(self.device)

        # Allocate arrays for forward pass
        self.x_pred = torch.zeros(self.batch_size, self.m, T+1).to(self.device)
        self.P_pred = torch.zeros(self.batch_size, self.m, self.m, T+1).to(self.device)
        self.x_filt = torch.zeros(self.batch_size, self.m, T+1).to(self.device)
        self.P_filt = torch.zeros(self.batch_size, self.m, self.m, T+1).to(self.device)
        
        # Set initial conditions
        self.x_filt[:, :, 0] = self.m1x_0_batch.squeeze(-1)
        self.P_filt[:, :, :, 0] = self.m2x_0_batch

        # Forward pass
        for t in range(T):
            # Predict
            self.x_pred[:, :, t+1] = torch.bmm(self.batched_F, self.x_filt[:, :, t].unsqueeze(-1)).squeeze(-1)
            self.P_pred[:, :, :, t+1] = torch.bmm(self.batched_F, self.P_filt[:, :, :, t])
            self.P_pred[:, :, :, t+1] = torch.bmm(self.P_pred[:, :, :, t+1], self.batched_F_T) + self.Q

            # Update
            yt = y[:, :, t].unsqueeze(-1)
            innovation = yt - torch.bmm(self.batched_H, self.x_pred[:, :, t+1].unsqueeze(-1))
            
            S = torch.bmm(self.batched_H, self.P_pred[:, :, :, t+1])
            S = torch.bmm(S, self.batched_H_T) + self.R
            
            K = torch.bmm(self.P_pred[:, :, :, t+1], self.batched_H_T)
            K = torch.bmm(K, torch.inverse(S))
            
            self.x_filt[:, :, t+1] = self.x_pred[:, :, t+1] + torch.bmm(K, innovation).squeeze(-1)
            
            I_KH = torch.eye(self.m, device=self.device).unsqueeze(0).expand(self.batch_size, -1, -1) - torch.bmm(K, self.batched_H)
            self.P_filt[:, :, :, t+1] = torch.bmm(I_KH, self.P_pred[:, :, :, t+1])

    def Backward(self):
        """Backward pass: RTS Smoother"""
        T = self.x_filt.shape[2] - 1
        
        # Allocate arrays for smoothed estimates
        self.x_smooth = torch.zeros_like(self.x_filt).to(self.device)
        self.P_smooth = torch.zeros_like(self.P_filt).to(self.device)
        
        # Initialize with final filtered estimates
        self.x_smooth[:, :, T] = self.x_filt[:, :, T]
        self.P_smooth[:, :, :, T] = self.P_filt[:, :, :, T]
        
        # Backward pass
        for t in reversed(range(T)):
            # Compute smoother gain
            A = torch.bmm(self.P_filt[:, :, :, t], self.batched_F_T)
            
            # Handle numerical issues
            P_pred_inv = torch.inverse(self.P_pred[:, :, :, t+1] + torch.eye(self.m, device=self.device).unsqueeze(0).expand(self.batch_size, -1, -1) * 1e-6)
            A = torch.bmm(A, P_pred_inv)
            
            # Smooth
            self.x_smooth[:, :, t] = self.x_filt[:, :, t] + torch.bmm(A, (self.x_smooth[:, :, t+1] - self.x_pred[:, :, t+1]).unsqueeze(-1)).squeeze(-1)
            self.P_smooth[:, :, :, t] = self.P_filt[:, :, :, t] + torch.bmm(A, torch.bmm(self.P_smooth[:, :, :, t+1] - self.P_pred[:, :, :, t+1], A.transpose(-1, -2)))

    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):
        self.m1x_0_batch = m1x_0_batch
        self.m2x_0_batch = m2x_0_batch


    def GenerateBatch(self, y):
        """Generate batch with forward-backward smoothing"""
        # Forward pass
        self.Forward(y)
        
        # Backward pass  
        self.Backward()
        
        # Return smoothed states (excluding initial state)
        self.x = self.x_smooth[:, :, 1:]
        
        # Set posterior attributes for compatibility with KalmanFilter
        T = self.x_smooth.shape[2] - 1
        self.m1x_posterior = self.x_smooth[:, :, T:T+1].clone()  # Last smoothed state [batch, m, 1]
        self.m2x_posterior = self.P_smooth[:, :, :, T].clone()   # Last smoothed covariance [batch, m, m]


def KSTest(args, SysModel, test_input, test_target, allStates=True,
           randomInit=False, test_init=None, test_lengthMask=None,
           changepoint=None, changeparameters=None, first_dim_only=False):
    """
    Kalman Smoother Test Function
    
    Parameters match KFTest for easy comparison:
    - args: arguments object
    - SysModel: system model
    - test_input: test observations [N_T, n, T]
    - test_target: test true states [N_T, m, T]
    - allStates: if True, compute MSE on all states; if False, only position
    - randomInit: if True, use random initialization
    - test_init: initial conditions [N_T, m, 1]
    - test_lengthMask: mask for variable length sequences
    - changepoint: change point index
    - changeparameters: dictionary of changed parameters
    - first_dim_only: if True, compute MSE for position, velocity, acceleration separately
    
    Returns:
    [MSE_KS_linear_arr, MSE_KS_linear_avg, MSE_KS_dB_avg, KS_out, detailed_results (optional)]
    """
    
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
        MSE_KS_linear_arr = torch.zeros(args.N_T, 3)  # 3 dimensions: pos, vel, acc
        MSE_KS_linear_avg = torch.zeros(3)
        MSE_KS_dB_avg = torch.zeros(3)
    else:
        MSE_KS_linear_arr = torch.zeros(args.N_T)
        
    # allocate memory for KS output
    KS_out = torch.zeros(args.N_T, SysModel.m, args.T_test)
    
    # Determine dimensions for error calculation
    if not first_dim_only and not allStates:
        # Original logic: calculate position error
        loc = torch.tensor([True,False,False]) # for position only
        if SysModel.m == 2: 
            loc = torch.tensor([True,False]) # for position only

    start = time.time()

    KS = KalmanSmoother(SysModel, args)
    
    # Init and Forward-Backward Computation 
    if(randomInit):
        KS.Init_batched_sequence(test_init, SysModel.m2x_0.view(1,SysModel.m,SysModel.m).expand(args.N_T,-1,-1))        
    else:
        KS.Init_batched_sequence(SysModel.m1x_0.view(1,SysModel.m,1).expand(args.N_T,-1,-1), SysModel.m2x_0.view(1,SysModel.m,SysModel.m).expand(args.N_T,-1,-1))
    
    # Handle parameter changes like in KFTest
    if change_happened:
        # Process data before change point
        KS.GenerateBatch(test_input[:, :, :changepoint])
        X_before = KS.x
        
        # Save state at change point for continuity (same as KFTest)
        final_state = KS.m1x_posterior.clone()  # Last smoothed state
        final_covariance = KS.m2x_posterior.clone()  # Last smoothed covariance
        
        # Update parameters at change point
        if changed_param == 'Q':
            KS.Q = Q_after.to(KS.device)
        elif changed_param == 'R':
            KS.R = R_after.to(KS.device)
        elif changed_param == 'F':
            KS.F = F_after
        elif changed_param == 'H':
            KS.H = H_after
        
        # Re-initialize KS with state at change point
        KS.Init_batched_sequence(final_state, final_covariance)
            
        # Process data after change point
        KS.GenerateBatch(test_input[:, :, changepoint:])
        x_after = torch.cat([X_before, KS.x], dim=2)
        KS_out = x_after
    else:
        KS.GenerateBatch(test_input)
        KS_out = KS.x
    
    end = time.time()
    t = end - start
    
    # MSE loss calculation
    for j in range(args.N_T):
        if first_dim_only:
            # Calculate MSE for each dimension separately: position, velocity, acceleration
            # Position (dimension 0)
            if args.randomLength:
                MSE_KS_linear_arr[j,0] = loss_fn(KS_out[j,0:1,test_lengthMask[j]], test_target[j,0:1,test_lengthMask[j]]).item()
            else:      
                MSE_KS_linear_arr[j,0] = loss_fn(KS_out[j,0:1,:], test_target[j,0:1,:]).item()
            
            # Velocity (dimension 1) if available
            if SysModel.m > 1:
                if args.randomLength:
                    MSE_KS_linear_arr[j,1] = loss_fn(KS_out[j,1:2,test_lengthMask[j]], test_target[j,1:2,test_lengthMask[j]]).item()
                else:      
                    MSE_KS_linear_arr[j,1] = loss_fn(KS_out[j,1:2,:], test_target[j,1:2,:]).item()
            
            # Acceleration (dimension 2) if available
            if SysModel.m > 2:
                if args.randomLength:
                    MSE_KS_linear_arr[j,2] = loss_fn(KS_out[j,2:3,test_lengthMask[j]], test_target[j,2:3,test_lengthMask[j]]).item()
                else:      
                    MSE_KS_linear_arr[j,2] = loss_fn(KS_out[j,2:3,:], test_target[j,2:3,:]).item()
                    
        elif allStates:
            # Calculate error for all states
            if args.randomLength:
                MSE_KS_linear_arr[j] = loss_fn(KS_out[j,:,test_lengthMask[j]], test_target[j,:,test_lengthMask[j]]).item()
            else:      
                MSE_KS_linear_arr[j] = loss_fn(KS_out[j,:,:], test_target[j,:,:]).item()
        else: # mask on state (original logic)
            if args.randomLength:
                MSE_KS_linear_arr[j] = loss_fn(KS_out[j,loc,test_lengthMask[j]], test_target[j,loc,test_lengthMask[j]]).item()
            else:           
                MSE_KS_linear_arr[j] = loss_fn(KS_out[j,loc,:], test_target[j,loc,:]).item()

    # Calculate averages and dB values
    if first_dim_only:
        MSE_KS_linear_avg = torch.mean(MSE_KS_linear_arr, dim=0)  # Average over trajectories for each dimension
        MSE_KS_dB_avg = 10 * torch.log10(MSE_KS_linear_avg)
        
        # Standard deviation for each dimension
        MSE_KS_linear_std = torch.std(MSE_KS_linear_arr, dim=0, unbiased=True)
        KS_std_dB = 10 * torch.log10(MSE_KS_linear_std + MSE_KS_linear_avg) - MSE_KS_dB_avg
        
        # Output results for each dimension
        print("KS MSE by Dimension:")
        dimensions = ['Position', 'Velocity', 'Acceleration']
        for dim_idx in range(min(3, SysModel.m)):
            dim_name = dimensions[dim_idx]
            print(f"  {dim_name}: {MSE_KS_dB_avg[dim_idx]:.6f} dB")
    else:
        MSE_KS_linear_avg = torch.mean(MSE_KS_linear_arr)
        MSE_KS_dB_avg = 10 * torch.log10(MSE_KS_linear_avg)
        
        # Standard deviation
        MSE_KS_linear_std = torch.std(MSE_KS_linear_arr, unbiased=True)
        KS_std_dB = 10 * torch.log10(MSE_KS_linear_std + MSE_KS_linear_avg) - MSE_KS_dB_avg
        
        print(f"KS MSE: {MSE_KS_dB_avg:.6f} dB")
    
    # Print Run Time
    print("KS Inference Time:", t)
    
    # Return results - include dimension-wise results if first_dim_only=True
    if first_dim_only:
        detailed_results = {
            'MSE_position_dB': MSE_KS_dB_avg[0].item(),
            'MSE_velocity_dB': MSE_KS_dB_avg[1].item() if SysModel.m > 1 else None,
            'MSE_acceleration_dB': MSE_KS_dB_avg[2].item() if SysModel.m > 2 else None,
            'MSE_linear_arr_by_dim': MSE_KS_linear_arr,  # [N_T, 3] array
            'MSE_linear_avg_by_dim': MSE_KS_linear_avg,  # [3] array
            'MSE_dB_avg_by_dim': MSE_KS_dB_avg  # [3] array
        }
        return [MSE_KS_linear_arr, MSE_KS_linear_avg, MSE_KS_dB_avg, KS_out, detailed_results]
    else:
        return [MSE_KS_linear_arr, MSE_KS_linear_avg, MSE_KS_dB_avg, KS_out] 