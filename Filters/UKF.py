"""
Unscented Kalman Filter (UKF) for non-linear systems
"""

import torch
from torch.distributions.multivariate_normal import MultivariateNormal


class UnscentedKalmanFilter:
    """Unscented Kalman Filter for non-linear systems"""
    
    def __init__(self, SystemModel, args):
        self.f = SystemModel.f
        self.h = SystemModel.h
        self.m = SystemModel.m  # state dimension
        self.n = SystemModel.n  # observation dimension
        self.Q = SystemModel.Q  # process noise covariance
        self.R = SystemModel.R  # observation noise covariance
        
        # Device
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        # UKF parameters
        self.alpha = getattr(args, 'ukf_alpha', 1e-3)  # spread parameter
        self.beta = getattr(args, 'ukf_beta', 2.0)     # distribution parameter (2 is optimal for Gaussian)
        self.kappa = getattr(args, 'ukf_kappa', 3.0 - self.m)   # secondary scaling parameter (common choice)
        
        # Derived parameters
        self.lambda_param = self.alpha**2 * (self.m + self.kappa) - self.m
        
        # Ensure lambda_param is reasonable (avoid negative values that are too large)
        if self.lambda_param < -self.m:
            self.lambda_param = -self.m + 1e-3
        self.n_sigma = 2 * self.m + 1  # number of sigma points
        
        # Weights
        self.Wm = torch.zeros(self.n_sigma).to(self.device)  # weights for means
        self.Wc = torch.zeros(self.n_sigma).to(self.device)  # weights for covariance
        
        # Ensure denominator is not zero
        denom = self.m + self.lambda_param
        if abs(denom) < 1e-8:
            denom = 1e-8 if denom >= 0 else -1e-8
        
        self.Wm[0] = self.lambda_param / denom
        self.Wc[0] = self.lambda_param / denom + (1 - self.alpha**2 + self.beta)
        
        for i in range(1, self.n_sigma):
            self.Wm[i] = 1 / (2 * denom)
            self.Wc[i] = 1 / (2 * denom)
            
    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):
        self.m1x_0_batch = m1x_0_batch
        self.m2x_0_batch = m2x_0_batch
        self.batch_size = m1x_0_batch.shape[0]
        
    def generate_sigma_points(self, x_mean, P):
        """Generate sigma points for UKF"""
        batch_size = x_mean.shape[0]
        sigma_points = torch.zeros(batch_size, self.n_sigma, self.m, 1).to(self.device)
        
        # Ensure P is positive definite by adding regularization
        P_reg = P + 1e-6 * torch.eye(self.m).expand_as(P).to(self.device)
        
        # Scale factor for sigma points
        scale_factor = self.m + self.lambda_param
        
        # Use SVD decomposition for better numerical stability
        try:
            # Try Cholesky first
            if scale_factor > 0:
                L = torch.linalg.cholesky(scale_factor * P_reg)
            else:
                # If scale_factor is negative or zero, use SVD
                U, S, V = torch.svd(P_reg)
                # Ensure all singular values are positive
                S = torch.clamp(S, min=1e-8)
                L = U * torch.sqrt(abs(scale_factor) * S).unsqueeze(-2)
        except:
            # Fallback to SVD decomposition
            try:
                U, S, V = torch.svd(P_reg)
                # Ensure all singular values are positive
                S = torch.clamp(S, min=1e-8)
                L = U * torch.sqrt(abs(scale_factor) * S).unsqueeze(-2)
            except:
                # Final fallback: use diagonal approximation
                P_diag = torch.diagonal(P_reg, dim1=-2, dim2=-1)
                P_diag = torch.clamp(P_diag, min=1e-8)  # Ensure positive
                L = torch.diag_embed(torch.sqrt(abs(scale_factor) * P_diag))
        
        # First sigma point (mean)
        sigma_points[:, 0, :, :] = x_mean
        
        # Remaining sigma points
        for i in range(self.m):
            sigma_points[:, i+1, :, :] = x_mean + L[:, :, i:i+1]
            sigma_points[:, i+1+self.m, :, :] = x_mean - L[:, :, i:i+1]
            
        return sigma_points
        
    def predict(self, x_prev, P_prev):
        """Prediction step"""
        batch_size = x_prev.shape[0]
        
        # Generate sigma points
        sigma_points = self.generate_sigma_points(x_prev, P_prev)
        
        # Propagate sigma points through state transition function
        sigma_points_pred = torch.zeros_like(sigma_points)
        for i in range(self.n_sigma):
            sigma_points_pred[:, i, :, :] = self.f(sigma_points[:, i, :, :])
        
        # Predicted mean
        x_pred = torch.zeros_like(x_prev)
        for i in range(self.n_sigma):
            x_pred += self.Wm[i] * sigma_points_pred[:, i, :, :]
        
        # Predicted covariance
        P_pred = self.Q.expand(batch_size, -1, -1).clone().to(self.device)
        for i in range(self.n_sigma):
            diff = sigma_points_pred[:, i, :, :] - x_pred
            P_pred += self.Wc[i] * torch.bmm(diff, diff.transpose(-1, -2))
        
        # Ensure P_pred is positive definite
        P_pred += 1e-8 * torch.eye(self.m).expand_as(P_pred).to(self.device)
            
        return x_pred, P_pred, sigma_points_pred
        
    def update(self, x_pred, P_pred, sigma_points_pred, y):
        """Update step"""
        batch_size = x_pred.shape[0]
        
        # Propagate sigma points through observation function
        sigma_points_obs = torch.zeros(batch_size, self.n_sigma, self.n, 1).to(self.device)
        for i in range(self.n_sigma):
            sigma_points_obs[:, i, :, :] = self.h(sigma_points_pred[:, i, :, :])
        
        # Predicted observation mean
        y_pred = torch.zeros(batch_size, self.n, 1).to(self.device)
        for i in range(self.n_sigma):
            y_pred += self.Wm[i] * sigma_points_obs[:, i, :, :]
        
        # Innovation covariance
        S = self.R.expand(batch_size, -1, -1).clone().to(self.device)
        for i in range(self.n_sigma):
            diff_y = sigma_points_obs[:, i, :, :] - y_pred
            S += self.Wc[i] * torch.bmm(diff_y, diff_y.transpose(-1, -2))
        
        # Cross covariance
        Pxy = torch.zeros(batch_size, self.m, self.n).to(self.device)
        for i in range(self.n_sigma):
            diff_x = sigma_points_pred[:, i, :, :] - x_pred  # [batch, m, 1]
            diff_y = sigma_points_obs[:, i, :, :] - y_pred   # [batch, n, 1]
            cross_cov = torch.bmm(diff_x, diff_y.transpose(-1, -2))  # [batch, m, n]
            Pxy += self.Wc[i] * cross_cov
        
        # Kalman gain
        try:
            K = torch.bmm(Pxy, torch.inverse(S))
        except:
            # Add regularization if inversion fails
            S_reg = S + 1e-6 * torch.eye(self.n).expand_as(S).to(self.device)
            K = torch.bmm(Pxy, torch.inverse(S_reg))
        
        # Innovation
        innovation = y - y_pred
        
        # Updated state and covariance
        x_updated = x_pred + torch.bmm(K, innovation)
        P_updated = P_pred - torch.bmm(torch.bmm(K, S), K.transpose(-1, -2))
        
        # Ensure P_updated is positive definite
        P_updated += 1e-8 * torch.eye(self.m).expand_as(P_updated).to(self.device)
        
        return x_updated, P_updated, K
        
    def GenerateBatch(self, y_batch):
        """Process a batch of observation sequences"""
        batch_size, n_obs, T = y_batch.shape
        y_batch = y_batch.to(self.device)
        
        # Initialize output tensors
        self.x = torch.zeros(batch_size, self.m, T).to(self.device)
        self.KG_array = torch.zeros(batch_size, self.m, self.n, T).to(self.device)
        
        # Initialize state and covariance
        x_current = self.m1x_0_batch.to(self.device)
        P_current = self.m2x_0_batch.to(self.device)
        
        for t in range(T):
            # Prediction step
            if t > 0:
                x_pred, P_pred, sigma_points_pred = self.predict(x_current, P_current)
            else:
                x_pred = x_current
                P_pred = P_current
                sigma_points_pred = self.generate_sigma_points(x_pred, P_pred)
            
            # Update step
            y_current = y_batch[:, :, t:t+1]
            x_current, P_current, K = self.update(x_pred, P_pred, sigma_points_pred, y_current)
            
            # Store results
            self.x[:, :, t] = x_current.squeeze(-1)
            self.KG_array[:, :, :, t] = K
