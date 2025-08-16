"""
Particle Filter (PF) for non-linear systems
"""

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical


class ParticleFilter:
    """Particle Filter for non-linear systems"""
    
    def __init__(self, SystemModel, args):
        self.f = SystemModel.f
        self.h = SystemModel.h
        self.m = SystemModel.m  # state dimension
        self.n = SystemModel.n  # observation dimension
        
        # Device
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        self.Q = SystemModel.Q.to(self.device)  # process noise covariance
        self.R = SystemModel.R.to(self.device)  # observation noise covariance
        
        # PF parameters
        self.N_particles = getattr(args, 'pf_particles', 1000)  # number of particles
        self.resample_threshold = getattr(args, 'pf_resample_threshold', 0.5)  # effective sample size threshold
        
    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):
        self.m1x_0_batch = m1x_0_batch
        self.m2x_0_batch = m2x_0_batch
        self.batch_size = m1x_0_batch.shape[0]
        
    def initialize_particles(self):
        """Initialize particles from prior distribution"""
        batch_size = self.batch_size
        
        # Sample particles from initial distribution
        particles = torch.zeros(batch_size, self.N_particles, self.m, 1).to(self.device)
        weights = torch.ones(batch_size, self.N_particles).to(self.device) / self.N_particles
        
        for b in range(batch_size):
            if self.m == 1:
                particles[b, :, :, 0] = torch.normal(
                    self.m1x_0_batch[b, :, 0].to(self.device).expand(self.N_particles, -1),
                    torch.sqrt(torch.diag(self.m2x_0_batch[b].to(self.device))).expand(self.N_particles, -1)
                )
            else:
                try:
                    dist = MultivariateNormal(
                        self.m1x_0_batch[b, :, 0].cpu(), 
                        self.m2x_0_batch[b].cpu()
                    )
                    particles[b, :, :, 0] = dist.sample((self.N_particles,)).to(self.device)
                except:
                    # Fallback to diagonal sampling if covariance is singular
                    for i in range(self.m):
                        particles[b, :, i, 0] = torch.normal(
                            self.m1x_0_batch[b, i, 0].to(self.device).expand(self.N_particles),
                            torch.sqrt(self.m2x_0_batch[b, i, i].to(self.device)).expand(self.N_particles)
                        )
        
        return particles, weights
    
    def predict_particles(self, particles):
        """Predict particles using state transition model"""
        batch_size, N_particles, m, _ = particles.shape
        predicted_particles = torch.zeros_like(particles)
        
        # Add process noise
        if self.m == 1:
            process_noise = torch.normal(
                torch.zeros(batch_size, N_particles, self.m, 1).to(self.device),
                torch.sqrt(self.Q).expand(batch_size, N_particles, self.m, 1)
            )
        else:
            process_noise = torch.zeros(batch_size, N_particles, self.m, 1).to(self.device)
            try:
                dist = MultivariateNormal(torch.zeros(self.m), self.Q.cpu())
                for b in range(batch_size):
                    process_noise[b, :, :, 0] = dist.sample((N_particles,)).to(self.device)
            except:
                # Fallback to diagonal noise
                for b in range(batch_size):
                    for i in range(self.m):
                        process_noise[b, :, i, 0] = torch.normal(
                            torch.zeros(N_particles).to(self.device),
                            torch.sqrt(self.Q[i, i]).expand(N_particles)
                        )
        
        # Propagate particles through state transition
        for b in range(batch_size):
            for p in range(N_particles):
                # Add batch dimension for f function
                particle_state = particles[b, p, :, :].unsqueeze(0)  # [1, m, 1]
                f_x = self.f(particle_state)  # [1, m, 1]
                f_x = f_x.squeeze(0)  # [m, 1]
                predicted_particles[b, p, :, :] = f_x + process_noise[b, p, :, :]
        
        return predicted_particles
    
    def compute_weights(self, particles, observation):
        """Compute particle weights based on observation likelihood - Vectorized version"""
        batch_size, N_particles, m, _ = particles.shape
        weights = torch.zeros(batch_size, N_particles).to(self.device)
        
        # Vectorized computation for all particles at once
        for b in range(batch_size):
            # Reshape particles for batch processing: [N_particles, m, 1] -> [N_particles, m, 1]
            particles_b = particles[b]  # [N_particles, m, 1]
            
            # Propagate all particles through observation function at once
            h_x_batch = self.h(particles_b)  # [N_particles, n, 1]
            
            # Compute likelihood for all particles
            if self.n == 1:
                diff = observation[b, :, :].expand(N_particles, -1, -1) - h_x_batch
                likelihood = torch.exp(-0.5 * (diff**2 / self.R).sum(dim=(1, 2)))
            else:
                diff = observation[b, :, :].expand(N_particles, -1, -1) - h_x_batch
                try:
                    R_inv = torch.inverse(self.R)
                    # Vectorized likelihood computation
                    likelihood = torch.exp(-0.5 * torch.sum(diff * torch.mm(diff.squeeze(-1), R_inv), dim=(1, 2)))
                except:
                    # Fallback to diagonal R
                    R_diag = torch.diag(self.R).unsqueeze(-1)
                    likelihood = torch.exp(-0.5 * ((diff**2) / R_diag).sum(dim=(1, 2)))
            
            weights[b] = likelihood
        
        # Normalize weights
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-10)
        
        return weights
    
    def effective_sample_size(self, weights):
        """Compute effective sample size"""
        return 1.0 / (weights**2).sum(dim=1)
    
    def resample(self, particles, weights):
        """Systematic resampling"""
        batch_size, N_particles, m, _ = particles.shape
        resampled_particles = torch.zeros_like(particles)
        
        for b in range(batch_size):
            # Systematic resampling
            indices = torch.zeros(N_particles, dtype=torch.long, device=self.device)
            cumsum = torch.cumsum(weights[b].to(self.device), dim=0)
            
            u = torch.rand(1, device=self.device) / N_particles
            i, j = 0, 0
            
            while j < N_particles:
                while i < N_particles - 1 and cumsum[i] < u:
                    i += 1
                indices[j] = i
                u += 1.0 / N_particles
                j += 1
            
            # Resample particles
            resampled_particles[b] = particles[b, indices]
        
        # Reset weights to uniform
        uniform_weights = torch.ones(batch_size, N_particles, device=self.device) / N_particles
        
        return resampled_particles, uniform_weights
    
    def estimate_state(self, particles, weights):
        """Estimate state from weighted particles"""
        # Weighted mean
        weighted_particles = particles * weights.unsqueeze(-1).unsqueeze(-1)
        state_estimate = weighted_particles.sum(dim=1)
        
        return state_estimate
    
    def GenerateBatch(self, y_batch):
        """Process a batch of observation sequences"""
        batch_size, n_obs, T = y_batch.shape
        y_batch = y_batch.to(self.device)
        
        # Initialize output tensors
        self.x = torch.zeros(batch_size, self.m, T).to(self.device)
        self.KG_array = torch.zeros(batch_size, self.m, self.n, T).to(self.device)  # Not applicable for PF, but kept for compatibility
        
        # Initialize particles
        particles, weights = self.initialize_particles()
        
        for t in range(T):
            # Prediction step
            if t > 0:
                particles = self.predict_particles(particles)
            
            # Update step
            observation = y_batch[:, :, t:t+1]
            weights = self.compute_weights(particles, observation)
            
            # State estimation
            state_estimate = self.estimate_state(particles, weights)
            self.x[:, :, t] = state_estimate.squeeze(-1)
            
            # Resampling
            ess = self.effective_sample_size(weights)
            resample_mask = ess < self.resample_threshold * self.N_particles
            
            if resample_mask.any():
                for b in range(batch_size):
                    if resample_mask[b]:
                        particles_b = particles[b:b+1]
                        weights_b = weights[b:b+1]
                        resampled_particles, resampled_weights = self.resample(particles_b, weights_b)
                        particles[b] = resampled_particles[0]
                        weights[b] = resampled_weights[0]
