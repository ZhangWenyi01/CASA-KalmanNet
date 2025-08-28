"""
This file contains the parameters for the simulations with linear kinematic model
* Constant Acceleration Model (CA)
    # full state P, V, A
    # only postion P
* Constant Velocity Model (CV)
"""

import torch

m = 3 # dim of state for CA model
m_cv = 2 # dim of state for CV model

delta_t_gen =  1e-2

#########################################################
### state evolution matrix F and observation matrix H ###
#########################################################
F_gen = torch.tensor([[1, delta_t_gen,0.5*delta_t_gen**2],
                  [0,       1,       delta_t_gen],
                  [0,       0,         1]]).float()


F_CV = torch.tensor([[1, delta_t_gen],
                     [0,           1]]).float()      

# Set rotation angle (in radians), e.g., 30 degrees
theta = torch.tensor(10.0) * torch.pi / 180  # Convert 30 degrees to radians

# Build 2D rotation matrix for the given angle (rotate all components)
# Here we assume rotation around z-axis, so z component remains unchanged, but x and y components rotate
rotation_matrix = torch.tensor([
    [torch.cos(theta), -torch.sin(theta), 0],
    [torch.sin(theta),  torch.cos(theta), 0],
    [0,                0,                1]
])

# Apply rotation matrix to F_gen
F_rotated = rotation_matrix @ F_gen @ rotation_matrix.T

# Full observation
H_identity = torch.eye(3)
# Observe only the position
H_onlyPos = torch.tensor([[1, 0, 0]]).float()

# Apply rotation matrix to H_onlyPos
H_onlyPos_rotated = H_onlyPos @ rotation_matrix.T

###############################################
### process noise Q and observation noise R ###
###############################################
# Noise Parameters
v = 0 # dB
gamma = -7.5
linear_factor = 10 ** (gamma / 10)
r2 = torch.tensor([1/linear_factor]).float()
q2 = r2*10 ** (v / 10)

# Only For CPD
q2 = torch.tensor([1.0]).float()
r2 = torch.tensor([1.0]).float()
Q_gen = q2 * torch.tensor([[1/20*delta_t_gen**5, 1/8*delta_t_gen**4,1/6*delta_t_gen**3],
                           [ 1/8*delta_t_gen**4, 1/3*delta_t_gen**3,1/2*delta_t_gen**2],
                           [ 1/6*delta_t_gen**3, 1/2*delta_t_gen**2,       delta_t_gen]]).float()


Q_CV = q2 * torch.tensor([[1/3*delta_t_gen**3, 1/2*delta_t_gen**2],
                          [1/2*delta_t_gen**2,        delta_t_gen]]).float()  

R_3 = r2 * torch.eye(3)
R_2 = r2 * torch.eye(2)

R_onlyPos = r2