# CPDNet: Change Point Detection Network for State Estimation

CPDNet is a deep learning framework for detecting change points in dynamic systems and improving state estimation performance. The project combines traditional Kalman filtering techniques with neural networks to achieve robust change point detection and adaptive state estimation.

## Overview

This project implements a comprehensive framework for:
- **Change Point Detection (CPD)**: Neural network-based detection of parameter changes in dynamic systems
- **State Estimation**: Multiple filtering algorithms including Kalman Filter, Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF), and Particle Filter (PF)
- **Neural Network Integration**: KalmanNet and CPDNet for learning-based state estimation and change point detection
- **Multiple System Models**: Support for linear and nonlinear systems including Lorenz attractor dynamics

## Project Structure

```
CPDNet/
├── CPDNet/                    # CPDNet neural network implementation
│   ├── CPDNet_nn.py          # Main CPDNet architecture
│   └── best-model.pt         # Trained model weights
├── KNet/                      # KalmanNet implementation
│   ├── KalmanNet_nn.py       # KalmanNet architecture
│   └── best-model.pt         # Trained model weights
├── Filters/                   # Traditional filtering algorithms
│   ├── EKF.py                # Extended Kalman Filter
│   ├── UKF.py                # Unscented Kalman Filter
│   ├── PF.py                 # Particle Filter
│   ├── Linear_KF.py          # Linear Kalman Filter
│   └── *_test.py             # Test functions for each filter
├── Simulations/               # System models and parameters
│   ├── config.py             # Global configuration settings
│   ├── Linear_CPD/           # Linear system parameters
│   ├── Lorenz_Atractor/      # Lorenz attractor parameters
│   └── utils.py              # Utility functions
├── Pipelines/                 # Training and evaluation pipelines
│   ├── Pipeline_CPD.py       # CPDNet training pipeline
│   ├── Pipeline_EKF.py       # EKF training pipeline
│   └── Pipeline_lor.py       # Lorenz system pipeline
├── Plot/                      # Visualization tools
│   └── plot_CPDNet_results.py # Results plotting utilities
└── main_*.py                 # Main experiment scripts
```

## Key Features

### 1. Change Point Detection
- **CPDNet**: LSTM-based neural network for detecting parameter changes
- **Adaptive Thresholding**: Configurable detection thresholds
- **Multiple Parameter Support**: Detection of changes in Q, R, F, and H matrices

### 2. State Estimation Algorithms
- **Linear Kalman Filter**: For linear Gaussian systems
- **Extended Kalman Filter (EKF)**: For nonlinear systems with linearization
- **Unscented Kalman Filter (UKF)**: For nonlinear systems using sigma points
- **Particle Filter (PF)**: For highly nonlinear systems with non-Gaussian noise

### 3. Neural Network Integration
- **KalmanNet**: Deep learning approach to Kalman filtering
- **CPDNet**: Specialized network for change point detection
- **Hybrid Approaches**: Combining traditional filters with neural networks

### 4. System Models
- **Linear Systems**: Constant acceleration and velocity models
- **Nonlinear Systems**: Lorenz attractor with various observation models
- **Configurable Parameters**: Easy modification of system dynamics

## Installation

### Prerequisites
- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib
- SciPy
<!-- 
### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd CPDNet
```

2. Install dependencies:
```bash
pip install torch numpy matplotlib scipy
``` -->

## Usage

### Running Experiments

#### 1. Linear System with Change Point Detection
```bash
python main_linear_CPD.py
```
This script runs experiments on linear systems with change point detection using CPDNet.

#### 2. Lorenz Attractor System
```bash
python main_lor_DT.py
```
Runs experiments on the Lorenz attractor system with discrete-time observations.

### Configuration

#### System Parameters
Edit `Simulations/model_name/parameters.py` to modify:
- State dimension (m) and observation dimension (n)
- System matrices (F, H, Q, R)
- Initial conditions and noise parameters

#### Training Parameters
Edit `Simulations/config.py` to adjust:
- Dataset sizes (N_E, N_CV, N_T)
- Training parameters (n_steps, n_batch, lr)
- Network architecture settings

#### Change Point Settings
Modify change point parameters in main scripts:
```python
change_point_params = {
    'changed_param': 'Q',        # Parameter to change
    'Q': 2000*Q_gen,            # New Q matrix
    'R': 2*R_onlyPos,           # New R matrix
    'F': F_rotated,             # New F matrix
    'H': H_onlyPos_rotated      # New H matrix
}
```

### Visualization

#### Plot CPDNet Results
```bash
# List available result files
python plot_all_CPDNet_results.py --list

# Plot specific results
python plot_all_CPDNet_results.py --file CPDNet_results_R_1.2.pt

# Plot with custom batch index
python plot_all_CPDNet_results.py --file CPDNet_results_Q_100.pt --batch 5

# Plot without saving PDF
python plot_all_CPDNet_results.py --file CPDNet_results_F_0.95.pt --no-save
```

## Results and Output

### Data Storage
- **Model Weights**: Saved as `.pt` files in respective directories
- **Results**: Stored in `plot_data/` directory as `.pt` files
- **Plots**: Generated as high-resolution PDF files in `CPDNet_plots/`

### Performance Metrics
- **MSE (dB)**: Mean squared error in decibels
- **Change Point Detection**: Accuracy and timing of detections
- **State Estimation**: Comparison with ground truth states

## Customization

### Adding New System Models
1. Create new parameter file in `Simulations/`
2. Implement system dynamics functions (f, h)
3. Add corresponding main script

### Modifying CPDNet Architecture
1. Edit `CPDNet/CPDNet_nn.py`
2. Adjust LSTM layers, hidden sizes, and output dimensions
3. Update training parameters in config

### Adding New Filters
1. Implement filter class in `Filters/`
2. Add corresponding test function
3. Integrate with pipeline system

<!-- ## Citation

If you use this code in your research, please cite:

```bibtex
@article{cpdnet2024,
  title={CPDNet: Change Point Detection Network for Adaptive State Estimation},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
``` -->

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Contact

For questions and support, please open an issue on GitHub.