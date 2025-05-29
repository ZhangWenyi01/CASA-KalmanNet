import matplotlib.pyplot as plt
import torch
import os
import numpy as np

def plot_CPDNet_results(plot_data, batch_idx=20, param_type='R', lambda_value=2):
    """
    Plot results from loaded data
    
    Args:
        plot_data: Dictionary containing the plot data
        batch_idx: Batch index to plot
        param_type: Type of parameter (R, Q, F, H)
        lambda_value: Lambda value for the parameter
    """
    # Create figure with two subplots
    plt.figure(figsize=(10, 7))
    
    # Plot CPD results
    plt.subplot(2, 1, 1)
    plt.plot(plot_data['predicted_cpd'][batch_idx, 0, :-5], label="Predicted CPD probability", linestyle='--')
    plt.plot(plot_data['actual_cpd'][batch_idx, 0, :-5], label="Actual CPD probability", linestyle='-')
    plt.axvline(x=plot_data['changepoint'], color='green', linestyle='--', 
                label=f'Changepoint ({plot_data["changepoint"]})')
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    # plt.title(f"CPDNet Prediction ({param_type} = {lambda_value})")
    plt.title(f"CPDNet Prediction")
    plt.legend()
    plt.grid()

    # Plot state and observation results
    plt.subplot(2, 1, 2)
    plt.plot(plot_data['estimation_state'][batch_idx, 0, :-5], label='estimation state', color='red')
    plt.plot(plot_data['true_state'][batch_idx, 0, :-5], label='true state', color='blue')
    plt.plot(plot_data['estimation_y'][batch_idx, 0, :-5], label='estimation y', color='black')
    plt.plot(plot_data['true_y'][batch_idx, 0, :-5], label='true y')
    plt.axvline(x=plot_data['changepoint'], color='green', linestyle='--', 
                label=f'Changepoint ({plot_data["changepoint"]})')
    plt.title(f'KalmanNet output and true state')
    plt.xlabel('Time Step')
    plt.ylabel('Value (Dimension 1)')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the figure as PDF
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    save_path = os.path.join(desktop_path, f'CPDNet_results_{param_type}_{lambda_value}.pdf')
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    plt.show()

def calculate_CPDNet_mse(plot_data):
    """
    Calculate Mean Squared Error (MSE) between predicted and actual CPD probabilities
    
    Args:
        plot_data: Dictionary containing the plot data
        
    Returns:
        float: MSE value
    """
    predicted = plot_data['predicted_cpd'][:, 0, :-5]
    actual = plot_data['actual_cpd'][:, 0, :-5]
    mse = 10*np.log10(np.mean((predicted - actual) ** 2))
    return mse

def process_CPDNet_data(plot_data, batch_idx=10, param_type='R', lambda_value=2):
    """
    Process the data and plot results
    
    Args:
        plot_data: Dictionary containing the plot data
        batch_idx: Batch index to process and plot
        param_type: Type of parameter (R, Q, F, H)
        lambda_value: Lambda value for the parameter
    """
    # Calculate MSE
    mse = calculate_CPDNet_mse(plot_data)
    print(f"MSE between predicted and actual CPD ({param_type} = {lambda_value}): {mse:.6f}")
    
    # Plot and save results
    plot_CPDNet_results(plot_data, batch_idx=batch_idx, param_type=param_type, lambda_value=lambda_value)

def extract_params_from_filename(filename):
    """
    Extract parameter type and lambda value from filename
    
    Args:
        filename: Name of the data file
        
    Returns:
        tuple: (param_type, lambda_value)
    """
    # Remove path and extension
    basename = os.path.basename(filename)
    name_parts = basename.split('_')
    
    # Extract parameter type (R, Q, F, H)
    param_type = name_parts[2]
    
    # Extract lambda value
    # lambda_value = float(name_parts[3][:-3])
    lambda_value = 11111
    
    return param_type, lambda_value

if __name__ == "__main__":
    # Load data
    data_path = os.path.join('plot_data', 'CPDNet_results_Q_Gradually.pt')
    plot_data = torch.load(data_path)
    
    # Extract parameters from filename
    param_type, lambda_value = extract_params_from_filename(data_path)
    # Process and plot data
    process_CPDNet_data(plot_data, batch_idx=10, param_type=param_type, lambda_value=lambda_value) 