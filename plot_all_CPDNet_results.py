"""
Plot CPDNet results from plot_data folder
This file reads CPDNet result data and creates plots using the same style as CPDNNTest
"""

import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import glob
import argparse
from pathlib import Path

def check_data_structure(plot_data):
    """
    Check the structure of loaded plot data
    
    Args:
        plot_data: Dictionary containing the plot data
        
    Returns:
        bool: True if data has required structure, False otherwise
    """
    print("Checking data structure...")
    print("Available keys:", list(plot_data.keys()))
    
    required_keys = ['predicted_cpd', 'actual_cpd', 'estimation_state', 
                    'true_state', 'estimation_y', 'true_y', 'changepoint']
    
    missing_keys = [key for key in required_keys if key not in plot_data]
    if missing_keys:
        print(f"Warning: Missing required keys: {missing_keys}")
        return False
    
    print("Data shapes:")
    for key in required_keys:
        if hasattr(plot_data[key], 'shape'):
            print(f"  {key}: {plot_data[key].shape}")
        else:
            print(f"  {key}: {type(plot_data[key])} = {plot_data[key]}")
    
    print(f"Changepoint: {plot_data['changepoint']}")
    return True

def plot_CPDNet_results_enhanced(plot_data, batch_idx=6, param_type='R', lambda_value=2, save_path=None):
    """
    Plot CPDNet results with enhanced visualization for black and white printing
    
    Args:
        plot_data: Dictionary containing the plot data
        batch_idx: Batch index to plot
        param_type: Type of parameter (R, Q, F, H)
        lambda_value: Lambda value for the parameter
        save_path: Path to save the figure (optional)
    """
    # Check if batch_idx is valid
    max_batch = plot_data['predicted_cpd'].shape[0] - 1
    if batch_idx > max_batch:
        print(f"Warning: batch_idx {batch_idx} exceeds maximum batch index {max_batch}")
        batch_idx = max_batch
        print(f"Using batch_idx = {batch_idx}")
    
    # Create figure with two subplots
    plt.figure(figsize=(12, 8))
    
    # Plot CPD results (top subplot)
    plt.subplot(2, 1, 1)
    plt.plot(plot_data['predicted_cpd'][batch_idx, 0, :-5], 
             label="Predicted CPD probability", 
             linestyle='--', linewidth=2, marker='s', markersize=4, markevery=5)
    plt.plot(plot_data['actual_cpd'][batch_idx, 0, :-5], 
             label="Actual CPD probability", 
             linestyle='-', linewidth=2, marker='o', markersize=4, markevery=5)
    plt.axvline(x=plot_data['changepoint'], color='red', linestyle='--', linewidth=3,
                label=f'Changepoint ({plot_data["changepoint"]})')
    plt.xlabel("Time Steps")
    plt.ylabel("CPD Probability")
    plt.title(f"CPDNet Prediction - {param_type} = {lambda_value}")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Plot state and observation results (bottom subplot)
    plt.subplot(2, 1, 2)
    plt.plot(plot_data['estimation_state'][batch_idx, 0, :-5], 
             label='estimation state', 
             color='red', linestyle='-', linewidth=2, marker='o', markersize=3, markevery=5)
    plt.plot(plot_data['true_state'][batch_idx, 0, :-5], 
             label='true state', 
             color='blue', linestyle='--', linewidth=2, marker='s', markersize=3, markevery=5)
    plt.plot(plot_data['estimation_y'][batch_idx, 0, :-5], 
             label='estimation y', 
             color='black', linestyle='-.', linewidth=2, marker='^', markersize=3, markevery=5)
    plt.plot(plot_data['true_y'][batch_idx, 0, :-5], 
             label='true y', 
             color='green', linestyle=':', linewidth=2, marker='d', markersize=3, markevery=5)
    plt.axvline(x=plot_data['changepoint'], color='red', linestyle='--', linewidth=3,
                label=f'Changepoint ({plot_data["changepoint"]})')
    plt.title(f'KalmanNet Output and True State - {param_type} = {lambda_value}')
    plt.xlabel('Time Step')
    plt.ylabel('Value (Dimension 1)')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()

def calculate_CPDNet_mse(plot_data):
    """
    Calculate Mean Squared Error (MSE) between predicted and actual CPD probabilities
    
    Args:
        plot_data: Dictionary containing the plot data
        
    Returns:
        float: MSE value in dB
    """
    predicted = plot_data['predicted_cpd'][:, 0, :-5]
    actual = plot_data['actual_cpd'][:, 0, :-5]
    mse = 10 * np.log10(np.mean((predicted - actual) ** 2))
    return mse

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
    if len(name_parts) > 3:
        if name_parts[3] == 'Gradually':
            lambda_value = 'Gradually'
        else:
            try:
                lambda_value = float(name_parts[3])
            except ValueError:
                lambda_value = name_parts[3]
    else:
        lambda_value = 'Unknown'
    
    return param_type, lambda_value

def list_available_files(plot_data_dir='plot_data'):
    """
    List all available CPDNet result files
    
    Args:
        plot_data_dir: Directory containing the plot data files
    """
    data_files = glob.glob(os.path.join(plot_data_dir, 'CPDNet_results_*.pt'))
    
    if not data_files:
        print(f"No CPDNet result files found in {plot_data_dir}")
        return []
    
    print(f"Available CPDNet result files:")
    print("="*60)
    for i, data_file in enumerate(sorted(data_files)):
        basename = os.path.basename(data_file)
        param_type, lambda_value = extract_params_from_filename(basename)
        print(f"{i+1:2d}. {basename} -> {param_type} = {lambda_value}")
    
    return data_files

def plot_specific_file(filename, batch_idx=6, save_figure=True):
    """
    Plot results from a specific CPDNet result file
    
    Args:
        filename: Path to the CPDNet result file
        batch_idx: Batch index to plot
        save_figure: Whether to save the figure
    """
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found")
        return
    
    print(f"Loading and plotting: {filename}")
    print("="*50)
    
    try:
        # Load data
        plot_data = torch.load(filename, map_location='cpu')
        
        # Check data structure
        if not check_data_structure(plot_data):
            print("Warning: Data structure may be incomplete")
        
        # Extract parameters from filename
        param_type, lambda_value = extract_params_from_filename(filename)
        
        # Calculate MSE
        mse = calculate_CPDNet_mse(plot_data)
        print(f"\nParameter: {param_type} = {lambda_value}")
        print(f"MSE (dB): {mse:.4f}")
        
        # Create save path if saving figure
        save_path = None
        if save_figure:
            output_dir = "CPDNet_plots"
            os.makedirs(output_dir, exist_ok=True)
            save_filename = f"CPDNet_{param_type}_{lambda_value}.pdf"
            save_path = os.path.join(output_dir, save_filename)
        
        # Plot results
        plot_CPDNet_results_enhanced(plot_data, batch_idx=batch_idx, 
                                   param_type=param_type, lambda_value=lambda_value,
                                   save_path=save_path)
        
        print(f"Plot completed successfully!")
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Plot CPDNet results')
    parser.add_argument('--file', '-f', type=str, 
                       help='Specific CPDNet result file to plot (e.g., CPDNet_results_H_0.95.pt)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List all available CPDNet result files')
    parser.add_argument('--batch', '-b', type=int, default=6,
                       help='Batch index to plot (default: 6)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save figures as PDF')
    
    args = parser.parse_args()
    
    print("CPDNet Results Visualization Tool")
    print("="*50)
    
    if args.list:
        # List available files
        list_available_files()
        return
    
    if args.file:
        # Plot specific file
        if not args.file.endswith('.pt'):
            args.file += '.pt'
        
        # Check if file exists in plot_data directory
        if not os.path.exists(args.file):
            plot_data_path = os.path.join('plot_data', args.file)
            if os.path.exists(plot_data_path):
                args.file = plot_data_path
            else:
                print(f"Error: File {args.file} not found in current directory or plot_data/")
                return
        
        plot_specific_file(args.file, batch_idx=args.batch, save_figure=not args.no_save)
    else:
        # Show help and list available files
        print("No file specified. Use --file to specify a file or --list to see available files.")
        print("\nAvailable files:")
        list_available_files()

if __name__ == "__main__":
    main()
