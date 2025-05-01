import os
import torch
import mrcfile
import numpy as np
import matplotlib.pyplot as plt
import argparse
from glob import glob

def load_pt_file(pt_file):
    """Load a PyTorch tensor file and convert to numpy"""
    tensor = torch.load(pt_file)
    if isinstance(tensor, torch.Tensor):
        # Remove any singleton dimensions and convert to numpy
        return tensor.squeeze().cpu().numpy()
    return tensor

def load_mrc_file(mrc_file):
    """Load an MRC file"""
    with mrcfile.open(mrc_file) as mrc:
        return mrc.data

def visualize_3d_chunk(data, title, slice_axis=2):
    """Visualize middle slices of a 3D chunk along specified axis"""
    if slice_axis not in [0, 1, 2]:
        raise ValueError("slice_axis must be 0, 1, or 2")
    
    middle_slice = data.shape[slice_axis] // 2
    
    if slice_axis == 0:
        slice_data = data[middle_slice, :, :]
    elif slice_axis == 1:
        slice_data = data[:, middle_slice, :]
    else:  # slice_axis == 2
        slice_data = data[:, :, middle_slice]
    
    plt.imshow(slice_data, cmap='gray')
    plt.title(title)
    plt.colorbar()

def compare_chunks(pt_file, mrc_file, output_dir):
    """Compare a prediction chunk (.pt) with its corresponding input chunk (.mrc)"""
    # Load data
    pt_data = load_pt_file(pt_file)
    mrc_data = load_mrc_file(mrc_file)
    
    # Create figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot middle slices
    middle_slice = mrc_data.shape[2] // 2
    
    # Input MRC
    ax1.imshow(mrc_data[:, :, middle_slice], cmap='gray')
    ax1.set_title('Input MRC')
    plt.colorbar(ax1.imshow(mrc_data[:, :, middle_slice], cmap='gray'), ax=ax1)
    
    # Prediction PT
    ax2.imshow(pt_data[:, :, middle_slice], cmap='gray')
    ax2.set_title('Prediction PT')
    plt.colorbar(ax2.imshow(pt_data[:, :, middle_slice], cmap='gray'), ax=ax2)
    
    # Difference
    diff = pt_data - mrc_data
    im3 = ax3.imshow(diff[:, :, middle_slice], cmap='bwr')
    ax3.set_title('Difference')
    plt.colorbar(im3, ax=ax3)
    
    # Save the comparison
    chunk_number = os.path.basename(pt_file).split('_')[-1].split('.')[0]
    output_file = os.path.join(output_dir, f'chunk_comparison_{chunk_number}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return np.mean(np.abs(diff)), np.std(diff)

def main():
    parser = argparse.ArgumentParser(description='Visualize and compare chunk files')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input MRC chunks')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory containing prediction PT files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save comparisons')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all prediction files
    pred_files = sorted(glob(os.path.join(args.pred_dir, "predictions_output_*.pt")))
    
    # Process each chunk
    differences = []
    for pred_file in pred_files:
        # Get corresponding MRC file
        chunk_number = os.path.basename(pred_file).split('_')[-1].split('.')[0]
        mrc_file = os.path.join(args.input_dir, "sr_32_32", f"{chunk_number}.mrc")
        
        if not os.path.exists(mrc_file):
            print(f"Warning: Could not find corresponding MRC file for {pred_file}")
            continue
        
        print(f"Processing chunk {chunk_number}...")
        mean_diff, std_diff = compare_chunks(pred_file, mrc_file, args.output_dir)
        differences.append({
            'chunk': chunk_number,
            'mean_diff': mean_diff,
            'std_diff': std_diff
        })
        
    # Print summary statistics
    print("\nSummary Statistics:")
    print("------------------")
    mean_diffs = [d['mean_diff'] for d in differences]
    std_diffs = [d['std_diff'] for d in differences]
    print(f"Average absolute difference across all chunks: {np.mean(mean_diffs):.4f}")
    print(f"Standard deviation of differences across all chunks: {np.mean(std_diffs):.4f}")
    
    # Save statistics to file
    stats_file = os.path.join(args.output_dir, 'comparison_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write("Chunk Comparison Statistics\n")
        f.write("-------------------------\n")
        for d in differences:
            f.write(f"Chunk {d['chunk']}:\n")
            f.write(f"  Mean absolute difference: {d['mean_diff']:.4f}\n")
            f.write(f"  Standard deviation: {d['std_diff']:.4f}\n")
        f.write("\nOverall Statistics:\n")
        f.write(f"Average absolute difference: {np.mean(mean_diffs):.4f}\n")
        f.write(f"Average standard deviation: {np.mean(std_diffs):.4f}\n")

if __name__ == "__main__":
    main() 