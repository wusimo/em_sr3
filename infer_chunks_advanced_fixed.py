import os
import argparse
import torch
import numpy as np
import mrcfile
import glob
import shutil
import tempfile
from copy import deepcopy
import sys
import logging
import core.logger as Logger
from model import create_model
import data as Data
import core.metrics as Metrics
import torch.nn as nn
import empatches

# Initialize EMPatches
emp = empatches.EMPatches()

def expansion3d(x, dim=128):
    """Add a channel dimension to the input tensor"""
    x = x.unsqueeze(1)  # now shape: [batch, 1, z, y, x]
    return x

def prepare_input(input_map_file_path, temp_input_dir):
    """Split the input map into chunks using EMPatches and save them to the temp directory"""
    logger = logging.getLogger('base')
    
    logger.info(f"Opening input map: {input_map_file_path}")
    with mrcfile.open(input_map_file_path) as mrc:
        mapdata = mrc.data.astype(np.float32).copy()
    
    logger.info(f"Input map shape: {mapdata.shape}")
    
    # Normalize the input
    percentile_99p999 = np.percentile(mapdata[np.nonzero(mapdata)], 99.999)
    logger.info(f"99.999th percentile: {percentile_99p999}")
    mapdata /= percentile_99p999
    mapdata[mapdata < 0] = 0
    mapdata[mapdata > 1] = 1

    # Extract patches using EMPatches
    block_size = 32
    stride_size = 24
    
    logger.info(f"Extracting patches with block_size={block_size}, stride_size={stride_size}")
    
    blocks, indices = emp.extract_patches(mapdata, patchsize=block_size, stride=stride_size, vox=True)
    
    logger.info(f"Extracted {len(blocks)} patches")
    
    # Create the directory structure expected by the model
    sr_dir = os.path.join(temp_input_dir, "sr_32_32")
    hr_dir = os.path.join(temp_input_dir, "hr_32")
    os.makedirs(sr_dir, exist_ok=True)
    os.makedirs(hr_dir, exist_ok=True)
    
    # Save patches, indices and create a mapping file
    patch_mapping = {}  # Store mapping between patch index and original position
    
    for i, block in enumerate(blocks):
        # Save to sr_32_32 directory
        sr_filepath = os.path.join(sr_dir, f"{i}.mrc")
        with mrcfile.new(sr_filepath, overwrite=True) as mrc:
            mrc.set_data(deepcopy(block))
        
        # Save the same patch to hr_32 directory
        hr_filepath = os.path.join(hr_dir, f"{i}.mrc")
        with mrcfile.new(hr_filepath, overwrite=True) as mrc:
            mrc.set_data(deepcopy(block))
            
        # Store mapping
        patch_mapping[i] = {
            'original_index': i,
            'shape': block.shape,
            'position': indices[i]
        }
    
    # Save indices and mapping
    indices_filepath = os.path.join(temp_input_dir, "block_indices.pt")
    mapping_filepath = os.path.join(temp_input_dir, "patch_mapping.pt")
    torch.save(indices, indices_filepath, pickle_protocol=5)
    torch.save(patch_mapping, mapping_filepath, pickle_protocol=5)
    
    logger.info(f"Saved {len(patch_mapping)} patch mappings")
    return len(blocks)

def process_chunks(model, input_dir, output_dir, batch_size=4):
    """Process each chunk using the SR3 model"""
    logger = logging.getLogger('base')
    
    # Get all input files from the sr_32_32 directory
    sr_dir = os.path.join(input_dir, "sr_32_32")
    input_files = sorted(glob.glob(os.path.join(sr_dir, "*.mrc")), 
                         key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    logger.info(f"Found {len(input_files)} input files in {sr_dir}")
    
    if len(input_files) == 0:
        logger.error(f"{sr_dir} has no valid image file")
        return
    
    # Create dataset
    dataset_opt = {
        "name": "test",
        "mode": "HR",
        "dataroot": input_dir,
        "datatype": "mrc",
        "l_resolution": 32,
        "r_resolution": 32,
        "batch_size": batch_size,
        "num_workers": 4,
        "use_shuffle": False,
        "data_len": -1
    }
    
    dataset = Data.create_dataset(dataset_opt, 'val')
    dataloader = Data.create_dataloader(dataset, dataset_opt, 'val')
    
    # Process each batch
    for i, data in enumerate(dataloader):
        # Prepare data
        data['HR'] = expansion3d(data['HR'])
        data['SR'] = expansion3d(data['SR'])
        
        # Process with model
        model.feed_data(data)
        model.test(continous=False)
        visuals = model.get_current_visuals()
        
        # Save results with both predictions and original filenames
        for j in range(len(visuals['SR'])):
            idx = i * batch_size + j
            if idx < len(input_files):
                # Save prediction
                output_file = os.path.join(output_dir, f"predictions_output_{idx}.pt")
                torch.save(visuals['SR'][j].cpu(), output_file)
                
                # Save original filename for reference
                filename_file = os.path.join(output_dir, f"predictions_filename_{idx}.pt")
                torch.save(input_files[idx], filename_file)
                
                logger.info(f"Processed and saved chunk {idx}/{len(input_files)}")

def merge_output(temp_output_dir, output_map_file_path, temp_input_dir, input_map_file_path):
    """Merge processed chunks back into a single MRC file using EMPatches with improved index handling"""
    logger = logging.getLogger('base')
    
    # Load block indices and patch mapping
    block_indices = torch.load(os.path.join(temp_input_dir, "block_indices.pt"))
    patch_mapping = torch.load(os.path.join(temp_input_dir, "patch_mapping.pt"))
    logger.info(f"Loaded {len(block_indices)} original block indices and {len(patch_mapping)} patch mappings")
    
    # Get prediction and filename files
    prediction_files = sorted(glob.glob(os.path.join(temp_output_dir, "predictions_output_*.pt")), 
                            key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    filename_files = sorted(glob.glob(os.path.join(temp_output_dir, "predictions_filename_*.pt")),
                          key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    
    logger.info(f"Found {len(prediction_files)} prediction files and {len(filename_files)} filename files")
    
    if len(prediction_files) != len(filename_files):
        logger.error("Mismatch between prediction files and filename files")
        return False
    
    # Initialize arrays for predictions and indices
    preds = [None] * len(block_indices)
    indices = [None] * len(block_indices)
    
    # Load predictions and match them with indices
    for pred_file, filename_file in zip(prediction_files, filename_files):
        pred_idx = int(os.path.basename(pred_file).split('_')[-1].split('.')[0])
        original_filename = torch.load(filename_file)
        original_idx = int(os.path.basename(original_filename).split('.')[0])
        
        # Verify mapping
        if original_idx not in patch_mapping:
            logger.error(f"Missing mapping for index {original_idx}")
            continue
            
        pred = torch.load(pred_file).squeeze().numpy()
        logger.info(f"Processing prediction {pred_idx} from original position {original_idx}")
        
        # Store prediction and corresponding index
        preds[original_idx] = pred
        indices[original_idx] = block_indices[original_idx]
    
    # Verify all positions are filled
    if any(x is None for x in preds) or any(x is None for x in indices):
        logger.error("Some positions in the final array are empty")
        missing_preds = [i for i, x in enumerate(preds) if x is None]
        missing_indices = [i for i, x in enumerate(indices) if x is None]
        logger.error(f"Missing predictions at positions: {missing_preds}")
        logger.error(f"Missing indices at positions: {missing_indices}")
        return False
    
    # Use EMPatches to merge the predictions
    logger.info("Merging patches...")
    merged_arr = emp.merge_patches(preds, indices, mode="avg")
    logger.info(f"Merged array shape: {merged_arr.shape}")
    
    # Load original metadata
    with mrcfile.open(input_map_file_path) as mrc:
        voxel_size = np.array(mrc.voxel_size.data)
        axis_order = [mrc.header.mapc, mrc.header.mapr, mrc.header.maps]
        origin = np.array(mrc.nstart.data)
        unit_cell = np.array(mrc.header.cella)
    
    # Save merged map with proper metadata
    logger.info("Saving final merged map...")
    with mrcfile.new(output_map_file_path, overwrite=True) as mrc:
        mrc.set_data(merged_arr)
        mrc.voxel_size = voxel_size
        mrc.header.mapc = axis_order[0]
        mrc.header.mapr = axis_order[1]
        mrc.header.maps = axis_order[2]
        mrc.nstart = origin
        mrc.header.cella = unit_cell
    
    logger.info(f"Successfully saved merged map to {output_map_file_path}")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr3_emready.json',
                        help='JSON file for configuration')
    parser.add_argument('-i', '--input_map', type=str, required=True,
                        help='Input MRC map file path')
    parser.add_argument('-o', '--output_map', type=str, required=True,
                        help='Output MRC map file path')
    parser.add_argument('-b', '--batch_size', type=int, default=4,
                        help='Batch size for inference')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model generator checkpoint (*_gen.pth)')
    parser.add_argument('--phase', type=str, default='test',
                        help='Phase of the model (train, test, etc.)')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='GPU IDs to use')
    parser.add_argument('--enable_wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode')
    
    args = parser.parse_args()
    
    # Parse configs
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    
    # Setup logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    Logger.setup_logger(None, opt['path']['log'], 'inference', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    
    # Create model
    model = create_model(opt)
    logger.info('Model created')
    
    # Load model checkpoint for inference
    if os.path.exists(args.model_path):
        logger.info(f"Loading model from {args.model_path}")
        # For inference, we only need the generator weights
        if isinstance(model.netG, nn.DataParallel):
            model.netG.module.load_state_dict(torch.load(args.model_path))
        else:
            model.netG.load_state_dict(torch.load(args.model_path))
        model.netG.eval()  # Set to evaluation mode
    else:
        logger.error(f"Model checkpoint not found: {args.model_path}")
        return
    
    # Set noise schedule for inference
    model.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    # Use current directory for input and output
    input_dir = "input"
    output_dir = "output"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output directory if it doesn't exist
    output_dir_path = os.path.dirname(args.output_map)
    if output_dir_path and not os.path.exists(output_dir_path):
        logger.info(f"Creating output directory: {output_dir_path}")
        os.makedirs(output_dir_path, exist_ok=True)
    
    try:
        # Step 1: Prepare input (split into chunks)
        logger.info(f"==> Splitting map into chunks: {args.input_map}")
        num_chunks = prepare_input(args.input_map, input_dir)
        logger.info(f"Created {num_chunks} chunks")
        
        # Step 2: Process chunks with model
        logger.info("==> Processing chunks with SR3 model")
        process_chunks(model, input_dir, output_dir, args.batch_size)
        
        # Step 3: Merge chunks
        logger.info("==> Merging processed chunks")
        success = merge_output(output_dir, args.output_map, input_dir, args.input_map)
        
        if success:
            logger.info(f"==> Successfully generated map: {args.output_map}")
        else:
            logger.error("==> Failed to generate map")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise e
    
    finally:
        # Clean up files if needed
        logger.info("==> Processing complete")

if __name__ == "__main__":
    main() 