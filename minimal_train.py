import os
import torch
import argparse
import logging
import core.logger as Logger
from model import create_model
import data as Data
from tensorboardX import SummaryWriter
import core.metrics as Metrics

def expansion3d(x):
    """Add channel dimension to 3D volume"""
    return x.unsqueeze(1)

def verify_data_paths(opt):
    """Verify that data paths exist and contain matching files"""
    logger = logging.getLogger('base')
    
    for phase, dataset_opt in opt['datasets'].items():
        dataroot = dataset_opt['dataroot']
        hr_path = os.path.join(dataroot, dataset_opt['hr_folder'])
        sr_path = os.path.join(dataroot, dataset_opt['sr_folder'])
        
        # Check if directories exist
        if not os.path.exists(hr_path):
            logger.error(f"{phase}: HR path does not exist: {hr_path}")
            return False
        if not os.path.exists(sr_path):
            logger.error(f"{phase}: SR path does not exist: {sr_path}")
            return False
            
        # Check if directories contain files
        hr_files = os.listdir(hr_path)
        sr_files = os.listdir(sr_path)
        
        if len(hr_files) == 0:
            logger.error(f"{phase}: No files found in HR directory: {hr_path}")
            return False
        if len(sr_files) == 0:
            logger.error(f"{phase}: No files found in SR directory: {sr_path}")
            return False
            
        logger.info(f"{phase}: Found {len(hr_files)} files in HR directory")
        logger.info(f"{phase}: Found {len(sr_files)} files in SR directory")
    
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/minimal_train.json',
                        help='JSON file for configuration')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('--phase', type=str, default='train',
                        help='Phase to run: train or val')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--enable_wandb', action='store_true', help='Enable wandb logging')
    args = parser.parse_args()
    
    # Parse configs
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    
    # Setup logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    
    # Setup TensorBoard logger
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])
    
    # Create directories
    os.makedirs(opt['path']['checkpoint'], exist_ok=True)
    
    # Verify data paths
    if not verify_data_paths(opt):
        logger.error("Data path verification failed. Please check your dataset structure.")
        return
    
    # Create train and validation datasets
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
            logger.info(f'Number of training samples: {len(train_set)}')
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
            logger.info(f'Number of validation samples: {len(val_set)}')
    
    # Create model
    diffusion = create_model(opt)
    logger.info('Initial Model Finished')
    
    # Training
    current_step = 0
    current_epoch = 0
    n_iter = opt['train']['n_iter']
    
    # Set noise schedule
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    
    logger.info('Starting Training:')
    logger.info(f'Total iterations: {n_iter}')
    
    while current_step < n_iter:
        current_epoch += 1
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > n_iter:
                break
                
            # Process data
            train_data['HR'] = expansion3d(train_data['HR'])
            train_data['SR'] = expansion3d(train_data['SR'])
            
            # Training step
            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()
            
            # Logging
            if current_step % opt['train']['print_freq'] == 0:
                logs = diffusion.get_current_log()
                message = f'<epoch:{current_epoch:3d}, iter:{current_step:8,d}> '
                for k, v in logs.items():
                    message += f'{k}: {v:.4e} '
                    # Log to TensorBoard
                    tb_logger.add_scalar(k, v, current_step)
                logger.info(message)
            
            # Validation
            if current_step % opt['train']['val_freq'] == 0:
                logger.info('Performing validation...')
                avg_psnr = 0.0
                idx = 0
                for _, val_data in enumerate(val_loader):
                    idx += 1
                    val_data['HR'] = expansion3d(val_data['HR'])
                    val_data['SR'] = expansion3d(val_data['SR'])
                    diffusion.feed_data(val_data)
                    diffusion.test(continous=False)
                    visuals = diffusion.get_current_visuals()
                    avg_psnr += Metrics.calculate_psnr_new(visuals['SR'], visuals['HR'])
                    
                avg_psnr = avg_psnr / idx if idx > 0 else 0
                logger.info(f'# Validation # PSNR: {avg_psnr:.4e}')
                
                # Log validation metrics to TensorBoard
                tb_logger.add_scalar('validation/psnr', avg_psnr, current_step)
            
            # Save checkpoint
            if current_step % opt['train']['save_checkpoint_freq'] == 0:
                logger.info('Saving model checkpoint...')
                diffusion.save_network(current_epoch, current_step)
    
    logger.info('Training completed successfully.')
    tb_logger.close()

if __name__ == '__main__':
    main() 