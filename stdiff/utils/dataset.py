import torch
from torch.utils import data
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torch import Tensor
import pytorch_lightning as pl

import numpy as np
from PIL import Image
from pathlib import Path
import os
import copy
from typing import List
from tqdm import tqdm
import random
from typing import Optional
from itertools import groupby
from operator import itemgetter
from functools import partial
import random
from einops import rearrange
import hashlib
import json

import cv2


def get_global_minmax_cache_key(dataset_dir, test_folder_ids, num_observed_frames, num_predict_frames, use_val):
    """
    Generate a cache key for global min/max values based on training sequence configuration.
    
    Args:
        dataset_dir: Path to dataset directory
        test_folder_ids: List of test folder IDs (determines training sequences)
        num_observed_frames: Number of observed frames
        num_predict_frames: Number of predicted frames
        use_val: Whether validation split is used
    
    Returns:
        str: Cache key (hash)
    """
    # Create a unique key from the configuration
    key_data = {
        'dataset_dir': str(Path(dataset_dir).absolute()),
        'test_folder_ids': sorted([int(i) for i in test_folder_ids]),  # Sort for consistency
        'num_observed_frames': int(num_observed_frames),
        'num_predict_frames': int(num_predict_frames),
        'use_val': bool(use_val)
    }
    # Create hash from sorted JSON string
    key_string = json.dumps(key_data, sort_keys=True)
    cache_key = hashlib.md5(key_string.encode()).hexdigest()
    return cache_key


def load_global_minmax_from_cache(dataset_dir, test_folder_ids, num_observed_frames, num_predict_frames, use_val):
    """
    Load global min/max values from cache if available.
    
    Returns:
        tuple: (global_min, global_max) if found, (None, None) otherwise
    """
    cache_key = get_global_minmax_cache_key(dataset_dir, test_folder_ids, num_observed_frames, num_predict_frames, use_val)
    
    # Store cache in dataset directory under .cache subdirectory
    cache_dir = Path(dataset_dir) / '.cache'
    cache_file = cache_dir / f'global_minmax_{cache_key}.json'
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                global_min = cache_data.get('global_min')
                global_max = cache_data.get('global_max')
                if global_min is not None and global_max is not None:
                    print(f"Loaded global min/max from cache: [{global_min:.6f}, {global_max:.6f}]")
                    print(f"Cache file: {cache_file}")
                    return float(global_min), float(global_max)
        except Exception as e:
            print(f"Warning: Failed to load cache file {cache_file}: {e}")
    
    return None, None


def save_global_minmax_to_cache(dataset_dir, test_folder_ids, num_observed_frames, num_predict_frames, use_val, global_min, global_max):
    """
    Save global min/max values to cache.
    """
    cache_key = get_global_minmax_cache_key(dataset_dir, test_folder_ids, num_observed_frames, num_predict_frames, use_val)
    
    # Store cache in dataset directory under .cache subdirectory
    cache_dir = Path(dataset_dir) / '.cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f'global_minmax_{cache_key}.json'
    
    cache_data = {
        'global_min': float(global_min),
        'global_max': float(global_max),
        'dataset_dir': str(Path(dataset_dir).absolute()),
        'test_folder_ids': sorted([int(i) for i in test_folder_ids]),
        'num_observed_frames': int(num_observed_frames),
        'num_predict_frames': int(num_predict_frames),
        'use_val': bool(use_val)
    }
    
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Saved global min/max to cache: [{global_min:.6f}, {global_max:.6f}]")
        print(f"Cache file: {cache_file}")
    except Exception as e:
        print(f"Warning: Failed to save cache file {cache_file}: {e}")


class LitDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.len_train_loader = None
        self.len_val_loader = None
        self.len_test_loader = None
        self.img_size = cfg.Dataset.image_size

        self.norm_transform = lambda x: x * 2. - 1.

        if cfg.Dataset.name == 'KTH':
            self.train_transform = transforms.Compose([VidCenterCrop((120, 120)), VidResize((self.img_size, self.img_size)), VidRandomHorizontalFlip(0.5), VidRandomVerticalFlip(0.5), VidToTensor(), self.norm_transform])
            self.test_transform = transforms.Compose([VidCenterCrop((120, 120)), VidResize((self.img_size, self.img_size)), VidToTensor(), self.norm_transform])
            self.val_person_ids = [5]
        
        if cfg.Dataset.name == 'KITTI':
            self.train_transform = transforms.Compose([VidResize((self.img_size, self.img_size)), VidRandomHorizontalFlip(0.5), VidRandomVerticalFlip(0.5), VidToTensor(), self.norm_transform])
            self.test_transform = transforms.Compose([VidResize((self.img_size, self.img_size)), VidToTensor(), self.norm_transform])

        if cfg.Dataset.name == 'KITTI_RANGE':
            # Range images are already 64x2048, resize to model input size if needed
            # Handle ListConfig from omegaconf
            if hasattr(self.img_size, '__getitem__') and hasattr(self.img_size, '__len__') and len(self.img_size) == 2:
                # Non-square image size: [height, width] (handles list, tuple, ListConfig)
                resize_size = (int(self.img_size[0]), int(self.img_size[1]))
            else:
                # Square image size (backward compatibility)
                resize_size = (int(self.img_size), int(self.img_size))
            # Initialize global min/max as None, will be computed in setup()
            self.range_image_global_min = None
            self.range_image_global_max = None
            # Transforms will be set up in setup() after global min/max is computed
            # For now, just store resize_size
            self.kitti_range_resize_size = resize_size

        if cfg.Dataset.name == 'MNIST':
            self.train_transform = transforms.Compose([VidResize((self.img_size, self.img_size)), VidRandomHorizontalFlip(0.5), VidRandomVerticalFlip(0.5), VidToTensor(), self.norm_transform])
            self.test_transform = transforms.Compose([VidResize((self.img_size, self.img_size)), VidToTensor(), self.norm_transform])
        
        if cfg.Dataset.name == 'SMMNIST':
            self.train_transform = self.test_transform = transforms.Compose([VidResize((self.img_size, self.img_size)), VidToTensor(), self.norm_transform])
            self.test_transform = self.test_transform = transforms.Compose([VidResize((self.img_size, self.img_size)), VidToTensor(), self.norm_transform])

        if cfg.Dataset.name == 'BAIR':
            self.train_transform = transforms.Compose([VidResize((self.img_size, self.img_size)), VidRandomHorizontalFlip(0.5), VidRandomVerticalFlip(0.5), VidToTensor(), self.norm_transform])
            self.test_transform = transforms.Compose([VidResize((self.img_size, self.img_size)), VidToTensor(), self.norm_transform])

        if cfg.Dataset.name == 'CityScapes':
            self.train_transform = transforms.Compose([VidResize((self.img_size, self.img_size)), VidToTensor(), self.norm_transform])
            self.test_transform = transforms.Compose([VidResize((self.img_size, self.img_size)), VidToTensor(), self.norm_transform])
        
        if cfg.Dataset.name == 'Human36M':
            self.train_transform = transforms.Compose([VidResize((self.img_size, self.img_size)), VidToTensor(), self.norm_transform])
            self.test_transform = transforms.Compose([VidResize((self.img_size, self.img_size)), VidToTensor(), self.norm_transform])
        
        o_resize = None
        p_resize = None
        vp_size = cfg.STDiff.Diffusion.unet_config.sample_size
        vo_size = cfg.STDiff.DiffNet.MotionEncoder.image_size
        # Handle non-square sizes: convert to tuple for comparison (handles list, tuple, ListConfig)
        if hasattr(vp_size, '__getitem__') and hasattr(vp_size, '__len__') and len(vp_size) == 2:
            vp_size_tuple = (int(vp_size[0]), int(vp_size[1]))
        else:
            vp_size_tuple = (int(vp_size), int(vp_size))
        if hasattr(vo_size, '__getitem__') and hasattr(vo_size, '__len__') and len(vo_size) == 2:
            vo_size_tuple = (int(vo_size[0]), int(vo_size[1]))
        else:
            vo_size_tuple = (int(vo_size), int(vo_size))
        if hasattr(self.img_size, '__getitem__') and hasattr(self.img_size, '__len__') and len(self.img_size) == 2:
            img_size_tuple = (int(self.img_size[0]), int(self.img_size[1]))
        else:
            img_size_tuple = (int(self.img_size), int(self.img_size))
        if vp_size_tuple != img_size_tuple:
            p_resize = transforms.Resize(vp_size_tuple, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        if vo_size_tuple != img_size_tuple:
            o_resize = transforms.Resize(vo_size_tuple, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        self.collate_fn = partial(svrfcn, rand_Tp=cfg.Dataset.rand_Tp, rand_predict=cfg.Dataset.rand_predict, o_resize=o_resize, p_resize=p_resize, half_fps=cfg.Dataset.half_fps)

    def setup(self, stage: Optional[str] = None):
        # Assign Train/val split(s) for use in Dataloaders
        if stage in (None, "fit"):
            if self.cfg.Dataset.name == 'KTH':
                KTHTrainData = KTHDataset(self.cfg.Dataset.dir, transform = self.train_transform, train = True, val = True, 
                                          num_observed_frames= self.cfg.Dataset.num_observed_frames, num_predict_frames= self.cfg.Dataset.num_predict_frames,
                                          val_person_ids = self.val_person_ids)#, actions = ['walking_no_empty'])
                self.train_set, self.val_set = KTHTrainData()
            
            if self.cfg.Dataset.name == 'KITTI':
                KITTITrainData = KITTIDataset(self.cfg.Dataset.dir, [10, 11, 12, 13], transform = self.train_transform, train = True, val = True,
                                                num_observed_frames= self.cfg.Dataset.num_observed_frames, num_predict_frames= self.cfg.Dataset.num_predict_frames
                                                )
                self.train_set, self.val_set = KITTITrainData()

            if self.cfg.Dataset.name == 'KITTI_RANGE':
                # Get test folder IDs from config or use default
                test_folder_ids = self.cfg.Dataset.get("test_folder_ids", [8, 9, 10])
                # Convert to integers in case YAML parsed them as strings (e.g., [06, 07, 08, 09])
                test_folder_ids = [int(i) for i in test_folder_ids]
                # In deploy mode, use all sequences for training (no validation split)
                # Otherwise, split for validation
                use_val = self.cfg.Dataset.phase != 'deploy'
                
                # Try to load global min/max from cache first
                cached_min, cached_max = load_global_minmax_from_cache(
                    self.cfg.Dataset.dir, test_folder_ids, 
                    self.cfg.Dataset.num_observed_frames, 
                    self.cfg.Dataset.num_predict_frames, 
                    use_val
                )
                
                if cached_min is not None and cached_max is not None:
                    # Use cached values
                    self.range_image_global_min = cached_min
                    self.range_image_global_max = cached_max
                    # Build transforms and train/val sets (same as cache-miss path)
                    self.train_transform = transforms.Compose([
                        VidResize((self.kitti_range_resize_size[0], self.kitti_range_resize_size[1])),
                        VidToTensor(),
                        self.norm_transform
                    ])
                    self.test_transform = transforms.Compose([
                        VidResize((self.kitti_range_resize_size[0], self.kitti_range_resize_size[1])),
                        VidToTensor(),
                        self.norm_transform
                    ])
                    KITTIRangeTrainData = KITTIRangeImageDataset(self.cfg.Dataset.dir, test_folder_ids,
                                                                 transform=self.train_transform, train=True, val=use_val,
                                                                 num_observed_frames=self.cfg.Dataset.num_observed_frames,
                                                                 num_predict_frames=self.cfg.Dataset.num_predict_frames,
                                                                 global_min=self.range_image_global_min,
                                                                 global_max=self.range_image_global_max)
                    if use_val:
                        self.train_set, self.val_set = KITTIRangeTrainData()
                    else:
                        self.train_set = KITTIRangeTrainData()
                        self.val_set = None
                else:
                    # Cache miss - compute global min/max from training set
                    # First, create dataset without transform to compute global min/max
                    # We'll use a dummy transform for now
                    # Note: Don't pass global_min/global_max here since we're computing them
                    dummy_transform = transforms.Compose([VidToTensor()])
                    KITTIRangeTrainData_temp = KITTIRangeImageDataset(self.cfg.Dataset.dir, test_folder_ids, 
                                                                       transform = dummy_transform, train = True, val = use_val,
                                                                       num_observed_frames= self.cfg.Dataset.num_observed_frames, 
                                                                       num_predict_frames= self.cfg.Dataset.num_predict_frames)
                    if use_val:
                        train_set_temp, _ = KITTIRangeTrainData_temp()
                    else:
                        train_set_temp = KITTIRangeTrainData_temp()
                    
                    # Compute global min/max from training set (only valid pixels)
                    # Use a sample-based approach for efficiency (sample up to 1000 clips)
                    print("Computing global min/max from training set...")
                    all_valid_pixels = []
                    num_clips_to_sample = min(1000, len(train_set_temp))  # Sample up to 1000 clips
                    sample_indices = random.sample(range(len(train_set_temp)), num_clips_to_sample) if len(train_set_temp) > 1000 else range(len(train_set_temp))
                    
                    for idx in tqdm(sample_indices, desc="Computing global min/max", disable=False):
                        clip_files = train_set_temp.clips[idx]
                        for npy_path in clip_files:
                            range_img = np.load(npy_path.absolute().as_posix())
                            if len(range_img.shape) > 2:
                                range_img = range_img.squeeze()
                            valid_mask = range_img > 0
                            if valid_mask.any():
                                all_valid_pixels.extend(range_img[valid_mask].astype(np.float32).tolist())
                    
                    if len(all_valid_pixels) > 0:
                        all_valid_pixels_array = np.array(all_valid_pixels, dtype=np.float32)
                        self.range_image_global_min = float(np.min(all_valid_pixels_array))
                        self.range_image_global_max = float(np.max(all_valid_pixels_array))
                        print(f"Computed global min from {len(all_valid_pixels)} valid pixels: {self.range_image_global_min:.6f}")
                        print(f"Computed global max from {len(all_valid_pixels)} valid pixels: {self.range_image_global_max:.6f}")
                        
                        # Save to cache for future use
                        save_global_minmax_to_cache(
                            self.cfg.Dataset.dir, test_folder_ids,
                            self.cfg.Dataset.num_observed_frames,
                            self.cfg.Dataset.num_predict_frames,
                            use_val,
                            self.range_image_global_min,
                            self.range_image_global_max
                        )
                    else:
                        self.range_image_global_min = 0.0
                        self.range_image_global_max = 85.0
                        print("Warning: No valid pixels found, using min=0.0, max=85.0")
                    
                    # Now create transforms with global min-max normalization
                    self.train_transform = transforms.Compose([
                        VidResize((self.kitti_range_resize_size[0], self.kitti_range_resize_size[1])),
                        VidToTensor(),
                        self.norm_transform
                    ])
                    self.test_transform = transforms.Compose([
                        VidResize((self.kitti_range_resize_size[0], self.kitti_range_resize_size[1])),
                        VidToTensor(),
                        self.norm_transform
                    ])
                    
                    # Create actual datasets with proper transforms and global min/max
                    KITTIRangeTrainData = KITTIRangeImageDataset(self.cfg.Dataset.dir, test_folder_ids, 
                                                                 transform = self.train_transform, train = True, val = use_val,
                                                             num_observed_frames= self.cfg.Dataset.num_observed_frames, 
                                                             num_predict_frames= self.cfg.Dataset.num_predict_frames,
                                                             global_min=self.range_image_global_min, 
                                                             global_max=self.range_image_global_max)
                    if use_val:
                        self.train_set, self.val_set = KITTIRangeTrainData()
                    else:
                        self.train_set = KITTIRangeTrainData()
                        self.val_set = None

            if self.cfg.Dataset.name == 'BAIR':
                BAIR_train_whole_set = BAIRDataset(Path(self.cfg.Dataset.dir).joinpath('train'), self.train_transform, color_mode = 'RGB', 
                                                   num_observed_frames = self.cfg.Dataset.num_observed_frames, num_predict_frames = self.cfg.Dataset.num_predict_frames,
                                                   )()
                train_val_ratio = 0.95
                BAIR_train_set_length = int(len(BAIR_train_whole_set) * train_val_ratio)
                BAIR_val_set_length = len(BAIR_train_whole_set) - BAIR_train_set_length
                self.train_set, self.val_set = random_split(BAIR_train_whole_set, [BAIR_train_set_length, BAIR_val_set_length],
                                                            generator=torch.Generator().manual_seed(2021))
            if self.cfg.Dataset.name == 'CityScapes':
                self.train_set = CityScapesDataset(Path(self.cfg.Dataset.dir).joinpath('train'), self.train_transform, color_mode = 'RGB', 
                                                   num_observed_frames = self.cfg.Dataset.num_observed_frames, num_predict_frames = self.cfg.Dataset.num_predict_frames,
                                                   )()
                self.val_set = CityScapesDataset(Path(self.cfg.Dataset.dir).joinpath('val'), self.train_transform, color_mode = 'RGB', 
                                                   num_observed_frames = self.cfg.Dataset.num_observed_frames, num_predict_frames = self.cfg.Dataset.num_predict_frames,
                                                   )()

            if self.cfg.Dataset.name == 'MNIST':
                self.train_set = MovingMNISTDataset(Path(self.cfg.Dataset.dir).joinpath('moving-mnist-train.npz'), self.train_transformo)
                self.val_set = MovingMNISTDataset(Path(self.cfg.Dataset.dir).joinpath('moving-mnist-valid.npz'), self.train_transform)
            
            if self.cfg.Dataset.name == 'SMMNIST':
                self.train_set = StochasticMovingMNIST(True, Path(self.cfg.Dataset.dir), self.cfg.Dataset.num_observed_frames, self.cfg.Dataset.num_predict_frames, self.train_transform)
                train_val_ratio = 0.95
                SMMNIST_train_set_length = int(len(self.train_set) * train_val_ratio)
                SMMNIST_val_set_length = len(self.train_set) - SMMNIST_train_set_length
                self.train_set, self.val_set = random_split(self.train_set, [SMMNIST_train_set_length, SMMNIST_val_set_length],
                                                            generator=torch.Generator().manual_seed(2021))
            
            if self.cfg.Dataset.name == 'Human36M':
                self.train_set = Human36MDataset(Path(self.cfg.Dataset.dir).joinpath('train'), self.train_transform, color_mode = 'RGB', num_observed_frames = self.cfg.Dataset.num_observed_frames, num_predict_frames = self.cfg.Dataset.num_predict_frames)()
                self.val_set = Human36MDataset(Path(self.cfg.Dataset.dir).joinpath('valid'), self.train_transform, color_mode = 'RGB', num_observed_frames = self.cfg.Dataset.num_observed_frames, num_predict_frames = self.cfg.Dataset.num_predict_frames)()

            #Use all training dataset for the final training
            if self.cfg.Dataset.phase == 'deploy':
                if hasattr(self, 'val_set') and self.val_set is not None:
                    self.train_set = ConcatDataset([self.train_set, self.val_set])
                # If val_set is None, train_set already contains all data

            dev_set_size = self.cfg.Dataset.dev_set_size
            if dev_set_size is not None:
                self.train_set, _ = random_split(self.train_set, [dev_set_size, len(self.train_set) - dev_set_size], generator=torch.Generator().manual_seed(2021))
                if hasattr(self, 'val_set') and self.val_set is not None:
                    self.val_set, _ = random_split(self.val_set, [dev_set_size, len(self.val_set) - dev_set_size], generator=torch.Generator().manual_seed(2021))
            
            # Only compute train/val loader lengths if train_set exists (i.e., in fit stage)
            if hasattr(self, 'train_set'):
                self.len_train_loader = len(self.train_dataloader())
                if hasattr(self, 'val_set') and self.val_set is not None:
                    self.len_val_loader = len(self.val_dataloader())
                else:
                    self.len_val_loader = 0
            else:
                # In test stage, train_set doesn't exist
                self.len_train_loader = 0
                self.len_val_loader = 0

        # Assign Test split(s) for use in Dataloaders (also when stage="fit" so test_loader is available)
        if stage in (None, "fit", "test"):
            if self.cfg.Dataset.name == 'KTH':
                KTHTestData = KTHDataset(self.cfg.Dataset.dir, transform = self.test_transform, train = False, val = False, 
                                        num_observed_frames= self.cfg.Dataset.test_num_observed_frames, num_predict_frames= self.cfg.Dataset.test_num_predict_frames)#, actions = ['walking_no_empty'])
                self.test_set = KTHTestData()
            
            if self.cfg.Dataset.name == 'KITTI':
                KITTITrainData = KITTIDataset(self.cfg.Dataset.dir, [10, 11, 12, 13], transform = self.test_transform, train = False, val = False,
                                                num_observed_frames= self.cfg.Dataset.test_num_observed_frames, num_predict_frames= self.cfg.Dataset.test_num_predict_frames,
                                                )
                self.test_set = KITTITrainData()

            if self.cfg.Dataset.name == 'KITTI_RANGE':
                test_folder_ids = self.cfg.Dataset.get("test_folder_ids", [8, 9, 10])
                # Convert to integers in case YAML parsed them as strings (e.g., [06, 07, 08, 09])
                test_folder_ids = [int(i) for i in test_folder_ids]
                
                # Initialize transforms if they don't exist (needed when stage="test" only)
                if not hasattr(self, 'test_transform') or self.test_transform is None:
                    # Transforms need global min/max, so we'll set them up after we get those values
                    pass  # Will be set up below after global min/max are determined
                
                # Use global min/max computed during setup (if available)
                # If not available (e.g., when setup is called with stage="test" only), try cache or compute them
                global_min = getattr(self, 'range_image_global_min', None)
                global_max = getattr(self, 'range_image_global_max', None)
                
                if global_min is None or global_max is None:
                    # Try to load from cache first
                    use_val = self.cfg.Dataset.phase != 'deploy'
                    cached_min, cached_max = load_global_minmax_from_cache(
                        self.cfg.Dataset.dir, test_folder_ids,
                        self.cfg.Dataset.num_observed_frames,
                        self.cfg.Dataset.num_predict_frames,
                        use_val
                    )
                    
                    if cached_min is not None and cached_max is not None:
                        # Use cached values
                        global_min = cached_min
                        global_max = cached_max
                        self.range_image_global_min = global_min
                        self.range_image_global_max = global_max
                    else:
                        # Cache miss - compute global min/max from training set
                        print("Computing global min/max for test dataset...")
                        dummy_transform = transforms.Compose([VidToTensor()])
                        KITTIRangeTrainData_temp = KITTIRangeImageDataset(self.cfg.Dataset.dir, test_folder_ids, 
                                                                           transform = dummy_transform, train = True, val = use_val,
                                                                           num_observed_frames= self.cfg.Dataset.num_observed_frames, 
                                                                           num_predict_frames= self.cfg.Dataset.num_predict_frames)
                        if use_val:
                            train_set_temp, _ = KITTIRangeTrainData_temp()
                        else:
                            train_set_temp = KITTIRangeTrainData_temp()
                        
                        # Compute global min/max from training set (only valid pixels)
                        all_valid_pixels = []
                        num_clips_to_sample = min(1000, len(train_set_temp))
                        sample_indices = random.sample(range(len(train_set_temp)), num_clips_to_sample) if len(train_set_temp) > 1000 else range(len(train_set_temp))
                        
                        for idx in tqdm(sample_indices, desc="Computing global min/max", disable=False):
                            clip_files = train_set_temp.clips[idx]
                            for npy_path in clip_files:
                                range_img = np.load(npy_path.absolute().as_posix())
                                if len(range_img.shape) > 2:
                                    range_img = range_img.squeeze()
                                valid_mask = range_img > 0
                                if valid_mask.any():
                                    all_valid_pixels.extend(range_img[valid_mask].astype(np.float32).tolist())
                        
                        if len(all_valid_pixels) > 0:
                            all_valid_pixels_array = np.array(all_valid_pixels, dtype=np.float32)
                            global_min = float(np.min(all_valid_pixels_array))
                            global_max = float(np.max(all_valid_pixels_array))
                            self.range_image_global_min = global_min
                            self.range_image_global_max = global_max
                            print(f"Computed global min from {len(all_valid_pixels)} valid pixels: {global_min:.6f}")
                            print(f"Computed global max from {len(all_valid_pixels)} valid pixels: {global_max:.6f}")
                            
                            # Save to cache for future use
                            save_global_minmax_to_cache(
                                self.cfg.Dataset.dir, test_folder_ids,
                                self.cfg.Dataset.num_observed_frames,
                                self.cfg.Dataset.num_predict_frames,
                                use_val,
                                global_min,
                                global_max
                            )
                        else:
                            global_min = 0.0
                            global_max = 85.0
                            self.range_image_global_min = global_min
                            self.range_image_global_max = global_max
                            print("Warning: No valid pixels found, using min=0.0, max=85.0")
                
                # Create test_transform if it doesn't exist (needed when stage="test" only)
                if not hasattr(self, 'test_transform') or self.test_transform is None:
                    self.test_transform = transforms.Compose([
                        VidResize((self.kitti_range_resize_size[0], self.kitti_range_resize_size[1])),
                        VidToTensor(),
                        self.norm_transform
                    ])
                
                KITTIRangeTestData = KITTIRangeImageDataset(self.cfg.Dataset.dir, test_folder_ids,
                                                            transform = self.test_transform, train = False, val = False,
                                                            num_observed_frames= self.cfg.Dataset.test_num_observed_frames, 
                                                            num_predict_frames= self.cfg.Dataset.test_num_predict_frames,
                                                            global_min=global_min, 
                                                            global_max=global_max)
                self.test_set = KITTIRangeTestData()

            if self.cfg.Dataset.name == 'BAIR':
                self.test_set = BAIRDataset(Path(self.cfg.Dataset.dir).joinpath('test'), self.test_transform, color_mode = 'RGB', 
                                            num_observed_frames= self.cfg.Dataset.test_num_observed_frames, num_predict_frames= self.cfg.Dataset.test_num_predict_frames, )()
            if self.cfg.Dataset.name == 'CityScapes':
                self.test_set = CityScapesDataset(Path(self.cfg.Dataset.dir).joinpath('test'), self.test_transform, color_mode = 'RGB', 
                                                   num_observed_frames = self.cfg.Dataset.test_num_observed_frames, num_predict_frames = self.cfg.Dataset.test_num_predict_frames,
                                                   )()
            if self.cfg.Dataset.name == 'MNIST':
                self.test_set = MovingMNISTDataset(Path(self.cfg.Dataset.dir).joinpath('moving-mnist-test.npz'), self.test_transform)
            
            if self.cfg.Dataset.name == 'SMMNIST':
                self.test_set = StochasticMovingMNIST(False, Path(self.cfg.Dataset.dir), self.cfg.Dataset.test_num_observed_frames, self.cfg.Dataset.test_num_predict_frames, self.test_transform)
            
            if self.cfg.Dataset.name == 'Human36M':
                self.test_set = Human36MDataset(Path(self.cfg.Dataset.dir).joinpath('test'), self.train_transform, color_mode = 'RGB', num_observed_frames = self.cfg.Dataset.test_num_observed_frames, num_predict_frames = self.cfg.Dataset.test_num_predict_frames)()

            dev_set_size = self.cfg.Dataset.dev_set_size
            if dev_set_size is not None:
                self.test_set, _ = random_split(self.test_set, [dev_set_size, len(self.test_set) - dev_set_size], generator=torch.Generator().manual_seed(2021))
            self.len_test_loader = len(self.test_dataloader())

    def train_dataloader(self):
        if not hasattr(self, 'train_set') or self.train_set is None:
            return None
        return DataLoader(self.train_set, shuffle = True, batch_size=self.cfg.Dataset.batch_size, num_workers=self.cfg.Dataset.num_workers, drop_last = True, collate_fn = self.collate_fn)

    def val_dataloader(self):
        if self.val_set is not None:
            return DataLoader(self.val_set, shuffle = False, batch_size=self.cfg.Dataset.batch_size, num_workers=self.cfg.Dataset.num_workers, drop_last = True, collate_fn = self.collate_fn)
        else:
            return None

    def test_dataloader(self):
        return DataLoader(self.test_set, shuffle = False, batch_size=self.cfg.Dataset.batch_size, num_workers=self.cfg.Dataset.num_workers, drop_last = False, collate_fn = self.collate_fn)


def get_lightning_module_dataloader(cfg, stage=None):
    pl_datamodule = LitDataModule(cfg)
    # If stage not provided: use "fit" when phase != 'deploy', else "test" (test script only needs test set)
    # Caller can pass stage="fit" for training (need train set) or stage="test" for testing only
    if stage is None:
        stage = "fit" if cfg.Dataset.phase != 'deploy' else "test"
    pl_datamodule.setup(stage=stage)
    
    # Only create loaders for datasets that exist
    train_loader = None
    val_loader = None
    if hasattr(pl_datamodule, 'train_set') and pl_datamodule.train_set is not None:
        train_loader = pl_datamodule.train_dataloader()
    if hasattr(pl_datamodule, 'val_set') and pl_datamodule.val_set is not None:
        val_loader = pl_datamodule.val_dataloader()
    
    test_loader = pl_datamodule.test_dataloader()
    # Return datamodule as well so we can access global min/max for KITTI_RANGE
    return train_loader, val_loader, test_loader, pl_datamodule

class KTHDataset(object):
    """
    KTH dataset, a wrapper for ClipDataset
    the original frame size is (H, W) = (120, 160)
    Split the KTH dataset and return the train and test dataset
    """
    def __init__(self, KTH_dir, transform, train, val,
                 num_observed_frames, num_predict_frames, actions=['boxing', 'handclapping', 'handwaving', 'jogging_no_empty', 'running_no_empty', 'walking_no_empty'], val_person_ids = None
                 ):
        """
        Args:
            KTH_dir --- Directory for extracted KTH video frames
            train --- True for training dataset, False for test dataset
            transform --- trochvison transform functions
            num_observed_frames --- number of past frames
            num_predict_frames --- number of future frames
        """
        self.num_observed_frames = num_observed_frames
        self.num_predict_frames = num_predict_frames
        self.clip_length = num_observed_frames + num_predict_frames
        self.transform = transform
        self.color_mode = 'RGB'

        self.KTH_path = Path(KTH_dir).absolute()
        self.actions = actions
        self.train = train
        self.val = val
        if self.train:
            self.person_ids = list(range(1, 17))
            if self.val:
                if val_person_ids is None: #one person for the validation
                    self.val_person_ids = [random.randint(1, 17)]
                    self.person_ids.remove(self.val_person_ids[0])
                else:
                    self.val_person_ids = val_person_ids
        else:
            self.person_ids = list(range(17, 26))

        frame_folders = self.__getFramesFolder__(self.person_ids)
        self.clips = self.__getClips__(frame_folders)
        
        if self.val:
            val_frame_folders = self.__getFramesFolder__(self.val_person_ids)
            self.val_clips = self.__getClips__(val_frame_folders)

    def __call__(self):
        """
        Returns:
            clip_set --- ClipDataset object
        """
        
        clip_set = ClipDataset(self.num_observed_frames, self.num_predict_frames, self.clips, self.transform, self.color_mode)
        if self.val:
            val_clip_set = ClipDataset(self.num_observed_frames, self.num_predict_frames, self.val_clips, self.transform, self.color_mode)
            return clip_set, val_clip_set
        else:
            return clip_set
    
    def __getClips__(self, frame_folders):
        clips = []
        for folder in frame_folders:
            img_files = sorted(list(folder.glob('*')))
            clip_num = len(img_files) // self.clip_length
            rem_num = len(img_files) % self.clip_length
            img_files = img_files[rem_num // 2 : rem_num//2 + clip_num*self.clip_length]
            for i in range(clip_num):
                clips.append(img_files[i*self.clip_length : (i+1)*self.clip_length])

        return clips
    
    def __getFramesFolder__(self, person_ids):
        """
        Get the KTH frames folders for ClipDataset
        Returns:
            return_folders --- ther returned video frames folders
        """

        frame_folders = []
        for a in self.actions:
            action_path = self.KTH_path.joinpath(a)
            frame_folders.extend([action_path.joinpath(s) for s in os.listdir(action_path) if '.avi' not in s])
        frame_folders = sorted(frame_folders)

        return_folders = []
        for ff in frame_folders:
            person_id = int(str(ff.name).strip().split('_')[0][-2:])
            if person_id in person_ids:
                return_folders.append(ff)

        return return_folders

class BAIRDataset(object):
    """
    BAIR dataset, a wrapper for ClipDataset
    the original frame size is (H, W) = (64, 64)
    The train and test frames has been previously splitted: ref "Self-Supervised Visual Planning with Temporal Skip Connections"
    """
    def __init__(self, frames_dir: str, transform, color_mode = 'RGB', 
                 num_observed_frames = 10, num_predict_frames = 10):
        """
        Args:
            frames_dir --- Directory of extracted video frames and original videos.
            transform --- trochvison transform functions
            color_mode --- 'RGB' or 'grey_scale' color mode for the dataset
            num_observed_frames --- number of past frames
            num_predict_frames --- number of future frames
            clip_length --- number of frames for each video clip example for model
        """
        self.frames_path = Path(frames_dir).absolute()
        self.num_observed_frames = num_observed_frames
        self.num_predict_frames = num_predict_frames
        self.clip_length = num_observed_frames + num_predict_frames
        self.transform = transform
        self.color_mode = color_mode

        self.clips = self.__getClips__()


    def __call__(self):
        """
        Returns:
            data_set --- ClipDataset object
        """
        data_set = ClipDataset(self.num_observed_frames, self.num_predict_frames, self.clips, self.transform, self.color_mode)

        return data_set
    
    def __getClips__(self):
        clips = []
        frames_folders = os.listdir(self.frames_path)
        frames_folders = [self.frames_path.joinpath(s) for s in frames_folders]
        for folder in frames_folders:
            img_files = sorted(list(folder.glob('*')))
            clip_num = len(img_files) // self.clip_length
            rem_num = len(img_files) % self.clip_length
            img_files = img_files[rem_num // 2 : rem_num//2 + clip_num*self.clip_length]
            for i in range(clip_num):
                clips.append(img_files[i*self.clip_length : (i+1)*self.clip_length])

        return clips

class CityScapesDataset(BAIRDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __getClips__(self):
        clips = []
        frames_folders = os.listdir(self.frames_path)
        frames_folders = [self.frames_path.joinpath(s) for s in frames_folders]
        for folder in frames_folders:
            all_imgs = sorted(list(folder.glob('*')))
            obj_dict = {}
            for f in all_imgs:
                id = str(f).split('_')[1]
                if id in obj_dict:
                    obj_dict[id].append(f)
                else:
                    obj_dict[id] = [f]
            for k, img_files in obj_dict.items():
                for k, g in groupby(enumerate(img_files), lambda ix: ix[0]-int(str(ix[1]).split('_')[2])):
                    clip_files = list(list(zip(*list(g)))[1])
                    
                    clip_num = len(clip_files) // self.clip_length
                    rem_num = len(clip_files) % self.clip_length
                    clip_files = clip_files[rem_num // 2 : rem_num//2 + clip_num*self.clip_length]
                    for i in range(clip_num):
                        clips.append(clip_files[i*self.clip_length : (i+1)*self.clip_length])

        return clips

class KITTIDataset(object):
    def __init__(self, KITTI_dir, test_folder_ids, transform, train, val,
                 num_observed_frames, num_predict_frames):
        """
        Args:
            KITTI_dir --- Directory for extracted KITTI video frames
            train --- True for training dataset, False for test dataset
            transform --- trochvison transform functions
            num_observed_frames --- number of past frames
            num_predict_frames --- number of future frames
        """
        self.num_observed_frames = num_observed_frames
        self.num_predict_frames = num_predict_frames
        self.clip_length = num_observed_frames + num_predict_frames
        self.transform = transform
        self.color_mode = 'RGB'

        self.KITTI_path = Path(KITTI_dir).absolute()
        self.train = train
        self.val = val

        # Convert test_folder_ids to integers in case they're strings (e.g., from YAML with leading zeros)
        test_folder_ids = [int(i) for i in test_folder_ids]

        self.all_folders = sorted(os.listdir(self.KITTI_path))
        self.num_examples = len(self.all_folders)
        
        self.folder_id = list(range(self.num_examples))
        if self.train:
            self.train_folders = [self.all_folders[i] for i in range(self.num_examples) if i not in test_folder_ids]
            if self.val:
                self.val_folders = self.train_folders[0:2]
                self.train_folders = self.train_folders[2:]
    
        else:
            self.test_folders = [self.all_folders[i] for i in test_folder_ids]
        
        if self.train:
            self.train_clips = self.__getClips__(self.train_folders)
            if self.val:
                self.val_clips = self.__getClips__(self.val_folders)
        else:
            self.test_clips = self.__getClips__(self.test_folders)

    def __call__(self):
        """
        Returns:
            clip_set --- ClipDataset object
        """
        if self.train:
            clip_set = ClipDataset(self.num_observed_frames, self.num_predict_frames, self.train_clips, self.transform, self.color_mode)
            if self.val:
                val_clip_set = ClipDataset(self.num_observed_frames, self.num_predict_frames, self.val_clips, self.transform, self.color_mode)
                return clip_set, val_clip_set
            return clip_set
        else:
            return ClipDataset(self.num_observed_frames, self.num_predict_frames, self.test_clips, self.transform, self.color_mode)
    
    def __getClips__(self, frame_folders):
        clips = []
        for folder in frame_folders:
            img_files = sorted(list(self.KITTI_path.joinpath(folder).glob('*')))
            clip_num = len(img_files) // self.clip_length
            rem_num = len(img_files) % self.clip_length
            img_files = img_files[rem_num // 2 : rem_num//2 + clip_num*self.clip_length]
            for i in range(clip_num):
                clips.append(img_files[i*self.clip_length : (i+1)*self.clip_length])

        return clips
    
class ClipDataset(Dataset):
    """
    Video clips dataset
    """
    def __init__(self, num_observed_frames, num_predict_frames, clips, transform, color_mode):
        """
        Args:
            num_observed_frames --- number of past frames
            num_predict_frames --- number of future frames
            clips --- List of video clips frames file path
            transfrom --- torchvision transforms for the image
            color_mode --- 'RGB' for RGB dataset, 'grey_scale' for grey_scale dataset

        Return batched Sample:
            past_clip --- Tensor with shape (batch_size, num_observed_frames, C, H, W)
            future_clip --- Tensor with shape (batch_size, num_predict_frames, C, H, W)
        """
        self.num_observed_frames = num_observed_frames
        self.num_predict_frames = num_predict_frames
        self.clips = clips
        self.transform = transform
        if color_mode != 'RGB' and color_mode != 'grey_scale':
            raise ValueError("Unsupported color mode!!")
        else:
            self.color_mode = color_mode

    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, index: int):
        """
        Returns:
            past_clip: Tensor with shape (num_observed_frames, C, H, W)
            future_clip: Tensor with shape (num_predict_frames, C, H, W)
        """
        if torch.is_tensor(index):
            index = index.to_list()
        
        clip_imgs = self.clips[index]
        imgs = []
        for img_path in clip_imgs:
            if self.color_mode == 'RGB':
                img = Image.open(img_path.absolute().as_posix()).convert('RGB')
            else:
                img = Image.open(img_path.absolute().as_posix()).convert('L')
            imgs.append(img)
        
        original_clip = self.transform(imgs)

        past_clip = original_clip[0:self.num_observed_frames, ...]
        future_clip = original_clip[-self.num_predict_frames:, ...]
        return past_clip, future_clip

    def visualize_clip(self, clip, file_name):
        """
        save a video clip to GIF file
        Args:
            clip: tensor with shape (clip_length, C, H, W)
        """
        imgs = []
        for i in range(clip.shape[0]):
            img = transforms.ToPILImage()(clip[i, ...])
            imgs.append(img)
        
        videodims = img.size
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')    
        video = cv2.VideoWriter(Path(file_name).absolute().as_posix(), fourcc, 10, videodims)
        for img in imgs:
            video.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        video.release()
        #imgs[0].save(str(Path(file_name).absolute()), save_all = True, append_images = imgs[1:])

class StochasticMovingMNIST(Dataset):
    """https://github.com/edenton/svg/blob/master/data/moving_mnist.py"""
    """Data Handler that creates Bouncing MNIST dataset on the fly."""
    def __init__(self, train_flag, data_root, num_observed_frames, num_predict_frames, transform, num_digits=2, image_size=64, deterministic=False):
        path = data_root
        self.num_observed_frames = num_observed_frames
        self.num_predict_frames = num_predict_frames
        self.seq_len = num_observed_frames + num_predict_frames
        self.transform = transform
        self.num_digits = num_digits  
        self.image_size = image_size 
        self.step_length = 0.1
        self.digit_size = 32
        self.deterministic = deterministic
        self.seed_is_set = False # multi threaded loading
        self.channels = 1 

        self.data = datasets.MNIST(
            path,
            train=train_flag,
            download=False,
            transform=transforms.Compose(
                [transforms.Resize(self.digit_size, antialias=True),
                 transforms.ToTensor()]))

        self.N = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        full_clip = torch.from_numpy(self.__getnparray__(idx))
        imgs = []
        for i in range(full_clip.shape[0]):
            img = transforms.ToPILImage()(full_clip[i, ...])
            imgs.append(img)
        
        full_clip = self.transform(imgs)

        past_clip = full_clip[0:self.num_observed_frames, ...]
        future_clip = full_clip[self.num_observed_frames:, ...]

        return past_clip, future_clip

    def __getnparray__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size, 
                      image_size, 
                      self.channels),
                    dtype=np.float32)
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]

            sx = np.random.randint(image_size-digit_size)
            sy = np.random.randint(image_size-digit_size)
            dx = np.random.randint(-4, 5)
            dy = np.random.randint(-4, 5)
            for t in range(self.seq_len):
                if sy < 0:
                    sy = 0 
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(1, 5)
                        dx = np.random.randint(-4, 5)
                elif sy >= image_size-32:
                    sy = image_size-32-1
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(-4, 0)
                        dx = np.random.randint(-4, 5)
                    
                if sx < 0:
                    sx = 0 
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(1, 5)
                        dy = np.random.randint(-4, 5)
                elif sx >= image_size-32:
                    sx = image_size-32-1
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(-4, 0)
                        dy = np.random.randint(-4, 5)
                   
                x[t, sy:sy+32, sx:sx+32, 0] += digit.numpy().squeeze()
                sy += dy
                sx += dx

        x[x>1] = 1.
        return x.transpose(0, 3, 1, 2)

def svrfcn(batch_data, rand_Tp = 3, rand_predict = True, o_resize = None, p_resize = None, half_fps = False):
    """
    Single video dataset random future frames collate function
    batch_data: list of tuples, each tuple is (observe_clip, predict_clip) or 
                (observe_clip, predict_clip, observe_mask, predict_mask) for range images
    """
    
    # Check if masks are provided (for range images)
    has_masks = len(batch_data[0]) == 4
    
    if has_masks:
        observe_clips, predict_clips, observe_masks, predict_masks = zip(*batch_data)
        observe_batch = torch.stack(observe_clips, dim=0)
        predict_batch = torch.stack(predict_clips, dim=0)
        observe_mask_batch = torch.stack(observe_masks, dim=0)  # (N, To, H, W)
        predict_mask_batch = torch.stack(predict_masks, dim=0)  # (N, Tp, H, W)
    else:
        observe_clips, predict_clips = zip(*batch_data)
        observe_batch = torch.stack(observe_clips, dim=0)
        predict_batch = torch.stack(predict_clips, dim=0)
        observe_mask_batch = None
        predict_mask_batch = None

    #output the last frame of observation, taken as the first frame of autoregressive prediction
    observe_last_batch = observe_batch[:, -1:, ...]
    
    max_Tp = predict_batch.shape[1]
    if rand_predict:
        assert rand_Tp <= max_Tp, "Invalid rand_Tp"
        rand_idx = np.sort(np.random.choice(max_Tp, rand_Tp, replace=False))
        rand_idx = torch.from_numpy(rand_idx)
        rand_predict_batch = predict_batch[:, rand_idx, ...]
        # Apply same random sampling to masks if they exist
        if has_masks:
            rand_predict_mask_batch = predict_mask_batch[:, rand_idx.long(), ...]  # (N, rand_Tp, H, W)
        else:
            rand_predict_mask_batch = None
    else:
        rand_idx = torch.linspace(0, max_Tp-1, max_Tp, dtype = torch.int)
        rand_predict_batch = predict_batch
        if has_masks:
            rand_predict_mask_batch = predict_mask_batch
        else:
            rand_predict_mask_batch = None
    
    To = observe_batch.shape[1]
    idx_o = torch.linspace(0, To-1 , To, dtype = torch.int)
    
    if has_masks:
        observe_last_mask_batch = observe_mask_batch[:, -1:, ...]  # (N, 1, H, W)
    else:
        observe_last_mask_batch = None

    if half_fps:
        if observe_batch.shape[1] > 2:
            observe_batch = observe_batch[:, ::2, ...]
            idx_o = idx_o[::2, ...]
            if has_masks:
                observe_mask_batch = observe_mask_batch[:, ::2, ...]
                observe_last_mask_batch = observe_mask_batch[:, -1:, ...]

        rand_predict_batch = rand_predict_batch[:, ::2, ...]
        rand_idx = rand_idx[::2, ...]
        observe_last_batch = observe_batch[:, -1:, ...]
        if has_masks:
            rand_predict_mask_batch = rand_predict_mask_batch[:, ::2, ...]

    if p_resize is not None:
        N, T, _, _, _ = rand_predict_batch.shape
        rand_predict_batch = p_resize(rand_predict_batch.flatten(0, 1))
        rand_predict_batch = rearrange(rand_predict_batch, "(N T) C H W -> N T C H W", N = N, T=T)
        #als resize the last frame of observation
        observe_last_batch = p_resize(observe_last_batch.flatten(0, 1))
        observe_last_batch = rearrange(observe_last_batch, "(N T) C H W -> N T C H W", N = N, T=1)
        # Resize masks if they exist
        if has_masks:
            N_mask, T_mask, H_mask, W_mask = rand_predict_mask_batch.shape
            rand_predict_mask_batch = torch.nn.functional.interpolate(
                rand_predict_mask_batch.flatten(0, 1).float().unsqueeze(1),
                size=(rand_predict_batch.shape[3], rand_predict_batch.shape[4]),
                mode='nearest'
            ).squeeze(1).bool()
            rand_predict_mask_batch = rearrange(rand_predict_mask_batch, "(N T) H W -> N T H W", N = N_mask, T=T_mask)
            
            observe_last_mask_batch = torch.nn.functional.interpolate(
                observe_last_mask_batch.flatten(0, 1).float().unsqueeze(1),
                size=(rand_predict_batch.shape[3], rand_predict_batch.shape[4]),
                mode='nearest'
            ).squeeze(1).bool()
            observe_last_mask_batch = rearrange(observe_last_mask_batch, "(N T) H W -> N T H W", N = N_mask, T=1)
        
    if o_resize is not None:
        N, T, _, _, _ = observe_batch.shape
        observe_batch = o_resize(observe_batch.flatten(0, 1))
        observe_batch = rearrange(observe_batch, "(N T) C H W -> N T C H W", N = N, T=T)
        # Also resize masks if they exist
        if has_masks:
            N_mask, T_mask, H_mask, W_mask = observe_mask_batch.shape
            observe_mask_batch = torch.nn.functional.interpolate(
                observe_mask_batch.flatten(0, 1).float().unsqueeze(1), 
                size=(observe_batch.shape[3], observe_batch.shape[4]), 
                mode='nearest'
            ).squeeze(1).bool()
            observe_mask_batch = rearrange(observe_mask_batch, "(N T) H W -> N T H W", N = N_mask, T=T_mask)
    
    # Return with or without masks
    if has_masks:
        return (observe_batch, rand_predict_batch, observe_last_batch, idx_o.to(torch.float), rand_idx.to(torch.float) + To, 
                observe_mask_batch, rand_predict_mask_batch, observe_last_mask_batch)
    else:
        return (observe_batch, rand_predict_batch, observe_last_batch, idx_o.to(torch.float), rand_idx.to(torch.float) + To)

#####################################################################################
class VidResize(object):
    def __init__(self, *args, **resize_kwargs):
        self.resize_kwargs = resize_kwargs
        self.resize_kwargs['antialias'] = True
        self.resize_kwargs['interpolation'] = transforms.InterpolationMode.BICUBIC
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.Resize(*self.args, **self.resize_kwargs)(clip[i])

        return clip

class VidCenterCrop(object):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.CenterCrop(*self.args, **self.kwargs)(clip[i])

        return clip

class VidCrop(object):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.functional.crop(clip[i], *self.args, **self.kwargs)

        return clip
        
class VidRandomHorizontalFlip(object):
    def __init__(self, p: float):
        assert p>=0 and p<=1, "invalid flip probability"
        self.p = p
    
    def __call__(self, clip: List[Image.Image]):
        if np.random.rand() < self.p:
            for i in range(len(clip)):
                clip[i] = transforms.functional.hflip(clip[i])
        return clip

class VidRandomVerticalFlip(object):
    def __init__(self, p: float):
        assert p>=0 and p<=1, "invalid flip probability"
        self.p = p
    
    def __call__(self, clip: List[Image.Image]):
        if np.random.rand() < self.p:
            for i in range(len(clip)):
                clip[i] = transforms.functional.vflip(clip[i])
        return clip

class VidToTensor(object):
    def __call__(self, clip: List[Image.Image]):
        """
        Return: clip --- Tensor with shape (T, C, H, W)
        """
        for i in range(len(clip)):
            clip[i] = transforms.ToTensor()(clip[i])
        clip = torch.stack(clip, dim = 0)

        return clip

class NumpyToTensor(object):
    """Convert numpy array directly to tensor (for KITTI_RANGE, preserves original values)"""
    def __call__(self, clip: List[np.ndarray]):
        """
        Args:
            clip: List of numpy arrays, each with shape (H, W)
        Returns:
            Tensor with shape (T, C, H, W) where C=1
        """
        tensors = []
        for arr in clip:
            # Convert numpy to tensor (preserves values exactly)
            tensor = torch.from_numpy(arr.astype(np.float32))
            # Add channel dimension: (H, W) -> (1, H, W)
            tensor = tensor.unsqueeze(0)
            tensors.append(tensor)
        # Stack along time dimension: (T, C, H, W)
        return torch.stack(tensors, dim=0)

class VidResizeTensor(object):
    """Resize tensor directly (for KITTI_RANGE)"""
    def __init__(self, *args, **resize_kwargs):
        self.resize_kwargs = resize_kwargs
        self.resize_kwargs['antialias'] = True
        self.resize_kwargs['interpolation'] = transforms.InterpolationMode.BICUBIC
        self.args = args

    def __call__(self, clip: torch.Tensor):
        """
        Args:
            clip: Tensor with shape (T, C, H, W)
        Returns:
            Resized tensor with shape (T, C, H_new, W_new)
        """
        T, C, H, W = clip.shape
        resized_clip = []
        for t in range(T):
            # Resize each frame: (C, H, W) -> (C, H_new, W_new)
            frame = transforms.Resize(*self.args, **self.resize_kwargs)(clip[t])
            resized_clip.append(frame)
        return torch.stack(resized_clip, dim=0)

class MeanCenterTransform(object):
    """Mean centering transform (subtract mean, no scaling)"""
    def __init__(self, mean):
        self.mean = mean
    
    def __call__(self, clip: torch.Tensor):
        """
        Args:
            clip: Tensor with shape (T, C, H, W)
        Returns:
            Mean-centered tensor: clip - mean
        """
        return clip - self.mean

class VidNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, clip: Tensor):
        """
        Return: clip --- Tensor with shape (T, C, H, W)
        """
        T, _, _, _ = clip.shape
        for i in range(T):
            clip[i, ...] = transforms.Normalize(self.mean, self.std)(clip[i, ...])

        return clip

class VidReNormalize(object):
    def __init__(self, mean, std):
        try:
            self.inv_std = [1.0/s for s in std]
            self.inv_mean = [-m for m in mean]
            self.renorm = transforms.Compose([transforms.Normalize(mean = [0., 0., 0.],
                                                                std = self.inv_std),
                                            transforms.Normalize(mean = self.inv_mean,
                                                                std = [1., 1., 1.])])
        except TypeError:
            #try normalize for grey_scale images.
            self.inv_std = 1.0/std
            self.inv_mean = -mean
            self.renorm = transforms.Compose([transforms.Normalize(mean = 0.,
                                                                std = self.inv_std),
                                            transforms.Normalize(mean = self.inv_mean,
                                                                std = 1.)])

    def __call__(self, clip: Tensor):
        """
        Return: clip --- Tensor with shape (T, C, H, W)
        """
        T, _, _, _ = clip.shape
        for i in range(T):
            clip[i, ...] = self.renorm(clip[i, ...])

        return clip

class VidPad(object):
    """
    If pad, Do not forget to pass the mask to the transformer encoder.
    """
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.Pad(*self.args, **self.kwargs)(clip[i])

        return clip

def mean_std_compute(dataset, device, color_mode = 'RGB'):
    """
    arguments:
        dataset: pytorch dataloader
        device: torch.device('cuda:0') or torch.device('cpu') for computation
    return:
        mean and std of each image channel.
        std = sqrt(E(x^2) - (E(X))^2)
    """
    data_iter= iter(dataset)
    sum_img = None
    square_sum_img = None
    N = 0

    pgbar = tqdm(desc = 'summarizing...', total = len(dataset))
    for idx, sample in enumerate(data_iter):
        past, future = sample
        clip = torch.cat([past, future], dim = 0)
        N += clip.shape[0]

        img = torch.sum(clip, axis = 0)

        if idx == 0:
            sum_img = img
            square_sum_img = torch.square(img)
            sum_img = sum_img.to(torch.device(device))
            square_sum_img = square_sum_img.to(torch.device(device))
        else:
            img = img.to(device)
            sum_img = sum_img + img
            square_sum_img = square_sum_img + torch.square(img)
        
        pgbar.update(1)
    
    pgbar.close()

    mean_img = sum_img/N
    mean_square_img = square_sum_img/N
    if color_mode == 'RGB':
        mean_r, mean_g, mean_b = torch.mean(mean_img[0, :, :]), torch.mean(mean_img[1, :, :]), torch.mean(mean_img[2, :, :])
        mean_r2, mean_g2, mean_b2 = torch.mean(mean_square_img[0,:,:]), torch.mean(mean_square_img[1,:,:]), torch.mean(mean_square_img[2,:,:])
        std_r, std_g, std_b = torch.sqrt(mean_r2 - torch.square(mean_r)), torch.sqrt(mean_g2 - torch.square(mean_g)), torch.sqrt(mean_b2 - torch.square(mean_b))

        return ([mean_r.cpu().numpy(), mean_g.data.cpu().numpy(), mean_b.cpu().numpy()], [std_r.cpu().numpy(), std_g.cpu().numpy(), std_b.cpu().numpy()])
    else:
        mean = torch.mean(mean_img)
        mean_2 = torch.mean(mean_square_img)
        std = torch.sqrt(mean_2 - torch.square(mean))

        return (mean.cpu().numpy(), std.cpu().numpy())

def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.Dataset:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x

def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.Dataset.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x

class RangeImageClipDataset(Dataset):
    """
    Video clips dataset for range images stored as .npy files
    """
    def __init__(self, num_observed_frames, num_predict_frames, clips, transform, color_mode, global_min=None, global_max=None):
        """
        Args:
            num_observed_frames --- number of past frames
            num_predict_frames --- number of future frames
            clips --- List of video clips frames file path (.npy files)
            transform --- torchvision transforms for the image
            color_mode --- 'grey_scale' for range images (single channel)
            global_min --- Global minimum value for normalization (if None, uses per-image min)
            global_max --- Global maximum value for normalization (if None, uses per-image max)

        Return batched Sample:
            past_clip --- Tensor with shape (batch_size, num_observed_frames, C, H, W)
            future_clip --- Tensor with shape (batch_size, num_predict_frames, C, H, W)
        """
        self.num_observed_frames = num_observed_frames
        self.num_predict_frames = num_predict_frames
        self.clips = clips
        self.transform = transform
        if color_mode != 'grey_scale':
            raise ValueError("Range images must use 'grey_scale' color mode!")
        self.color_mode = color_mode
        self.global_min = global_min
        self.global_max = global_max

    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, index: int):
        """
        Returns:
            past_clip: Tensor with shape (num_observed_frames, C, H, W)
            future_clip: Tensor with shape (num_predict_frames, C, H, W)
        """
        if torch.is_tensor(index):
            index = index.to_list()
        
        clip_files = self.clips[index]
        range_imgs = []  # List of PIL Images
        valid_masks_raw = []  # Store original valid masks before transforms
        
        for npy_path in clip_files:
            # Load .npy file
            range_img = np.load(npy_path.absolute().as_posix())
            
            # Ensure 2D array (H, W)
            if len(range_img.shape) > 2:
                range_img = range_img.squeeze()
            
            # Ensure float32 dtype
            range_img = range_img.astype(np.float32)
            
            # Handle invalid pixels (typically -1 or negative values)
            # Store valid mask BEFORE processing (for loss masking)
            valid_mask = range_img > 0
            valid_masks_raw.append(valid_mask.copy())
            
            # Set invalid pixels to 0 (keep original values for valid pixels)
            range_img[~valid_mask] = 0.0
            
            # Global min-max normalization: normalize to [0, 1] using global min/max
            if self.global_min is not None and self.global_max is not None and self.global_max > self.global_min:
                # Use global min/max for normalization
                range_img[valid_mask] = (range_img[valid_mask] - self.global_min) / (self.global_max - self.global_min)
            else:
                # Fallback to per-image normalization if global min/max not provided
                img_min = range_img[valid_mask].min() if valid_mask.any() else 0.0
                img_max = range_img[valid_mask].max() if valid_mask.any() else 1.0
                if img_max > img_min:
                    range_img[valid_mask] = (range_img[valid_mask] - img_min) / (img_max - img_min)
            
            # Convert to PIL Image (expects uint8 in [0, 255] range)
            # Scale from [0, 1] to [0, 255] and convert to uint8
            range_img_uint8 = (range_img * 255.0).astype(np.uint8)
            pil_img = Image.fromarray(range_img_uint8, mode='L')
            range_imgs.append(pil_img)
        
        # Apply transforms (VidResize -> VidToTensor -> norm_transform)
        original_clip = self.transform(range_imgs)  # (T, C, H, W)
        
        # Resize valid masks to match transformed image size
        # After transforms, images are (T, C, H, W)
        T, C, H, W = original_clip.shape
        valid_masks_tensor = torch.zeros((T, H, W), dtype=torch.bool)
        
        for i, mask in enumerate(valid_masks_raw):
            # Get original image dimensions
            orig_h, orig_w = mask.shape
            # Resize mask to match transformed image size using nearest neighbor
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H_orig, W_orig)
            if orig_h != H or orig_w != W:
                mask_resized = torch.nn.functional.interpolate(
                    mask_tensor, size=(H, W), mode='nearest'
                ).squeeze()  # (H, W)
            else:
                mask_resized = mask_tensor.squeeze()  # (H, W)
            valid_masks_tensor[i] = mask_resized.bool()

        past_clip = original_clip[0:self.num_observed_frames, ...]
        future_clip = original_clip[-self.num_predict_frames:, ...]
        past_valid_mask = valid_masks_tensor[0:self.num_observed_frames, ...]  # (To, H, W)
        future_valid_mask = valid_masks_tensor[-self.num_predict_frames:, ...]  # (Tp, H, W)
        
        return past_clip, future_clip, past_valid_mask, future_valid_mask


class KITTIRangeImageDataset(object):
    """
    KITTI Range Image dataset, a wrapper for RangeImageClipDataset
    Range images are stored as .npy files in sequence folders
    Structure: KITTI_dir/{sequence_id}/processed/range/*.npy
    """
    def __init__(self, KITTI_dir, test_folder_ids, transform, train, val,
                 num_observed_frames, num_predict_frames, global_min=None, global_max=None):
        """
        Args:
            KITTI_dir --- Directory for KITTI range images (e.g., /scratch/pydah/kitti/processed_data)
            test_folder_ids --- List of folder indices to use for testing (e.g., [10, 11, 12, 13])
            train --- True for training dataset, False for test dataset
            val --- True if validation split is needed
            transform --- torchvision transform functions
            num_observed_frames --- number of past frames
            num_predict_frames --- number of future frames
            global_min --- Global minimum value for normalization (if None, uses per-image min)
            global_max --- Global maximum value for normalization (if None, uses per-image max)
        """
        self.num_observed_frames = num_observed_frames
        self.num_predict_frames = num_predict_frames
        self.clip_length = num_observed_frames + num_predict_frames
        self.transform = transform
        self.color_mode = 'grey_scale'  # Range images are grayscale
        self.global_min = global_min
        self.global_max = global_max

        self.KITTI_path = Path(KITTI_dir).absolute()
        self.train = train
        self.val = val

        # Convert test_folder_ids to integers in case they're strings (e.g., from YAML with leading zeros)
        test_folder_ids = [int(i) for i in test_folder_ids]

        # Get all sequence folders (00, 01, 02, etc.)
        self.all_folders = sorted([f for f in os.listdir(self.KITTI_path) 
                                   if os.path.isdir(self.KITTI_path / f) and f.isdigit()])
        self.num_examples = len(self.all_folders)
        
        self.folder_id = list(range(self.num_examples))
        if self.train:
            # Get all folders except test folders
            self.train_folders = [self.all_folders[i] for i in range(self.num_examples) if i not in test_folder_ids]
            # Only split for validation if val=True AND we have enough sequences
            if self.val and len(self.train_folders) > 2:
                # Use first 2 sequences for validation, rest for training
                self.val_folders = self.train_folders[0:2]
                self.train_folders = self.train_folders[2:]
            elif self.val:
                # If we have 2 or fewer sequences, use all for training, none for validation
                self.val_folders = []
                # Keep all train_folders for training
        else:
            self.test_folders = [self.all_folders[i] for i in test_folder_ids]
        
        if self.train:
            self.train_clips = self.__getClips__(self.train_folders)
            if self.val and len(self.val_folders) > 0:
                self.val_clips = self.__getClips__(self.val_folders)
            else:
                self.val_clips = []
        else:
            self.test_clips = self.__getClips__(self.test_folders)

    def __call__(self):
        """
        Returns:
            clip_set --- RangeImageClipDataset object
        """
        if self.train:
            clip_set = RangeImageClipDataset(self.num_observed_frames, self.num_predict_frames, 
                                            self.train_clips, self.transform, self.color_mode,
                                            global_min=self.global_min, global_max=self.global_max)
            if self.val and len(self.val_clips) > 0:
                val_clip_set = RangeImageClipDataset(self.num_observed_frames, self.num_predict_frames, 
                                                    self.val_clips, self.transform, self.color_mode,
                                                    global_min=self.global_min, global_max=self.global_max)
                return clip_set, val_clip_set
            return clip_set
        else:
            return RangeImageClipDataset(self.num_observed_frames, self.num_predict_frames, 
                                        self.test_clips, self.transform, self.color_mode,
                                        global_min=self.global_min, global_max=self.global_max)
    
    def __getClips__(self, frame_folders):
        """
        Get clips from sequence folders
        Structure: {sequence_id}/processed/range/*.npy
        """
        clips = []
        for folder in frame_folders:
            # Path to range images: {sequence_id}/processed/range/
            range_dir = self.KITTI_path.joinpath(folder, "processed", "range")
            
            if not range_dir.exists():
                print(f"Warning: Range directory not found: {range_dir}")
                continue
            
            # Get all .npy files and sort them
            npy_files = sorted(list(range_dir.glob("*.npy")))
            
            if len(npy_files) == 0:
                print(f"Warning: No .npy files found in {range_dir}")
                continue
            
            # Calculate number of clips
            clip_num = len(npy_files) // self.clip_length
            rem_num = len(npy_files) % self.clip_length
            
            # Remove remainder frames from start (centered)
            npy_files = npy_files[rem_num // 2 : rem_num//2 + clip_num*self.clip_length]
            
            # Create clips
            for i in range(clip_num):
                clips.append(npy_files[i*self.clip_length : (i+1)*self.clip_length])

        return clips


def visualize_batch_clips(gt_past_frames_batch, gt_future_frames_batch, pred_frames_batch, file_dir, renorm_transform = None, desc = None,
                          pred_masks_batch=None, gt_future_masks_batch=None, gt_past_masks_batch=None):
    """
        pred_frames_batch: tensor with shape (N, future_clip_length, C, H, W)
        gt_future_frames_batch: tensor with shape (N, future_clip_length, C, H, W)
        gt_past_frames_batch: tensor with shape (N, past_clip_length, C, H, W)
        pred_masks_batch: optional tensor with shape (N, future_clip_length, 1, H, W) - raw mask values
        gt_future_masks_batch: optional tensor with shape (N, future_clip_length, 1, H, W) - binary masks
        gt_past_masks_batch: optional tensor with shape (N, past_clip_length, 1, H, W) - binary masks
    """
    if not Path(file_dir).exists():
        Path(file_dir).mkdir(parents=True, exist_ok=True) 
    def save_clip(clip, file_name):
        imgs = []
        if renorm_transform is not None:
            clip = renorm_transform(clip)
            clip = torch.clamp(clip, min = 0., max = 1.0)
        for i in range(clip.shape[0]):
            img = transforms.ToPILImage()(clip[i, ...])
            imgs.append(img)

        imgs[0].save(str(Path(file_name).absolute()), save_all = True, append_images = imgs[1:], loop = 0)
    
    def append_frames(batch, max_clip_length):
        d = max_clip_length - batch.shape[1]
        if d > 0:
            batch = torch.cat([batch, batch[:, -1:, :, :, :].repeat(1, d, 1, 1, 1)], dim = 1)
        return batch
    
    max_length = max(gt_future_frames_batch.shape[1], gt_past_frames_batch.shape[1])
    max_length = max(max_length, pred_frames_batch.shape[1])
    if gt_past_frames_batch.shape[1] < max_length:
        gt_past_frames_batch = append_frames(gt_past_frames_batch, max_length)
    if gt_future_frames_batch.shape[1] < max_length:
        gt_future_frames_batch = append_frames(gt_future_frames_batch, max_length)
    if pred_frames_batch.shape[1] < max_length:    
        pred_frames_batch = append_frames(pred_frames_batch, max_length)

    # Prepare image batch (horizontal concatenation: past | future GT | future pred)
    image_batch = torch.cat([gt_past_frames_batch, gt_future_frames_batch, pred_frames_batch], dim = -1) #shape (N, clip_length, C, H, 3W)
    image_batch = image_batch.cpu()
    
    # Prepare masks if provided
    mask_batch_3ch = None
    if pred_masks_batch is not None or gt_future_masks_batch is not None or gt_past_masks_batch is not None:
        # Prepare mask batches
        mask_batches = []
        
        if gt_past_masks_batch is not None:
            # Convert to float if boolean: (N, T, 1, H, W)
            if gt_past_masks_batch.dtype == torch.bool:
                gt_past_masks_batch = gt_past_masks_batch.float()
            mask_batches.append(gt_past_masks_batch)
        if gt_future_masks_batch is not None:
            # Convert to float if boolean: (N, T, 1, H, W)
            if gt_future_masks_batch.dtype == torch.bool:
                gt_future_masks_batch = gt_future_masks_batch.float()
            mask_batches.append(gt_future_masks_batch)
        if pred_masks_batch is not None:
            # pred_masks_batch is in [0, 1] range (already normalized in test script from raw [-1, 1])
            # Convert to binary by thresholding at 0.5: values > 0.5 become 1.0, values <= 0.5 become 0.0
            # Ensure it's float type
            if pred_masks_batch.dtype == torch.bool:
                pred_masks_batch = pred_masks_batch.float()
            pred_masks_binary = (pred_masks_batch > 0.5).float()
            mask_batches.append(pred_masks_binary)
        
        if len(mask_batches) > 0:
            # Find max length for masks
            max_mask_length = max([mb.shape[1] for mb in mask_batches] + [max_length])
            
            # Append frames to match max length
            mask_batches_padded = []
            for mb in mask_batches:
                if mb.shape[1] < max_mask_length:
                    mb = append_frames(mb, max_mask_length)
                mask_batches_padded.append(mb)
            
            # Concatenate masks horizontally (past masks | future GT masks | future pred masks)
            mask_batch = torch.cat(mask_batches_padded, dim = -1)  # shape (N, clip_length, 1, H, num_masks*W)
            mask_batch = mask_batch.cpu()
            
            # Ensure mask_batch is float type for interpolation
            if mask_batch.dtype != torch.float32:
                mask_batch = mask_batch.float()
            
            # Convert single channel mask to 3-channel for visualization (grayscale)
            # Repeat channel dimension: (N, T, 1, H, W) -> (N, T, 3, H, W)
            mask_batch_3ch = mask_batch.repeat(1, 1, 3, 1, 1)  # (N, clip_length, 3, H, num_masks*W)
    
    # Create combined GIFs with images on top and masks below (vertical concatenation)
    N = image_batch.shape[0]
    for n in range(N):
        if mask_batch_3ch is not None:
            # Get image clip: (clip_length, C, H, 3W)
            image_clip = image_batch[n, ...]
            # Get mask clip: (clip_length, 3, H, num_masks*W)
            mask_clip = mask_batch_3ch[n, ...]
            
            # Ensure same width for vertical concatenation
            img_width = image_clip.shape[-1]  # 3W
            mask_width = mask_clip.shape[-1]  # num_masks*W
            
            if mask_width != img_width:
                # Resize mask to match image width
                # mask_clip: (T, 3, H, W) -> need to resize width dimension
                T, C, H, W = mask_clip.shape
                # Ensure mask_clip is float for interpolation
                if mask_clip.dtype != torch.float32:
                    mask_clip = mask_clip.float()
                mask_clip_resized = torch.nn.functional.interpolate(
                    mask_clip,  # (T, 3, H, W)
                    size=(H, img_width),
                    mode='bilinear',
                    align_corners=False
                )  # (T, 3, H, img_width)
                mask_clip = mask_clip_resized
            
            # Concatenate vertically: images on top, masks below
            # image_clip: (T, C, H, W), mask_clip: (T, 3, H, W)
            # Result: (T, 3, 2H, W) - images on top, masks below
            if image_clip.shape[1] == 1:
                # Convert grayscale image to RGB
                image_clip = image_clip.repeat(1, 3, 1, 1)  # (T, 3, H, W)
            
            # Concatenate along height dimension (dim=-2): images on top, masks below
            combined_clip = torch.cat([image_clip, mask_clip], dim = -2)  # (T, 3, 2H, W)
        else:
            # No masks, just use images
            combined_clip = image_batch[n, ...]
            if combined_clip.shape[1] == 1:
                # Convert grayscale to RGB
                combined_clip = combined_clip.repeat(1, 3, 1, 1)
        
        file_name = file_dir.joinpath(f'{desc}_clip_{n}.gif' if desc else f'clip_{n}.gif')
        save_clip(combined_clip, file_name)