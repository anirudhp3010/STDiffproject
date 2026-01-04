"""
Script to analyze which sequences are used for training.
"""
from pathlib import Path
import os

# Configuration from your training config
KITTI_DIR = '/DATA/common/kitti/processed_data'
TEST_FOLDER_IDS = [9]  # From kitti_range_train_config.yaml
PHASE = 'deploy'  # From kitti_range_train_config.yaml
NUM_OBSERVED_FRAMES = 3
NUM_PREDICT_FRAMES = 3

def analyze_sequences():
    """Analyze which sequences are used for training and testing."""
    kitti_path = Path(KITTI_DIR)
    
    if not kitti_path.exists():
        print(f"Error: KITTI directory not found: {KITTI_DIR}")
        return
    
    # Get all sequence folders (00, 01, 02, etc.)
    all_folders = sorted([f for f in os.listdir(kitti_path) 
                         if os.path.isdir(kitti_path / f) and f.isdigit()])
    
    print("=" * 80)
    print("KITTI RANGE DATASET - SEQUENCE ANALYSIS")
    print("=" * 80)
    print(f"\nTotal sequences found: {len(all_folders)}")
    print(f"Sequences: {', '.join(all_folders)}")
    
    # Determine train/test split
    num_examples = len(all_folders)
    train_folders = [all_folders[i] for i in range(num_examples) if i not in TEST_FOLDER_IDS]
    test_folders = [all_folders[i] for i in TEST_FOLDER_IDS]
    
    print(f"\n{'='*80}")
    print("SEQUENCE SPLIT")
    print(f"{'='*80}")
    print(f"\nTest sequences (folder IDs {TEST_FOLDER_IDS}):")
    for folder in test_folders:
        print(f"  - Sequence {folder}")
    
    print(f"\nTraining sequences (all except test):")
    for folder in train_folders:
        print(f"  - Sequence {folder}")
    
    # Check validation split
    use_val = PHASE != 'deploy'
    if use_val and len(train_folders) > 2:
        val_folders = train_folders[0:2]
        actual_train_folders = train_folders[2:]
        print(f"\nValidation sequences (first 2 from training set):")
        for folder in val_folders:
            print(f"  - Sequence {folder}")
        print(f"\nActual training sequences (after validation split):")
        for folder in actual_train_folders:
            print(f"  - Sequence {folder}")
    else:
        print(f"\nNo validation split (phase='{PHASE}')")
        print(f"All {len(train_folders)} training sequences used for training")
    
    # Analyze clips per sequence
    print(f"\n{'='*80}")
    print("CLIPS PER SEQUENCE")
    print(f"{'='*80}")
    clip_length = NUM_OBSERVED_FRAMES + NUM_PREDICT_FRAMES
    print(f"\nClip length: {clip_length} frames ({NUM_OBSERVED_FRAMES} observed + {NUM_PREDICT_FRAMES} predicted)")
    
    total_clips = 0
    for folder in train_folders:
        range_dir = kitti_path.joinpath(folder, "processed", "range")
        if range_dir.exists():
            npy_files = sorted(list(range_dir.glob("*.npy")))
            clip_num = len(npy_files) // clip_length
            rem_num = len(npy_files) % clip_length
            total_clips += clip_num
            print(f"\nSequence {folder}:")
            print(f"  - Total frames: {len(npy_files)}")
            print(f"  - Clips: {clip_num} (each {clip_length} frames)")
            print(f"  - Remaining frames: {rem_num} (discarded)")
        else:
            print(f"\nSequence {folder}: Range directory not found!")
    
    print(f"\n{'='*80}")
    print(f"TOTAL TRAINING CLIPS: {total_clips}")
    print(f"{'='*80}")
    
    # Explain how clips are created
    print(f"\n{'='*80}")
    print("HOW CLIPS ARE CREATED")
    print(f"{'='*80}")
    print(f"""
1. For each sequence folder (e.g., '00', '01', etc.):
   - Load all .npy files from: {KITTI_DIR}/{{sequence_id}}/processed/range/*.npy
   - Sort files by name (temporal order)

2. Create clips:
   - Each clip contains {clip_length} consecutive frames
   - First {NUM_OBSERVED_FRAMES} frames: observed (input to model)
   - Last {NUM_PREDICT_FRAMES} frames: predicted (ground truth for training)
   
3. Example for a sequence with 100 frames:
   - Clips: {100 // clip_length} clips
   - Remaining: {100 % clip_length} frames (discarded from center)
   - Each clip: frames [0-{clip_length-1}], [{clip_length}-{clip_length*2-1}], etc.

4. During training:
   - Model sees {NUM_OBSERVED_FRAMES} frames (Vo)
   - Model predicts {NUM_PREDICT_FRAMES} frames (Vp)
   - Loss is computed between predictions and ground truth future frames
    """)


if __name__ == "__main__":
    analyze_sequences()

