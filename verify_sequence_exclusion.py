"""
Script to verify if sequence 09 is actually excluded from training.
This simulates the exact logic from the dataset code.
"""
import os
from pathlib import Path

# Configuration
KITTI_DIR = '/DATA/common/kitti/processed_data'
TEST_FOLDER_IDS = [9]  # From config

# Simulate the exact logic from dataset.py lines 1100-1108
kitti_path = Path(KITTI_DIR)

# Get all sequence folders (00, 01, 02, etc.) - EXACT logic from line 1101-1102
all_folders = sorted([f for f in os.listdir(kitti_path) 
                     if os.path.isdir(kitti_path / f) and f.isdigit()])
num_examples = len(all_folders)

print("=" * 80)
print("VERIFICATION: Is Sequence 09 Excluded from Training?")
print("=" * 80)

print(f"\n1. All folders found (sorted):")
for idx, folder in enumerate(all_folders):
    marker = " <-- INDEX 9" if idx == 9 else ""
    print(f"   Index {idx:2d}: '{folder}'{marker}")

print(f"\n2. test_folder_ids from config: {TEST_FOLDER_IDS}")

print(f"\n3. Applying training folder logic (line 1108):")
print(f"   train_folders = [all_folders[i] for i in range({num_examples}) if i not in {TEST_FOLDER_IDS}]")
print(f"\n   Iterating through indices:")

train_folders = []
for i in range(num_examples):
    folder = all_folders[i]
    is_excluded = i in TEST_FOLDER_IDS
    status = "EXCLUDED" if is_excluded else "INCLUDED"
    train_folders.append(folder) if not is_excluded else None
    print(f"   Index {i:2d} ('{folder}'): {status}")

print(f"\n4. Final train_folders list:")
for folder in train_folders:
    print(f"   - Sequence {folder}")

print(f"\n5. Verification:")
if '09' in train_folders:
    print("   ❌ ERROR: Sequence '09' IS in train_folders! It should be excluded!")
else:
    print("   ✅ CORRECT: Sequence '09' is NOT in train_folders. It is properly excluded.")

print(f"\n6. Test folders (line 1119):")
test_folders = [all_folders[i] for i in TEST_FOLDER_IDS]
for folder in test_folders:
    print(f"   - Sequence {folder}")

print(f"\n{'='*80}")
print("CONCLUSION:")
print(f"{'='*80}")
if '09' in train_folders:
    print("Sequence 09 IS being used for training (BUG in code or config)")
else:
    print("Sequence 09 is CORRECTLY excluded from training")
    print(f"Training uses {len(train_folders)} sequences: {', '.join(train_folders)}")
    print(f"Testing uses {len(test_folders)} sequence(s): {', '.join(test_folders)}")

