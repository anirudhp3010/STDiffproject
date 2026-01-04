#!/bin/bash
# Script to clean up old checkpoints and free disk space

CHECKPOINT_DIR="/home/anirudh/STDiffProject/STDiff_ckpts/kitti_range_64x512"

echo "Current disk usage:"
df -h /home/anirudh | grep -E "Filesystem|/dev/"

echo ""
echo "Checkpoint directory size:"
du -sh "$CHECKPOINT_DIR" 2>/dev/null || echo "Directory not found"

echo ""
echo "Listing all checkpoints:"
ls -lh "$CHECKPOINT_DIR"/checkpoint-* 2>/dev/null | awk '{print $9, $5}'

echo ""
echo "Total checkpoints:"
find "$CHECKPOINT_DIR" -name "checkpoint-*" -type d | wc -l

echo ""
read -p "Do you want to keep only the latest 3 checkpoints? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Keeping only latest 3 checkpoints..."
    # Get all checkpoint directories, sort by modification time, keep last 3
    cd "$CHECKPOINT_DIR"
    ls -dt checkpoint-* 2>/dev/null | tail -n +4 | xargs -r rm -rf
    echo "Cleanup complete!"
    echo ""
    echo "Remaining checkpoints:"
    ls -lh checkpoint-* 2>/dev/null
    echo ""
    echo "New disk usage:"
    df -h /home/anirudh | grep -E "Filesystem|/dev/"
fi

