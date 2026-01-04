# How to Run Training on a Server

This guide explains how to run your `train.sh` script on a server.

---

## Quick Start

### 1. **Make the script executable** (if not already done)
```bash
chmod +x train.sh
```

### 2. **Create logs directory** (if it doesn't exist)
```bash
mkdir -p stdiff/logs
```

### 3. **Run the training script**
```bash
./train.sh
```

Or directly:
```bash
nohup python3 -u ./stdiff/train_stdiff.py --train_config ./stdiff/configs/kitti_range_train_config.yaml > ./stdiff/logs/train.log 2>&1 &
```

---

## Understanding Your Script

Your `train.sh` contains:
```bash
nohup python3 -u ./stdiff/train_stdiff.py --train_config ./stdiff/configs/kitti_range_train_config.yaml > ./stdiff/logs/train.log 2>&1 &
```

**What each part does:**
- `nohup` - Runs command immune to hangups (continues if you disconnect)
- `python3 -u` - Runs Python with unbuffered output (see logs in real-time)
- `./stdiff/train_stdiff.py` - Your training script
- `--train_config ...` - Path to config file
- `> ./stdiff/logs/train.log` - Redirects stdout to log file
- `2>&1` - Redirects stderr to stdout (captures errors too)
- `&` - Runs in background

---

## Step-by-Step Instructions

### Step 1: Connect to Server
```bash
ssh username@server_address
cd /path/to/STDiffProject
```

### Step 2: Activate Conda Environment
```bash
conda activate stdiff
# or
source activate stdiff
```

### Step 3: Verify Setup
```bash
# Check Python version
python3 --version

# Check if CUDA is available
python3 -c "import torch; print(torch.cuda.is_available())"

# Check GPU
nvidia-smi
```

### Step 4: Create Logs Directory
```bash
mkdir -p stdiff/logs
```

### Step 5: Run Training
```bash
# Option 1: Using the script
./train.sh

# Option 2: Direct command
nohup python3 -u ./stdiff/train_stdiff.py --train_config ./stdiff/configs/kitti_range_train_config.yaml > ./stdiff/logs/train.log 2>&1 &
```

### Step 6: Note the Process ID
After running, you'll see output like:
```
[1] 12345
```
This is the process ID (PID). **Save this number!**

---

## Monitoring Training

### 1. **View Live Logs**
```bash
# Watch logs in real-time
tail -f stdiff/logs/train.log

# View last 100 lines
tail -n 100 stdiff/logs/train.log

# View entire log
cat stdiff/logs/train.log
```

### 2. **Check if Process is Running**
```bash
# Find the process
ps aux | grep train_stdiff.py

# Or using the PID
ps -p 12345  # Replace 12345 with your PID
```

### 3. **Monitor GPU Usage**
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Or check once
nvidia-smi
```

### 4. **Check Training Progress**
```bash
# Look for checkpoint saves in logs
grep "checkpoint" stdiff/logs/train.log

# Look for loss values
grep "loss" stdiff/logs/train.log | tail -20
```

---

## Managing the Training Process

### Stop Training
```bash
# Option 1: Using PID
kill 12345  # Replace with your PID

# Option 2: Find and kill
pkill -f train_stdiff.py

# Option 3: Force kill (if needed)
kill -9 12345
```

### Pause Training (Suspend)
```bash
# Suspend process
kill -STOP 12345

# Resume process
kill -CONT 12345
```

### Check if Training Completed
```bash
# Check if process is still running
ps -p 12345

# If process doesn't exist, check exit status
echo $?  # 0 = success, non-zero = error
```

---

## Using Screen or TMUX (Recommended)

For long-running training, use `screen` or `tmux` to keep the session alive:

### Using Screen
```bash
# Start a screen session
screen -S training

# Run training
./train.sh

# Detach: Press Ctrl+A, then D
# Reattach later:
screen -r training
```

### Using TMUX
```bash
# Start a tmux session
tmux new -s training

# Run training
./train.sh

# Detach: Press Ctrl+B, then D
# Reattach later:
tmux attach -t training
```

---

## Alternative: Using Accelerate (Multi-GPU)

If you have multiple GPUs, you can use Accelerate:

### 1. Configure Accelerate
```bash
accelerate config
```

### 2. Run with Accelerate
```bash
nohup accelerate launch ./stdiff/train_stdiff.py --train_config ./stdiff/configs/kitti_range_train_config.yaml > ./stdiff/logs/train.log 2>&1 &
```

---

## Troubleshooting

### Problem: "No such file or directory"
**Solution:** Make sure you're in the correct directory:
```bash
cd /home/anirudh/STDiffProject
pwd  # Verify you're in the right place
```

### Problem: "Permission denied"
**Solution:** Make script executable:
```bash
chmod +x train.sh
```

### Problem: "Module not found"
**Solution:** Activate conda environment:
```bash
conda activate stdiff
```

### Problem: "CUDA out of memory"
**Solution:** Reduce batch size in config file:
```yaml
Dataset:
    batch_size: 1  # Reduce this
```

### Problem: Training stops when you disconnect
**Solution:** Use `nohup` (already in your script) or `screen`/`tmux`

### Problem: Can't find logs
**Solution:** Check if logs directory exists:
```bash
ls -la stdiff/logs/
```

---

## Checking Training Status

### View Recent Logs
```bash
tail -f stdiff/logs/train.log
```

### Search for Errors
```bash
grep -i error stdiff/logs/train.log
grep -i exception stdiff/logs/train.log
```

### Check Checkpoint Saves
```bash
# Check if checkpoints are being saved
ls -lh STDiff_ckpts/kitti_range_64x512/

# Check latest checkpoint
ls -lt STDiff_ckpts/kitti_range_64x512/checkpoint-* | head -1
```

### Monitor Loss
```bash
# Extract loss values
grep "loss" stdiff/logs/train.log | tail -20
```

---

## Best Practices

1. **Always use `nohup` or `screen`/`tmux`** for long training runs
2. **Monitor logs regularly** to catch errors early
3. **Check GPU usage** to ensure training is actually running
4. **Save the PID** so you can manage the process
5. **Check disk space** before starting (training can use lots of space)
6. **Verify data paths** in config file are correct on server

---

## Example Workflow

```bash
# 1. Connect to server
ssh user@server

# 2. Navigate to project
cd /home/anirudh/STDiffProject

# 3. Activate environment
conda activate stdiff

# 4. Check GPU
nvidia-smi

# 5. Create logs directory
mkdir -p stdiff/logs

# 6. Start training
./train.sh

# 7. Note PID (e.g., [1] 12345)

# 8. Monitor logs
tail -f stdiff/logs/train.log

# 9. In another terminal, check GPU
watch -n 1 nvidia-smi

# 10. When done, check final status
tail -n 50 stdiff/logs/train.log
```

---

## Quick Reference Commands

```bash
# Start training
./train.sh

# View logs
tail -f stdiff/logs/train.log

# Check if running
ps aux | grep train_stdiff

# Stop training
pkill -f train_stdiff.py

# Check GPU
nvidia-smi

# Check disk space
df -h

# Check checkpoints
ls -lh STDiff_ckpts/kitti_range_64x512/
```

---

## Summary

Your `train.sh` script is already well-configured with `nohup` for background execution. Simply:

1. Make it executable: `chmod +x train.sh`
2. Create logs directory: `mkdir -p stdiff/logs`
3. Run it: `./train.sh`
4. Monitor: `tail -f stdiff/logs/train.log`

The training will continue even if you disconnect from the server!

