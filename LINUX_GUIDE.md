# 🐧 Advanced Linux Guide for Deep-MRIC

This guide provides advanced tips and workflows for running the Deep-MRIC models on Linux systems, especially for headless servers, SSH connections, and long-running training jobs.

---

## 1. Run Jupyter Notebooks Without GUI (Headless/Server Mode)

If you're on a Linux server without a display or via SSH:

```bash
# Activate environment
conda activate deep-mric

# Start Jupyter on a specific port (useful for SSH tunneling)
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0

# Or use JupyterLab
jupyter lab --no-browser --port=8888 --ip=0.0.0.0
```

Then access via SSH tunnel:
```bash
# On your local machine
ssh -L 8888:localhost:8888 user@linux-server
# Then open http://localhost:8888 in your browser
```

---

## 2. Run Notebooks from Command Line (No Browser Needed)

Convert and run notebooks as Python scripts:

```bash
conda activate deep-mric

# Convert notebook to Python script
jupyter nbconvert --to script train_cnn.ipynb

# Run the converted script
python3 train_cnn.py
```

Or run directly without conversion:
```bash
# Install nbconvert if needed
conda install nbconvert

# Run notebook directly
jupyter nbconvert --execute --to notebook --inplace train_cnn.ipynb
```

---

## 3. Run Training in Background (Long-Running Jobs)

Use `nohup` or `screen`/`tmux` to keep training running after disconnecting:

### Using nohup:
```bash
conda activate deep-mric
nohup jupyter nbconvert --execute --to notebook train_cnn.ipynb > training.log 2>&1 &
```

### Using screen (better for interactive monitoring):
```bash
# Install screen if needed: sudo apt-get install screen
screen -S training
conda activate deep-mric
jupyter notebook --no-browser
# Press Ctrl+A then D to detach
# Reattach later with: screen -r training
```

### Using tmux:
```bash
# Install tmux if needed: sudo apt-get install tmux
tmux new -s training
conda activate deep-mric
jupyter notebook --no-browser
# Press Ctrl+B then D to detach
# Reattach later with: tmux attach -t training
```

---

## 4. Monitor GPU Usage During Training

```bash
# Watch GPU usage in real-time (updates every 1 second)
watch -n 1 nvidia-smi

# Or use a continuous monitor
nvidia-smi -l 1

# Check which process is using GPU
fuser -v /dev/nvidia*
```

---

## 5. Run Multiple Models in Parallel

Train multiple models simultaneously (if you have enough GPU memory):

```bash
conda activate deep-mric

# Terminal 1
jupyter nbconvert --execute --to notebook train_cnn.ipynb &

# Terminal 2
jupyter nbconvert --execute --to notebook train_rnn_lstm.ipynb &

# Terminal 3
jupyter nbconvert --execute --to notebook train_gru.ipynb &
```

Or use GNU parallel:
```bash
parallel -j 3 jupyter nbconvert --execute --to notebook {} ::: train_cnn.ipynb train_rnn_lstm.ipynb train_gru.ipynb
```

---

## 6. Set Up Automatic Training Script

Create a bash script to run the full pipeline:

```bash
#!/bin/bash
# save as run_all_training.sh

conda activate deep-mric

# Preprocessing
python3 preprocess_vgg16.py

# Augmentation
python3 augment_training_data.py

# Training (run sequentially)
jupyter nbconvert --execute --to notebook train_cnn.ipynb
jupyter nbconvert --execute --to notebook train_rnn_lstm.ipynb
jupyter nbconvert --execute --to notebook train_gru.ipynb

echo "All training completed!"
```

Make it executable and run:
```bash
chmod +x run_all_training.sh
./run_all_training.sh
```

---

## 7. Check System Resources Before Training

```bash
# Check available disk space
df -h

# Check RAM
free -h

# Check CPU info
lscpu

# Check GPU info
nvidia-smi
```

---

## 8. Optimize for Linux Performance

```bash
# Set number of CPU threads for PyTorch (add to your notebook or script)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Disable GUI backend for matplotlib (if running headless)
export MPLBACKEND=Agg
```

---

## 9. Quick Verification Commands

```bash
# Verify conda environment
conda info --envs

# Check if all packages are installed
python3 -c "import torch, torchvision, cv2, sklearn, ultralytics; print('All OK!')"

# Test GPU
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
```

---

## 10. Useful Linux Commands for Training

```bash
# Find large files (check if models are saved correctly)
find models/ -type f -size +100M

# Monitor training progress in real-time
tail -f training.log

# Check running Python processes
ps aux | grep python

# Kill a stuck training process
pkill -f jupyter
```

---

## Quick Setup Checklist

1. **Install conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate deep-mric
   ```

2. **Copy dataset to Linux directory:**
   ```bash
   # Ensure data structure is correct
   ls -la data/raw_dataset/
   ```

3. **Run preprocessing:**
   ```bash
   python3 preprocess_vgg16.py
   python3 augment_training_data.py
   ```

4. **Start training (choose method):**
   - Interactive: `jupyter notebook --no-browser`
   - Background: `nohup jupyter nbconvert --execute --to notebook train_cnn.ipynb > training.log 2>&1 &`
   - Screen/Tmux: Use screen or tmux for detachable sessions

---

## Troubleshooting

### Issue: Jupyter not accessible via SSH
**Solution:** Use SSH port forwarding:
```bash
ssh -L 8888:localhost:8888 user@server
```

### Issue: Training stops when SSH disconnects
**Solution:** Use `nohup`, `screen`, or `tmux` (see section 3)

### Issue: Out of memory
**Solution:** 
- Check with `free -h`
- Reduce batch size in notebooks
- Close other applications

### Issue: GPU not detected
**Solution:**
```bash
nvidia-smi  # Check if GPU is visible
python3 -c "import torch; print(torch.cuda.is_available())"  # Check PyTorch CUDA
```

---

**Note:** This guide assumes you have already:
- ✅ Installed conda
- ✅ Created the conda environment (`conda env create -f environment.yml`)
- ✅ Copied your dataset to the Linux machine
- ✅ Have NVIDIA GPU with CUDA (optional but recommended)

