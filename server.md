# Server Configuration for Training

## Primary Training Server: `themachine` (Desktop)

**All GPU/CPU-intensive training MUST run on the desktop server, not the laptop.**

### Connection Details
- **Hostname**: `desktop` (via SSH config) or `themachine`
- **Tailscale IP**: 100.104.129.65
- **User**: rylan

### Hardware Specs
- **CPU**: 12 cores (AMD)
- **RAM**: 16GB
- **GPU**: NVIDIA GTX 1070 Ti (8GB VRAM)
  - ⚠️ CUDA version too old for PyTorch GPU - runs on CPU only
- **Storage**: 253GB free

### Project Location on Server
```
~/dev/personal/quantum-derivatives-trader/
```

### Syncing Code to Server
```bash
# From laptop, sync changes to desktop
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
  ~/dev/personal/quantum-derivatives-trader/ \
  desktop:~/dev/personal/quantum-derivatives-trader/
```

### Running Training
```bash
# SSH to desktop
ssh desktop

# Navigate to project
cd ~/dev/personal/quantum-derivatives-trader/

# Activate venv
source .venv/bin/activate

# Run training in tmux (persists after disconnect)
tmux new-session -d -s training "python scripts/train_hybrid.py --epochs 500 --n_qubits 4 --n_layers 2"

# Check status
tmux attach -t training
# Detach with Ctrl+B, D
```

### Overnight Experiments
Use the overnight script for batch experiments:
```bash
./scripts/run_overnight.sh
```

Logs go to: `logs/overnight_YYYYMMDD.log`
Results go to: `outputs/hybrid/YYYYMMDD_HHMMSS/`

### Why Not Laptop?
- Laptop (`theLittleMachine`) has limited resources
- Quantum circuit simulation is CPU-bound and slow
- Training can take hours - don't want to kill laptop battery/thermals
- Desktop can run overnight unattended

### Checking Training Status from Laptop
```bash
# Check if training is running
ssh desktop "ps aux | grep train_hybrid | grep python"

# Check logs
ssh desktop "tail -50 ~/dev/personal/quantum-derivatives-trader/logs/train_hybrid.log"

# Check tmux sessions
ssh desktop "tmux list-sessions"
```
