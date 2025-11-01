# 🧠 GPU Environment & SLURM Basics — UMich ARC Great Lakes

This README documents how to run **Cellpose with GPU acceleration** on the **UMich ARC Great Lakes** cluster.

It explains:

- What happens when you request a GPU node  
- How to use SLURM interactively or in batch mode  
- The exact cluster Anaconda module to use  
- How to create and verify two clean Cellpose environments (CPU → GPU)  

---

## 🚀 1. Why GPU nodes are required

Login nodes (`gl-loginX`) **do not have GPUs**.  
To actually run code that uses CUDA (PyTorch, Cellpose, etc.), you must request a **GPU node** through the SLURM scheduler.

Example command:

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:20:00 --pty bash
```

| Flag | Meaning |
|------|----------|
| `--partition=gpu` | Use the GPU queue instead of CPU nodes |
| `--gres=gpu:1` | Request **1 GPU card** |
| `--cpus-per-task=4` | Use 4 CPU cores for data loading and preprocessing |
| `--mem=16G` | Allocate 16 GB of system memory |
| `--time=00:20:00` | Reserve the node for 20 minutes |
| `--pty bash` | Start an *interactive shell* on the GPU node |

Output you’ll see:
```
srun: job queued and waiting for resources
srun: job has been allocated resources
```
That means SLURM found a GPU node and launched a new shell (for example `gl1016`).  
Everything you run after this executes **on the GPU node**.

---

## 🧠 2. What happens on a GPU node

Once your job starts, check:
```bash
nvidia-smi
```

Typical output:
```
Driver Version: 570.124.06     CUDA Version: 12.8
GPU Name: Tesla V100-PCIE-16GB
```

- **Driver Version** → determines which CUDA wheels you can use  
- **CUDA Version** → choose the right PyTorch tag (`cu128` for 12.8)  
- **GPU Name** → confirms what hardware you got

---

## ⚙️ 3. Initialize Conda from the cluster Anaconda module

Inside the GPU shell:

```bash
module purge
module load python3.10-anaconda/2023.03
CONDA_BASE=/sw/pkgs/arc/python3.10-anaconda/2023.03

# Enable 'conda activate'
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  eval "$("$CONDA_BASE/bin/conda" shell.bash hook)"
fi
```

Now you can activate your environments:
```bash
conda activate cellpose4   # or cellpose3
```

---

## 🧩 4. Minimal GPU configuration for Cellpose testing

| Resource | Recommended | Why |
|-----------|--------------|-----|
| Partition | `gpu` | Required for CUDA |
| GPUs | `--gres=gpu:1` | Cellpose uses one GPU |
| CPUs | `--cpus-per-task=4` | Balanced data throughput |
| Memory | `--mem=16G` | Fits most microscopy TIFFs |
| Time | `--time=00:20:00` | Plenty for short tests |
| Python | 3.10 | Matches cluster module |
| Torch | `torch==2.9.0+cu128` | Compatible with CUDA 12.8 |
| Cellpose | 4.0.7 or 3.1.1.2 | Versions under test |

For heavier runs: increase `--time` and `--mem`, or use a batch script.

---

## 🧮 5. CUDA tag reference for PyTorch wheels

| CUDA / Driver | PyTorch Tag | Example Command |
|----------------|-------------|----------------|
| CUDA 12.8 | `cu128` | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` |
| CUDA 12.6 | `cu126` | same pattern |
| CUDA 12.1 | `cu121` | 〃 |
| CUDA 11.8 | `cu118` | 〃 |

Use the tag matching your `nvidia-smi` CUDA Version.

---

## ✅ 6. Verify GPU access

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

Expected output on a GPU node:
```
torch: 2.9.0+cu128
cuda_available: True
device: Tesla V100-PCIE-16GB
```

---

## 🔬 7. Running Cellpose with GPU

```bash
cellpose   --dir /path/to/images   --pretrained_model cyto3   --diameter 30   --save_tif   --use_gpu
```

You should see:
```
>>> using GPU (CUDA)
```

---

## 🧰 8. Common patterns

### Interactive (testing)
```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:20:00 --pty bash
module load python3.10-anaconda/2023.03
source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
conda activate cellpose4
cellpose --use_gpu --dir ./ --pretrained_model cyto3 --diameter 30
```

### Batch (reproducible jobs)
`run_cellpose_gpu.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=cellpose_gpu
#SBATCH --account=parent0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=cellpose_gpu_%j.out

module load python3.10-anaconda/2023.03
source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
conda activate cellpose4

cellpose --dir /path/to/images --pretrained_model cyto3 --diameter 30 --save_tif --use_gpu
```

Submit with:
```bash
sbatch run_cellpose_gpu.slurm
```

---

## 📂 Recommended files in this project

| File | Purpose |
|------|----------|
| `create_and_test_cellpose_envs.sh` | CPU baseline env setup + verification |
| `test_cellpose_gpu_env.sh` | GPU probe (must run on GPU node) |
| `run_cellpose_gpu.slurm` | Example batch GPU job |
| `README_GPU_SETUP.md` | This documentation |

---

## 🧭 Summary

- **CPU login node:** safe place to build envs; no CUDA.  
- **GPU node:** requested via SLURM; run CUDA tasks here.  
- **Cellpose test:** 1 GPU, 4 CPUs, 16 GB RAM, 20 min wall time is ideal.  
- **CUDA 12.8 driver ⇒ PyTorch wheel tag `cu128`.**  
- Verified Cellpose 4.0.7 and 3.1.1.2 run correctly on both CPU and GPU.  
