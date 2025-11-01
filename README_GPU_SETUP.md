# 🧠 GPU Environment & SLURM Basics — UMich ARC Great Lakes (Updated)

This README explains how to **test, verify, and run Cellpose with GPU acceleration** on the **UMich ARC Great Lakes** cluster using the updated scripts.

It now includes the final working setup from your successful GPU tests (Tesla V100, CUDA 12.8, cu126 wheels).

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

Output example:
```
srun: job queued and waiting for resources
srun: job has been allocated resources
```
This means SLURM found a GPU node (for example `gl1016` or `gl1021`) and launched a new shell there.  
Everything you run after that executes **on the GPU node**, not on the login node.

---

## 🧠 2. Checking the GPU node

Once you’re on the GPU node:
```bash
nvidia-smi
```

Typical output:
```
Driver Version: 570.124.06     CUDA Version: 12.8
GPU Name: Tesla V100-PCIE-16GB
```

This tells you:
- **Driver Version:** Determines which CUDA wheels can be used by PyTorch.
- **CUDA Version:** For this example, CUDA 12.8 → PyTorch tag `cu126` (works perfectly).
- **GPU Name:** The specific hardware you were allocated (here, Tesla V100).

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

Now you can activate your Cellpose environments created from your CPU setup:

```bash
conda activate cellpose4   # or cellpose3
```

---

## 🧩 4. Minimal GPU configuration for Cellpose tests

| Resource | Recommended | Why |
|-----------|-------------|-----|
| Partition | `gpu` | Required for CUDA |
| GPUs | `--gres=gpu:1` | Cellpose uses one GPU |
| CPUs | `--cpus-per-task=4` | Reasonable balance for data loading |
| Memory | `--mem=16G` | Enough for typical microscopy TIFFs |
| Time | `--time=00:20:00` | Plenty for testing |
| Python | 3.10 | Matches the Anaconda module |
| Torch | `torch==2.9.0+cu126` | Works on CUDA 12.8 (forward compatible) |
| Cellpose | 4.0.7 and 3.1.1.2 | Tested and verified GPU-ready |

---

## ✅ 5. Verifying GPU availability automatically

Once on the GPU node, run the single one-click test script:
```bash
bash test_and_configure_cellpose_gpu_envs.sh
```

This will:
1. Detect your GPU’s CUDA version from `nvidia-smi`.
2. Pick the correct PyTorch CUDA build tag (`cu126` for your node).
3. Activate both `cellpose4` and `cellpose3`.
4. Replace Torch with a matching CUDA build if needed.
5. Print a clear JSON summary for each environment showing:
   ```json
   {
     "torch": "2.9.0+cu126",
     "cuda_available": true,
     "device_0": "Tesla V100-PCIE-16GB"
   }
   ```

If you see `cuda_available: true`, your environment is GPU-ready.

---

## 🔬 6. Running Cellpose with GPU

From the same GPU session:

```bash
conda activate cellpose4    # or cellpose3
cellpose   --dir /path/to/images   --pretrained_model cyto3   --diameter 30   --save_tif   --use_gpu
```

Expected log line:
```
>>> using GPU (CUDA)
```

You’ll find output mask files like `_cp_masks.tif` next to your input image.

---

## 🧰 7. Batch SLURM submission (optional)

If you want to run GPU jobs non-interactively, create a file `run_cellpose_gpu.slurm`:

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

## 📂 Project file summary

| File | Purpose |
|------|----------|
| `create_and_test_cellpose_envs.sh` | CPU environment creation and verification |
| `test_cellpose_gpu_envs_v2.sh` | One-click GPU check and setup (this test) |
| `run_cellpose_gpu.slurm` | Example GPU batch submission script |
| `README_GPU_SETUP_v2.md` | This documentation |

---

## 🧭 Summary

- Both environments (`cellpose4` and `cellpose3`) now detect and use the GPU correctly.  
- Great Lakes GPU nodes (CUDA 12.8, Tesla V100) run perfectly with Torch 2.9.0+cu126.  
- You can safely use `--use_gpu` in either env for segmentation.  
- For batch jobs, use the provided `run_cellpose_gpu.slurm` example.

---
