# Cellpose on UMich HPC

Template repository for running the [Cellpose](https://github.com/MouseLand/cellpose) segmentation toolkit on University of Michigan HPC clusters via SLURM.

## Layout
- `cellpose/`: Git clone of the upstream Cellpose project (kept as a submodule-style checkout).
- `config/`: YAML templates describing datasets and Cellpose options.
- `scripts/`: Helper scripts for launching Cellpose runs from configs.
- `slurm/`: Example SLURM submission scripts for Great Lakes/Armis.

## Getting Started
```bash
git clone git@github.com:<your-user>/cellpose_on_umich_hpc.git
cd cellpose_on_umich_hpc
conda env create -f environment.yml
conda activate cellpose-hpc
```

The environment installs Cellpose in editable mode from the local `cellpose/` checkout so you can track or patch the upstream code as needed.

To pull updates from upstream Cellpose:
```bash
cd cellpose
git remote add upstream https://github.com/MouseLand/cellpose.git   # first time only
git fetch upstream
git checkout main
git merge upstream/main
cd ..
```

## Configure a Job
1. Make sure your images live on a filesystem visible to the cluster (e.g. `/nfs/turbo`, `/scratch`, `/home`).
2. Update `config/cellpose_job.yaml` with the input directory, filename filter, and desired Cellpose options.
3. Double-check GPU vs CPU requirements and output locations.

Test locally (no SLURM) using the helper script:
```bash
python scripts/run_cellpose.py --config config/cellpose_job.yaml --dry-run
python scripts/run_cellpose.py --config config/cellpose_job.yaml          # executes Cellpose
```

## Submit to SLURM
Edit `slurm/run_cellpose.slurm` to set the correct account, partition, time, and module/conda activation commands. Then run:
```bash
sbatch slurm/run_cellpose.slurm
```

Logs land in `logs/` (created automatically) and results in `results/<run-name>/` by default.

## Next Steps
Customize the config, scripts, and SLURM template for your exact workflows, commit the changes, and push the repo to GitHub as `cellpose_on_umich_hpc`.
