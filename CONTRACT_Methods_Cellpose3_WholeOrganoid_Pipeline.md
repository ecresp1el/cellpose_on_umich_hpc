# CONTRACT — Methods & Staged Plan for Cellpose v3 Whole‑Organoid Pipeline

**Repository:** `ecresp1el/cellpose_on_umich_hpc`  
**Staging Reference:** `WHOLE_ORGANOID_STAGING_AND_FLOW.md` (authoritative directories)  
**Model Family:** Cellpose **v3** (Python API) specialist model for *whole‑organoid* segmentation  
**Contract Purpose:** This document is the canonical, method‑level contract. It defines **names**, **inputs/outputs**, **pre/postconditions**, **side‑effects**, **failure modes**, and **acceptance checks** for every method. It also lays out a **staged roadmap** with “Ready/Blocked” gates. All code, comments, and docstrings must adhere to this contract unless this file is version‑bumped and updated.

---

## 0) Workspace (Authoritative)

**Turbo root:** `/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/`

```
/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/
├─ dataset/
│  ├─ images/               # ALL organoid .tif/.tiff from any condition (pooled)
│  ├─ labels/               # any existing ground-truth masks as pngs
│  ├─ train/
│  │   ├─ images/           # ~10–20 curated for first training
│  │   └─ labels/           # ground truth masks
│  └─ valid/
│      ├─ images/           # small hold-out
│      └─ labels/
├─ results/
│  ├─ cp3_baseline_cyto3/   # first inference (no training)
│  ├─ cp3_v001/             # first tuned model outputs
│  ├─ cp3_v002/             # second tuned model outputs (if needed)
│  └─ qc_panels/            # side-by-side PNG panels for QC (optional aggregate)
└─ logs/
```

> All paths in config and code must resolve to this layout. Deviation requires updating this contract first.

---

## 1) Global Invariants (Must Always Hold)

1) **Native Scale Policy**  
   - **Training:** Cellpose v3 **API** with `rescale=False` (never CLI when using pretrained); `bsize=512`.  
   - **Evaluation:** `resample=False`; **no `diameter`** passed; `niter=2000`; `bsize=512`.

2) **Tidy Artifacts (Unified Stems)** — For input stem `S`:  
   - `eval/masks/S_masks.tif` (+ optional `S_masks.npy`)  
   - `eval/flows/S_flows.npy`, `S_flows.tif`  
   - `eval/prob/S_prob.tif`  
   - `eval/rois/S_rois.zip` (ImageJ ROI archive)  
   - `eval/panels/S_panel_1x4.png`  
   - `eval/json/S_summary.json`  
   - Run aggregate: `eval/eval_summary.json`

3) **Reproducibility** — Every run contains:
   - `cfg/config_snapshot.yaml`, `cfg/env_and_cfg.json`, `cfg/dataset_report.json`  
   - `train/weights_final.pt`, `train/metrics.json`, `train/stdout_stderr.log`

4) **Versioning** — New training attempts use a new `results/<model_version>/` (e.g., `cp3_v002`). Never overwrite prior runs.

---

## 2) Commenting & Docstring Conventions (Mandatory)

- **Docstring template (all public methods):**
  - **Summary:** 1–2 lines (imperative)  
  - **Args:** name, type, units (if applicable)  
  - **Returns:** type, semantic meaning  
  - **Writes:** explicit file paths (relative to run_dir)  
  - **Raises:** specific exceptions and when  
  - **Notes:** algorithmic rationale or non‑obvious behavior  
  - **Checks:** pre/postconditions asserted in the method

- **Inline comments:** prefer *why* over *what*; keep to <80 cols when possible.  
- **Logging:** use `RunLogger` for structured events; no `print` in library code (only in CLI wrappers).

---

## 3) Configuration (YAML Keys & Defaults)

```yaml
model_name_out: cp3_v001

paths:
  turbo_root: /nfs/turbo/umms-parent/cellpose_wholeorganoid_model
  data_images_train: /nfs/turbo/umms-parent/cellpose_wholeorganoid_model/dataset/train/images
  data_labels_train: /nfs/turbo/umms-parent/cellpose_wholeorganoid_model/dataset/train/labels
  data_images_valid: /nfs/turbo/umms-parent/cellpose_wholeorganoid_model/dataset/valid/images
  data_labels_valid: /nfs/turbo/umms-parent/cellpose_wholeorganoid_model/dataset/valid/labels
  results_root: /nfs/turbo/umms-parent/cellpose_wholeorganoid_model/results
  logs_root: /nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs

labels:
  mask_filter: _seg.npy        # or .png; DatasetManager resolves

train:
  use_pretrained: true         # cyto3 init
  model_type: cyto3
  n_epochs: 100
  learning_rate: 1e-5
  weight_decay: 0.1
  batch_size: 1
  rescale: false               # v3 API
  bsize: 512                   # API tiling

eval:
  niter: 2000
  resample: false
  bsize: 512
  flow_threshold: 0.4
  cellprob_threshold: 0.0
  save_panels: true
  save_rois: true
  save_flows: true
  save_prob: true
  eval_split: valid            # valid | all

system:
  seed: 1337
  use_cuda: true
```

---

## 4) Class & Method Contracts

### 4.1 `ConfigStore`
- **`load_from_yaml(path: str) -> Config`**  
  **Inputs:** YAML path  
  **Returns:** validated `Config` object  
  **Raises:** `FileNotFoundError`, `ValueError` (missing keys/invalid paths)  
  **Writes:** none  
  **Checks:** absolute paths; directories exist; `model_name_out` non‑empty

- **`save_snapshot(dst_dir: Path, cfg: Config) -> None`**  
  **Inputs:** `dst_dir`, `cfg`  
  **Writes:** `dst_dir/config_snapshot.yaml` and `dst_dir/env_and_cfg.json`  
  **Checks:** `dst_dir` exists; JSON serializable

---

### 4.2 `DatasetManager`
- **`verify_structure() -> DatasetReport`**  
  **Returns:** counts (`n_train`, `n_valid`, `n_masks`), warnings (missing labels)  
  **Writes:** `run_dir/cfg/dataset_report.json`  
  **Raises:** `AssertionError` if required dirs missing  
  **Checks:** `train/images` and `valid/images` non‑empty

- **`list_images(split: {"train","valid","all"}) -> list[Path]`**  
  **Returns:** sorted list of `.tif/.tiff` paths  
  **Writes:** none  
  **Checks:** split validity; dedupe across splits for `all`

- **`image_id(image_path: Path) -> str`**  
  **Returns:** sanitized/stable stem used for artifact filenames  
  **Writes:** none

- **`label_for(image_path: Path) -> Optional[Path]`**  
  **Returns:** label path based on `labels.mask_filter` or `None`  
  **Writes:** none

---

### 4.3 `RunLogger`
- **`log_env_and_cfg(cfg: Config, dst_dir: Path) -> None`**  
  **Writes:** `dst_dir/env_and_cfg.json` (Python, torch, CUDA/MPS, SLURM env, config snapshot)  
  **Checks:** JSON serializable, human‑readable

- **`tee_stdout(to_file: Path)`** *(context manager)*  
  **Writes:** mirrored stdout/stderr to `to_file`  
  **Checks:** creates parent folder, appends mode safe

- **`time_block(name: str)`** *(context manager)*  
  **Writes:** durations under `train/metrics.json` or eval aggregate  
  **Checks:** monotonic clock; unique block names per run

---

### 4.4 `TrainerCellpose3`
- **`load_model(use_pretrained: bool, model_type: str) -> ModelHandle`**  
  **Returns:** initialized model handle (cyto3 or scratch)  
  **Checks:** if `use_pretrained`, verify weights available; CPU/GPU availability logged

- **`build_train_args(cfg: Config) -> TrainArgs`**  
  **Returns:** mapping with `rescale=False`, `bsize=512`, `n_epochs`, `lr`, `wd`, `batch_size`  
  **Checks:** values numeric & >0 where applicable

- **`train(model, images: list[Path], labels: list[Path], args: TrainArgs) -> TrainedModelHandle`**  
  **Writes:** streaming logs via `RunLogger`; temporary checkpoints (optional)  
  **Checks:** `len(images)==len(labels)` when supervised; GPU memory sufficient

- **`save_weights(handle: TrainedModelHandle, dst_dir: Path) -> Path`**  
  **Writes:** `dst_dir/weights_final.pt` (plus checksum), returns path  
  **Checks:** file integrity (non‑zero size), hash logged

- **`record_training_metadata(dst_dir: Path, metrics: dict) -> None`**  
  **Writes:** `dst_dir/metrics.json` (loss curves, timings, hyperparams)  
  **Checks:** strictly append or overwrite atomically

---

### 4.5 `EvaluatorCellpose3`
- **`build_eval_args(cfg: Config) -> EvalArgs`**  
  **Returns:** `niter=2000`, `bsize=512`, `resample=False`, thresholds  
  **Checks:** thresholds within valid ranges

- **`evaluate(model, image_path: Path, args: EvalArgs) -> EvalResult`**  
  **Returns:** `EvalResult{masks, flows(dy,dx), cellprob, timings, thresholds}`  
  **Writes:** none (pure compute)  
  **Raises:** `RuntimeError` on model failure  
  **Checks:** no `diameter` passed; GPU/CPU path logged; deterministic seeds respected

- **`postprocess(result: EvalResult) -> EvalResult`** *(optional)*  
  **Returns:** modified result after optional morphology filters  
  **Checks:** idempotent when filters disabled (default off)

---

### 4.6 `ArtifactWriter`
- **`write_masks(image_id: str, masks) -> Path`** → `eval/masks/{id}_masks.tif` (+ optional npy)  
  **Checks:** masks are integer‑labeled; dtype uint16 when saved to TIFF

- **`write_prob(image_id: str, cellprob) -> Path`** → `eval/prob/{id}_prob.tif`  
  **Checks:** float32 allowed; if scaled, record scale in JSON

- **`write_flows(image_id: str, flows) -> dict`** → `eval/flows/{id}_flows.npy` + `{id}_flows.tif`  
  **Checks:** flows has shape (2,H,W); viz normalized

- **`write_rois(image_id: str, masks) -> Path`** → `eval/rois/{id}_rois.zip`  
  **Checks:** ImageJ ROI archive valid; ROIs match mask labels

- **`write_panel(image_id: str, png_bytes) -> Path`** → `eval/panels/{id}_panel_1x4.png`  
  **Checks:** image width = sum of 4 panels; height constant; lossless PNG

- **`write_summary_json(image_id: str, stats: dict) -> Path`** → `eval/json/{id}_summary.json`  
  **Checks:** includes `n_cells`, size stats, thresholds, `niter`, `bsize`, timings

- **`write_eval_aggregate(stats: dict) -> Path`** → `eval/eval_summary.json`  
  **Checks:** reproducible order; includes per‑image references

---

### 4.7 `QCReporter`
- **`make_panel(image, cellprob, flows, masks) -> bytes`**  
  **Returns:** PNG (1×4: input | prob | flow viz | mask overlay)  
  **Checks:** consistent color maps and scales; scale bars (optional)

- **`render_subset(results: Iterable[EvalResult], n: Optional[int]) -> None`**  
  **Writes:** panels via `ArtifactWriter.write_panel`  
  **Checks:** reproducible sampling if `n` provided (seeded)

- **`write_eval_readme(eval_dir: Path) -> None`** *(optional)*  
  **Writes:** brief README describing folders and filenames

---

### 4.8 `WholeOrganoidExperiment`
- **`prepare() -> None`**  
  **Writes:** run_dir structure + `cfg/` contents (snapshot, env, dataset report)  
  **Checks:** run_dir name unique (timestamped), paths resolved

- **`run_training() -> None`**  
  **Writes:** `train/` artifacts (weights, metrics, logs)  
  **Checks:** `rescale=False` honored; `bsize=512` used

- **`run_evaluation(split: Literal["valid","all"]) -> None`**  
  **Writes:** full set of eval artifacts under `eval/`  
  **Checks:** `niter=2000`, `bsize=512`, `resample=False` honored

- **`summarize() -> None`**  
  **Writes:** `eval/eval_summary.json`; prints human summary  
  **Checks:** totals match per‑image JSONs

- **`run_full_cycle() -> None`**  
  **Sequence:** `prepare → run_training → run_evaluation("valid") → summarize`

---

## 5) Staged Roadmap & Acceptance Criteria

### Stage A — Config, Dataset, Bootstrap
- **Deliverables:** `ConfigStore`, `DatasetManager`, `RunLogger`, basic `WholeOrganoidExperiment.prepare()`  
- **Acceptance:**  
  - Run dir created with `cfg/` files (snapshot/env/dataset_report)  
  - Verified counts; clear warnings for missing labels  
  - No training/eval executed

### Stage B — Training (Native Scale)
- **Deliverables:** `TrainerCellpose3` + `WholeOrganoidExperiment.run_training()`  
- **Acceptance:**  
  - Training completes on seed set; `rescale=False` and `bsize=512` logged  
  - `train/weights_final.pt`, `train/metrics.json`, `train/stdout_stderr.log` present

### Stage C — Evaluation & Artifacts
- **Deliverables:** `EvaluatorCellpose3`, `ArtifactWriter`, `QCReporter`, `WholeOrganoidExperiment.run_evaluation()`  
- **Acceptance:**  
  - For each `valid` image, all artifact files exist with correct stems  
  - `niter=2000`, `bsize=512`, `resample=False` logged per run  
  - `eval/eval_summary.json` aggregates counts/timings

### Stage D — Baselines & Re‑Eval
- **Deliverables:** baseline inference to `results/cp3_baseline_cyto3/`, threshold‑only re‑eval flow  
- **Acceptance:**  
  - Baseline folder populated without training  
  - Re‑eval produces a new version folder with adjusted thresholds only

---

## 6) SLURM & Entrypoints (Thin by Contract)

- **SLURM filename:** `slurm/train_wholeorganoid_cp3.slurm`  
- **Behavior:** activate env; call Python entrypoint with `--config <yaml>` and `--mode {full,train,eval}`  
- **No hyperparams in SLURM** — all are in YAML.

Example submit (concept):  
`sbatch slurm/train_wholeorganoid_cp3.slurm`

---

## 7) Change Control

- Any change to method names, signatures, paths, or invariants must:
  1) Update **this contract file** (increment a version header),  
  2) Add a brief “Rationale” note,  
  3) Only then update code to match.

---

## 8) References & Cross‑Repo Links

- Staging & Mac‐side tools: `https://github.com/ecresp1el/organoid-roi-tool`  
- HPC pipeline (this repo): `https://github.com/ecresp1el/cellpose_on_umich_hpc`  
- Upstream docs: Cellpose v3 API (for training/eval knobs), CP mask/ROI conventions.

---

**Owner:** Emmanuel L. Crespo — keep this file in repo root and treat it as the API of your pipeline.
