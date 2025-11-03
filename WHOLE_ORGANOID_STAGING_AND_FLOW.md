# Whole‑Organoid Segmentation — Staging & Conceptual Flow (Bare‑Bones)

**Audience:** You (future you), collaborators, and anyone integrating `organoid-roi-tool` (Mac) with `cellpose_on_umich_hpc` (Great Lakes HPC).

**Scope of this document:**  
This is a *single-source, “living” document* for the **staging phase** and the **overall conceptual flow**. It explains where files live, how the two repos connect, what you must do in what order, and why. It **does not** contain the training or inference code; those come later. At the end of this document you will find a **ready‑to‑run staging script for your Mac** and a small **one‑time cluster init script** to create the destination folder on Turbo.

---

## 0) What we’re building (high‑level)

We are creating a *specialist* **Cellpose v3** model to segment **one whole organoid per image** across a large dataset (≈500–700 images) with **variable organoid sizes** (e.g., some ~600 px diameter, some ~1200 px+). Our constraints:

- **Human‑in‑the‑loop (HITL):** You annotate a small seed set on your **Mac** using the **Cellpose GUI** (fast, interactive).
- **Heavy compute on Great Lakes (HPC):** We train and run large‑scale inference via **SLURM** using your existing `cellpose_on_umich_hpc` repo & Conda envs.  
- **Staging & data management:** All training/validation data and all results are stored under a **single, clearly named Turbo folder**:
  ```
  /nfs/turbo/umms-parent/cellpose_wholeorganoid_model/
  ```
  This keeps the specialist workflow *separate* from your existing `/nfs/turbo/umms-parent/cellpose_gpu_test` area — no disruption.

**Two repos, two roles:**

| Repo | Runs on | Purpose |
|------|---------|---------|
| `organoid-roi-tool` | **Mac** | Select interesting `.tif` images; optionally annotate masks in Cellpose GUI; output is a list of file paths or concrete `<image>.tif` + `<image>_masks.(png/tif)`. |
| `cellpose_on_umich_hpc` | **Great Lakes** | Orchestrate data **staging** (this doc) and later, training/inference/QC via SLURM. All I/O goes to Turbo. |

**Integration is file‑based:** you copy/rsync **files** from your Mac into the Turbo workspace. No complex API.

---

## 1) Target workspace layout on Turbo (no WT/KO split)

We do **not** split by WT/KO in this specialist project. The model should treat “an organoid is an organoid” and learn the size variance. File names still contain experiment context if needed.

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
│  └─ qc_panels/            # side-by-side PNG panels for QC
└─ logs/                    # SLURM stdout/err, notes
```

---

## 2) Staging — what happens and why

**Staging** = copying your local `.tif` and any existing masks from the Mac into the Turbo workspace.

- **One-time** initialization on the cluster creates the folder tree (script provided below).
- **Mac‑side rsync script** flattens all `.tif/.tiff` into `dataset/images/` (ignores WT/KO).
- If you already have masks (`*_masks.png` or `*_cp_masks.tif`) next to your images, the script can copy them too.
- **No validation** in the script — if a glob has no matches, `rsync` simply skips it and continues.
- The script prints each source/destination action and then shows you the first few files on Turbo.

After staging, you have:

- `dataset/images/` containing all candidate training/inference images (macroscopically diverse).
- `dataset/labels/` (optional) containing any known masks.
- `results/` and `logs/` ready for training/inference phases later.

---

## 3) JSON manifest — keep a record

Create a small `wholeorganoid_config.json` in your repo to record:

- `turbo_root` — where the data went,
- `sources` — which local folders you staged,
- `model_plan` — key assumptions (CP‑3, rescale off, typical diameter),
- `models` — track versions you train later (names, epochs, etc.).

This is purely **documentation for reproducibility**; you can plug it into automation later.

---

## 4) What’s next (conceptual only; not part of staging)

When you’re ready to proceed:

1. Pick ~10–20 diverse images from `dataset/images/` + annotate single‑organism masks (or use your existing masks in `dataset/labels/`) → copy into `dataset/train/{images,labels}` and `dataset/valid/{images,labels}`.
2. Train a **Cellpose v3** specialist (rescale **off**) on Great Lakes; save as `cp3_wholeorganoid_v001`.
3. Evaluate on `valid/` (no resample; `diameter=model.diam_mean`) and inspect QC overlays.
4. Apply to all images → inspect failures → annotate 5–20 more → retrain → repeat once or twice.

Keep the loop small and targeted.

---

# 📎 Files to use for STAGING (today)

## A) One‑time: **cluster init** script

Place this file at:

```
/home/elcrespo/Desktop/githubprojects/cellpose_on_umich_hpc/slurm/init_wholeorganoid_folder.sh
```

```bash
#!/usr/bin/env bash
# init_wholeorganoid_folder.sh
# One-time bootstrap of the dedicated Turbo workspace for the whole-organism model.

set -euo pipefail
DEST="/nfs/turbo/umms-parent/cellpose_wholeorganoid_model"

echo "[INIT] Creating unified whole-organoid workspace at $DEST ..."
mkdir -p "$DEST/dataset/images" \
         "$DEST/dataset/labels" \
         "$DEST/dataset/train/images" \
         "$DEST/dataset/train/labels" \
         "$DEST/dataset/valid/images" \
         "$DEST/dataset/valid/labels" \
         "$DEST/results" \
         "$DEST/results/qc_panels" \
         "$DEST/logs"

echo "[OK] Created:"
echo "  $DEST/dataset/images"
echo "  $DEST/dataset/labels"
echo "  $DEST/dataset/train/{images,labels}"
echo "  $DEST/dataset/valid/{images,labels}"
echo "  $DEST/results/"
echo "  $DEST/results/qc_panels/"
echo "  $DEST/logs/"
```

Run it once on the **cluster**:
```bash
bash /home/elcrespo/Desktop/githubprojects/cellpose_on_umich_hpc/slurm/init_wholeorganoid_folder.sh
```

---

## B) **Mac‑side rsync** staging script (no validation, flat pool)

Save this as `~/Desktop/rsync_wholeorganoid_to_turbo.sh` on your **Mac** and make it executable.  
It sends all `.tif/.tiff` from your listed local folders into the unified Turbo pool, optional masks too.

```bash
#!/usr/bin/env bash
# rsync_wholeorganoid_to_turbo.sh
# STAGE ONLY (no validation). Run on your Mac. Adjust the first 3 vars and SOURCES.

set -euo pipefail

### ── EDIT THESE VALUES ──────────────────────────────────────────
REMOTE_USER="ecrespo"                                  # your UMich uniqname
REMOTE_HOST="gl-login2.arc-ts.umich.edu"               # Great Lakes login node
REMOTE_ROOT="/nfs/turbo/umms-parent/cellpose_wholeorganoid_model"  # unified Turbo workspace

# Local source directories with .tif/.tiff files to stage (WT/KO mixed is fine)
SOURCES=(
  "/Volumes/Manny4TBUM/10_16_2025/lhx6_pdch19_WTvsKO_projectfolder/cellprofilerandcellpose_folder/cellpose_multichannel_zcyx/PCDHvsLHX6_WTvsKO_IHC/max/WT"
  "/Volumes/Manny4TBUM/10_16_2025/lhx6_pdch19_WTvsKO_projectfolder/cellprofilerandcellpose_folder/cellpose_multichannel_zcyx/PCDHvsLHX6_WTvsKO_IHC/max/KO"
  "/Volumes/Manny4TBUM/10_13_2025/nestin_dcx_pcdh19_kovswt/cellprofilerandcellpose_folder/cellpose_multichannel_zcyx/NS116A/max/WT"
  "/Volumes/Manny4TBUM/10_13_2025/nestin_dcx_pcdh19_kovswt/cellprofilerandcellpose_folder/cellpose_multichannel_zcyx/NS116A/KO"
)

# Copy any existing masks that are adjacent to images?
COPY_MASKS="yes"  # set to "no" to skip
### ───────────────────────────────────────────────────────────────

# Enable nullglob so unmatched globs don't cause errors
if [ -n "${BASH_VERSION:-}" ]; then shopt -s nullglob; fi
if [ -n "${ZSH_VERSION:-}" ]; then setopt NULL_GLOB || true; fi

DEST_IMAGES="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/dataset/images/"
DEST_LABELS="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/dataset/labels/"

echo "[REMOTE] Ensuring destination directories exist at ${REMOTE_ROOT} ..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${REMOTE_ROOT}/dataset/images' '${REMOTE_ROOT}/dataset/labels' '${REMOTE_ROOT}/results/qc_panels' '${REMOTE_ROOT}/logs'"

echo "[STAGE] Copying .tif/.tiff → ${DEST_IMAGES}"
for SRC in "${SOURCES[@]}"; do
  echo "  • From: $SRC"
  rsync -avh --progress "$SRC/"*.tif  "${DEST_IMAGES}" || true
  rsync -avh --progress "$SRC/"*.tiff "${DEST_IMAGES}" || true
  if [[ "${COPY_MASKS}" == "yes" ]]; then
    rsync -avh --progress "$SRC/"*_masks.*    "${DEST_IMAGES}" 2>/dev/null || true
    rsync -avh --progress "$SRC/"*_cp_masks.* "${DEST_IMAGES}" 2>/dev/null || true
  fi
done

echo "[CHECK] First 20 files on remote:"
ssh "${REMOTE_USER}@${REMOTE_HOST}" "ls -lh '${REMOTE_ROOT}/dataset/images' | head -n 20"

echo "[DONE] Staging complete."
echo "Your pooled images now live at: ${REMOTE_ROOT}/dataset/images/"
```

Run it from your **Mac Terminal**:
```bash
chmod +x ~/Desktop/rsync_wholeorganoid_to_turbo.sh
~/Desktop/rsync_wholeorganoid_to_turbo.sh
```

---


## ✅ Done: Staging Ready

- You have a **clear plan** and **docs** (this file).
- You have a **cluster init script** to create `/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/…`.
- You have a **Mac-side rsync script** to stage images (and optional masks) into `dataset/images/` (and `labels/`) — **no file checks**: it just copies what matches.
- You have an optional **JSON manifest** for tracking.

Next time, we’ll add the minimal training/eval steps on top of this exact structure.

