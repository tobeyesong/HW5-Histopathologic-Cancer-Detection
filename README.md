# Histopathologic Cancer Detection

A concise pipeline for classifying 96×96 histopathology image patches as tumor or normal using a fine‑tuned EfficientNet‑B0 model.

## Repository Structure
```
HISTO-CANCER-DETECT/
├── venv/                  # Python virtual environment (ignored)
├── artifacts/             # any custom artifacts (ignored)
├── data/                  # image data (ignored)
│   ├── train/             # training TIFF patches
│   ├── test/              # test TIFF patches
│   └── train_labels.csv   # CSV mapping train IDs to labels
├── notebooks/             # analysis and report notebooks
│   ├── HW 5.ipynb         # final notebook for PDF export
│   └── k_leaderboard.png  # screenshot of Kaggle leaderboard
├── outputs/               # training outputs (ignored)
│   ├── args.json          # training run arguments
│   ├── best.pt            # best checkpoint saved
│   ├── final.pt           # final checkpoint
│   └── metrics.json       # training metrics (AUC, loss per epoch)
├── src/                   # source code modules
│   ├── data.py            # Dataset & transform definitions
│   ├── model.py           # EfficientNet‑B0 instantiation
│   ├── pipeline.py        # train/validate loops
│   ├── train.py           # CLI entrypoint for training
│   └── infer.py           # CLI entrypoint for batched inference
├── .gitignore             # ignore rules for git
├── README.md              # this file
├── requirements.txt       # Python dependencies
└── submission.csv.gz      # example compressed submission file
```

> **Note:** Folders such as `venv/`, `data/`, `outputs/`, and `artifacts/` are excluded via `.gitignore` to keep the repository light.

## Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/histo-cancer-detection.git
   cd histo-cancer-detection
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # on macOS/Linux
   # or
   venv\Scripts\activate       # on Windows PowerShell
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure `.gitignore` is in place** (should contain rules to ignore `venv/`, `data/`, `outputs/`, etc.).

## Data Acquisition

Download the Kaggle dataset (requires `kaggle.json` in your home directory):
```bash
kaggle competitions download -c histopathologic-cancer-detection -p data
unzip data/histopathologic-cancer-detection.zip -d data
```

This will populate `data/train/` and `data/test/` with `.tif` images and the `train_labels.csv` file.

## Usage

### 1. Train the Model

Fine‑tune EfficientNet‑B0 on the training patches:
```bash
python -m src.train \
  --img_dir data/train \
  --csv_path data/train_labels.csv \
  --epochs 10 \
  --batch 256 \
  --lr 1e-3 \
  --outdir outputs/
```
- Writes `outputs/metrics.json` (loss/AUC per epoch) and `outputs/best.pt` (best validation AUC checkpoint).

### 2. Run Inference

Generate predictions on the test set in batches (fast, GPU‑accelerated):
```bash
python -m src.infer \
  --img_dir data/test \
  --model_path outputs/best.pt \
  --output submission.csv
gzip -f submission.csv
```
- Produces `submission.csv.gz` ready for Kaggle submission.

### 3. Review Notebook

Open the Jupyter notebook for EDA, model summary, and result plots:
```bash
jupyter lab notebooks/HW\ 5.ipynb
```
- Use tags to hide code input when exporting to PDF:
  ```bash
  jupyter nbconvert notebooks/HW\ 5.ipynb \
    --to pdf \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_input_tags hide_input
  ```

## Submission

After inference, submit via the Kaggle CLI:
```bash
kaggle competitions submit \
  -c histopathologic-cancer-detection \
  -f submission.csv.gz \
  -m "EfficientNet-B0 inference, val AUC=0.9953"
```

Check your submission status:
```bash
kaggle competitions submissions -c histopathologic-cancer-detection
```

# HW5-Histopathologic-Cancer-Detection
