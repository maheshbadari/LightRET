# Utility Scripts

---

## retvec_export.py

**Purpose:** Download pretrained RetVec-v1 from TensorFlow Hub, verify its architecture
against a known forward-pass reference, and export its weights to a NumPy archive
that PyTorch can load.

### Why this is needed

The official RetVec release is a TensorFlow SavedModel. LightRet uses a PyTorch port
(`src/models/retvec_embedder.py`) that loads the exact same weights. This script
performs the one-time extraction and writes `retvec_v1_weights.npz`.

### Architecture extracted

```
Input: (B, 16, 24) binary representation
    → Flatten  → (B, 384)
    → dense_3: Linear(384→256) + GELU
    → dense_4: Linear(256→256) + GELU
    → dense_5: Linear(256→256) + Tanh
Output: (B, 256) float32, values in [-1, 1]
```

### Dependencies

```bash
pip install tensorflow retvec
```

> These are only needed for this script. They do not need to be installed for training.

### Run

```bash
python retvec_export.py
```

### Output

| File | Size | Description |
|---|---|---|
| `retvec_v1_weights.npz` | ~1 MB | Weights for all 3 Dense layers |

### Verification

The script runs an exhaustive forward-pass comparison between the TF serving model
and the exported weights loaded into a NumPy replica. Any difference above 2e-6
causes an assertion error, ensuring the export is bit-exact.

---

## save_wikitext_local.py

**Purpose:** Download WikiText-103-raw-v1 from HuggingFace Hub and save it in
Arrow format for fully offline use on Kaggle.

### Why this is needed

Kaggle notebooks with internet OFF cannot download WikiText at training time.
This script runs once on your local machine, producing a self-contained Arrow
dataset that can be uploaded to Kaggle as a dataset input and loaded in
milliseconds with `load_from_disk`.

### Run

```bash
python save_wikitext_local.py
```

### Output

```
wikitext_local/
├── train/          ← 1,801,350 rows (~480 MB Arrow)
└── validation/     ← 3,760 rows (small)
```

### Zip and upload to Kaggle

```powershell
# Windows (PowerShell):
Compress-Archive wikitext_local wikitext_local.zip

# Linux / Mac:
zip -r wikitext_local.zip wikitext_local/
```

Upload `wikitext_local.zip` to Kaggle as a new dataset named **`wikitext-local`**.

### How the notebook uses it

The `_load_wikitext(split)` function in `src/data/dataset.py` checks these paths in order:

1. `/kaggle/input/wikitext-local/wikitext_local/{split}` — Kaggle dataset input
2. `wikitext_local/{split}` — local development path
3. HuggingFace Hub (fallback, requires internet)

If the Kaggle dataset is attached, no internet is needed for Stage 1 or Stage 2 training.

---

## paper/build_pdf.py

**Purpose:** Render the full LightRet research paper to a professional PDF using
ReportLab — no LaTeX installation required.

### Run

```bash
python paper/build_pdf.py
```

### Output

| File | Description |
|---|---|
| `paper/LightRet_paper.pdf` | Complete paper (~25 KB) |

### Contents of the generated PDF

- Two-column A4 layout with colored header/footer rules
- Title block with author, affiliation, and email
- All 23 numbered equations (RetVec forward pass, Stage 1–3 losses, BiGRU,
  Transformer, MHA, label projection)
- TikZ-equivalent pipeline diagram drawn with ReportLab Drawing API
- 9 tables (architecture, hyperparameters, noise levels, F1 results, ablation, speed)
- Algorithm box for BIO label projection
- 26 formatted references

### Filling in results

All numeric placeholders appear as `[XX.X]` in the PDF. After training completes:

1. Update the tables in `paper/sections/experiments.tex` with actual F1 scores
2. Re-run `python paper/build_pdf.py` to regenerate the PDF

### LaTeX alternative

If LaTeX is available on your system, compile the full-featured version:

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
# or simply:
make
```

The LaTeX source includes a full TikZ pipeline diagram, proper math environments,
and BibTeX citations.
