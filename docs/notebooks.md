# Kaggle Notebooks

Three Kaggle notebooks under `notebooks/` mirror the three training scripts.
Each notebook is self-contained: it copies source files, patches configuration,
embeds the latest `dataset.py` inline, and runs a complete training stage.

---

## Common Setup (all notebooks — Cell 1 & 2)

### Cell 1 — Install packages
```python
!pip install -q transformers datasets
```
`torch` and `numpy` are pre-installed on Kaggle. Only `transformers` and
`datasets` need installing.

### Cell 2 — Working directory setup

Every notebook's setup cell does the following in order:

| Step | What happens |
|---|---|
| Copy `src/` | `shutil.copytree(SOURCE_DATASET/src, /kaggle/working/src)` |
| Copy weights | `shutil.copy(WEIGHTS_DATASET/retvec_v1_weights.npz, /kaggle/working/)` |
| Create `weights/` | `mkdir /kaggle/working/weights` |
| `os.chdir` | Sets working directory to `/kaggle/working` |
| Write `dataset.py` | Decodes a base64-embedded copy of the latest `dataset.py` directly into `/kaggle/working/src/data/dataset.py` — no GitHub fetch needed |
| CoNLL local | Downloads `conll2003_local.zip` from GitHub (1.7 MB) and unzips it if no local Kaggle dataset is attached |

The `dataset.py` is **embedded as base64** in the notebook itself, so it is
always the correct version regardless of what is in the uploaded `lightret-source`
dataset. To update it, run the `build_pdf.py` re-embedding script (or the
notebook generation script) after modifying `src/data/dataset.py`.

---

## Required Kaggle Dataset Inputs

| Dataset name | Mounts at | Contents |
|---|---|---|
| `lightret-source` | `/kaggle/input/lightret-source` | Full project: `src/`, `train_*.py`, etc. |
| `lightret-weights` | `/kaggle/input/lightret-weights` | `retvec_v1_weights.npz` |
| `wikitext-local` | `/kaggle/input/wikitext-local` | WikiText-103 Arrow cache (Stage 1 & 2) |
| `conll2003-local` *(optional)* | `/kaggle/input/conll2003-local` | Pre-saved CoNLL-2003 Arrow (auto-downloaded from GitHub if missing) |

Stage 3 only needs `lightret-source`, `lightret-weights`, and the Stage 2
checkpoint as a dataset input.

---

## stage1_kaggle.ipynb

**Goal:** Train RetBERT to produce BERT-matching sentence embeddings.

### Notebook cells

| Cell | Purpose |
|---|---|
| 1 | Install `transformers datasets` |
| 2 | Setup working dir, embed `dataset.py`, download CoNLL zip |
| 3 | Patch `config.py` (ensures correct dataset names) |
| 4 | Verify GPU (`torch.cuda.is_available()`, print device name & memory) |
| 5 | Verify imports + quick forward-pass sanity check |
| 6 | Override hyperparameters (uncomment to reduce epochs for a test run) |
| 7 | Load `PretrainDataset` (WikiText + CoNLL, ~3.9M sentences) |
| 8 | Build BERT teacher + RetBERT student |
| 9 | Training loop (cosine LR, checkpointing on best loss) |
| 10 | Plot loss curve, print checkpoint size |

### Output
`/kaggle/working/weights/retbert_stage1.pt`

Download this file and upload as a new Kaggle dataset (e.g. `lightret-stage1`)
before running Stage 2.

### Internet requirement
**ON** — needed to download BERT weights (~440 MB) and WikiText-103 if
`wikitext-local` dataset is not attached.

### Estimated runtime (T4)
~8–10 hours for 5 epochs. Reduce `STAGE1_EPOCHS = 2` in Cell 6 for a quick test.

---

## stage2_kaggle.ipynb

**Goal:** Compress RetBERT's token-level representations into LightRet.

### Additional dataset input required
Upload `retbert_stage1.pt` as a Kaggle dataset (e.g. `lightret-stage1`) and
add it as input. The notebook looks for it at:
```
/kaggle/input/lightret-stage1/retbert_stage1.pt
```

### Key difference from Stage 1
- Larger batch size (64 vs 32) — LightRet is much smaller
- Token-level supervision (every word position, not just sentence-level)
- Teacher is RetBERT (not BERT), so no HuggingFace model download needed

### Output
`/kaggle/working/weights/lightret_stage2.pt`

### Estimated runtime (T4)
~4–5 hours for 5 epochs.

---

## stage3_kaggle.ipynb

**Goal:** Fine-tune LightRet + NER head on noisy CoNLL-2003.

### Additional dataset input required
Upload `lightret_stage2.pt` as a Kaggle dataset (e.g. `lightret-stage2`) and
add it as input. The notebook looks for it at:
```
/kaggle/input/lightret-stage2/lightret_stage2.pt
```

### Key differences from Stages 1 & 2
- Dataset is CoNLL-2003 only (~15K sentences) — much faster per epoch
- Two models run simultaneously: teacher (clean, frozen) + student (noisy, trainable)
- Noise is applied on-the-fly in `NERDataset.__getitem__`, so each epoch
  sees different noise patterns automatically
- Validation uses **no noise** (`apply_noise_aug=False`) to measure clean-text F1

### Validation metric
Entity-level F1 computed with `seqeval`. Checkpoint saved when validation F1 improves.

### Output
```
/kaggle/working/weights/lightret_stage3.pt     ← LightRet backbone
/kaggle/working/weights/ner_head_stage3.pt     ← BiLSTM NER head
```

### Internet requirement
**OFF** — CoNLL is loaded from the GitHub-downloaded zip. WikiText is not used
in Stage 3.

### Estimated runtime (T4)
~20–40 minutes for 10 epochs.

---

## Updating the Embedded dataset.py

The setup cell in all three notebooks contains a base64-encoded copy of
`src/data/dataset.py`. Whenever `dataset.py` changes, regenerate the embedding:

```bash
python -c "
import base64, json, pathlib

with open('src/data/dataset.py', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode('ascii')
chunks = [b64[i:i+80] for i in range(0, len(b64), 80)]
new_b64 = '    \"' + '\"\n    \"'.join(chunks) + '\"\n'

for nb_path in sorted(pathlib.Path('notebooks').glob('*.ipynb')):
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    for cell in nb['cells']:
        if cell['cell_type'] != 'code': continue
        full = ''.join(cell['source'])
        if '_D = (' not in full: continue
        start = full.index('_D = (\n') + len('_D = (\n')
        end   = full.index(')\n', start)
        new_full = full[:start] + new_b64 + full[end:]
        lines = new_full.splitlines(keepends=True)
        if lines and lines[-1].endswith('\n'):
            lines[-1] = lines[-1].rstrip('\n')
        cell['source'] = lines
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(nb_path.name, 'updated')
"
```

---

## Hyperparameter Overrides

Each notebook has a dedicated cell (Cell 6) where you can override config
defaults without touching `src/config.py`:

```python
import src.config as cfg

# Quick test run — uncomment to reduce training time
# cfg.STAGE1_EPOCHS     = 2
# cfg.STAGE1_BATCH_SIZE = 16    # lower if OOM on smaller GPU
```

This is the recommended way to tune training for different GPU tiers.
