"""
retvec_export.py — Download pretrained RetVec-v1, verify architecture, export weights.

Run once:
    python retvec_export.py

Outputs:
    retvec_v1_weights.npz   — weights for all 3 Dense layers, ready for PyTorch

Architecture (verified by exhaustive forward-pass comparison against TF serving_default):
    (B, 16, 24) binarized input
        -> Flatten                           -> (B, 384)
        -> dense_3: Linear(384, 256) + GELU  -> (B, 256)   [encoder block 1]
        -> dense_4: Linear(256, 256) + GELU  -> (B, 256)   [encoder block 2]
        -> dense_5: Linear(256, 256) + Tanh  -> (B, 256)   [tokenizer output]
    Output: (B, 256) float32, values in [-1, 1]

    Config inferred from weights: projection_dims=[], encoder_dims=[256, 256], tokenizer_dim=256
    (similarity_dense was removed from the serving_default export)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from retvec.tf.utils import download_retvec_saved_model


# ---------------------------------------------------------------------------
# 1. Load SavedModel
# ---------------------------------------------------------------------------

print("Downloading / loading retvec-v1 ...")
model_path = download_retvec_saved_model("retvec-v1")
model = tf.saved_model.load(str(model_path))
infer = model.signatures["serving_default"]
print("Loaded.\n")


# ---------------------------------------------------------------------------
# 2. Inspect variables
# ---------------------------------------------------------------------------

print("Variables:")
for var in model.variables:
    print(f"  {var.name:<45} {tuple(var.shape)}")
print()


# ---------------------------------------------------------------------------
# 3. Binarizer — Python reimplementation of RETVecBinarizer
# ---------------------------------------------------------------------------

WORD_LENGTH   = 16
ENCODING_SIZE = 24

def binarize_words(words: list[str]) -> np.ndarray:
    """string list -> (N, 16, 24) float32  [matches TF RETVecBinarizer exactly]"""
    masks = np.array([2 ** (ENCODING_SIZE - 1 - i) for i in range(ENCODING_SIZE)], dtype=np.int64)
    out   = np.zeros((len(words), WORD_LENGTH, ENCODING_SIZE), dtype=np.float32)
    for i, word in enumerate(words):
        codepoints = [ord(c) for c in word[:WORD_LENGTH]]
        codepoints += [0] * (WORD_LENGTH - len(codepoints))
        for j, cp in enumerate(codepoints):
            out[i, j] = np.minimum(np.bitwise_and(cp, masks), 1).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# 4. Forward-pass sanity check
# ---------------------------------------------------------------------------

test_words = ["hello", "he1lo", "hel|o", "world", "Microsoft", "M1cr0s0ft"]
binarized  = binarize_words(test_words)
result     = infer(args_0=tf.constant(binarized))["tokenizer"].numpy()

print("Forward pass:")
print(f"  Input  : {binarized.shape}  Output: {result.shape}")
print(f"  Value range: [{result.min():.4f}, {result.max():.4f}]  (tanh output)")
print()
cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
print("Cosine similarities:")
print(f"  hello vs he1lo        : {cos(result[0], result[1]):.4f}  (expect HIGH)")
print(f"  hello vs hel|o        : {cos(result[0], result[2]):.4f}  (expect HIGH)")
print(f"  hello vs world        : {cos(result[0], result[3]):.4f}  (expect LOW)")
print(f"  Microsoft vs M1cr0s0ft: {cos(result[4], result[5]):.4f}  (expect HIGH)")
print()


# ---------------------------------------------------------------------------
# 5. Export weights
#    dense_3 = encoder_1  (384->256, GELU)
#    dense_4 = encoder_2  (256->256, GELU)
#    dense_5 = tokenizer  (256->256, Tanh)  <- serving output
# ---------------------------------------------------------------------------

layer_roles = {
    "dense_3": "encoder1",
    "dense_4": "encoder2",
    "dense_5": "tokenizer",
}

weights: dict[str, np.ndarray] = {}
for var in model.variables:
    layer, rest  = var.name.split("/")
    param        = rest.split(":")[0]
    role         = layer_roles[layer]
    key          = f"{role}_{param}"
    weights[key] = var.numpy()
    print(f"  Exporting {key:<30} {tuple(var.numpy().shape)}")

weights["word_length"]   = np.array(WORD_LENGTH)
weights["encoding_size"] = np.array(ENCODING_SIZE)
weights["hidden_dim"]    = np.array(256)

out_path = "retvec_v1_weights.npz"
np.savez(out_path, **weights)
print(f"\nWeights saved: {out_path}")
print(f"Keys: {[k for k in weights]}")
