import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from pwea.predictor.text_features import transform_text_variant

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures"

cases = [
    ("ascii", "Hello world, this is a PWEA test."),
    ("unicode_latin", "Café naïve Straße İstanbul"),
    ("greek", "ΟΣ ΟΣΑ ΜΆΙΟΣ"),
    ("indic", "नमस्ते दुनिया ಕನ್ನಡ ಭಾಷೆ"),
    ("punctuation", "C++: feature-hashing, n-grams! (test_123)"),
    ("whitespace", "  multiple\tspaces\nand\u00a0unicode spaces  "),
    ("mixed", "AI/ML expert routing: Layer 48, Expert 127."),
    ("empty_like", "   punctuation !!! ___ 123 "),
]

records = []

binary_path = OUT / "pwea_feature_reference.bin"

with binary_path.open("wb") as f:
    f.write(b"PWEAFEAT")
    f.write(struct.pack("<II", len(cases), 32768))

    for name, prompt in cases:
        vector = transform_text_variant(
            prompt,
            "balanced_hybrid"
        )

        vector = np.asarray(vector, dtype="<f8").reshape(-1)

        if vector.shape != (32768,):
            raise RuntimeError(
                f"{name}: unexpected shape {vector.shape}"
            )

        if not np.isfinite(vector).all():
            raise RuntimeError(f"{name}: non-finite features")

        prompt_bytes = prompt.encode("utf-8")

        f.write(struct.pack("<I", len(prompt_bytes)))
        f.write(prompt_bytes)
        f.write(vector.tobytes())

        records.append({
            "name": name,
            "prompt": prompt,
            "dimensions": len(vector),
            "dtype": "float64-le",
            "word_norm": float(np.linalg.norm(vector[:16384])),
            "char_norm": float(np.linalg.norm(vector[16384:])),
            "sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
        })

manifest = {
    "format": "PWEAFEAT",
    "feature_variant": "balanced_hybrid",
    "feature_dimensions": 32768,
    "word_dimensions": 16384,
    "character_dimensions": 16384,
    "cases": records,
}

manifest_path = OUT / "pwea_feature_reference.json"
manifest_path.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print("Reference fixture generated.")
print("Cases:", len(records))
print("Binary:", binary_path)
print("Manifest:", manifest_path)

for item in records:
    print(
        item["name"],
        "word_norm=", item["word_norm"],
        "char_norm=", item["char_norm"],
        "sha256=", item["sha256"],
    )
