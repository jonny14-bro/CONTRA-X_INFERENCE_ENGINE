import json
import gguf
from pathlib import Path

model = Path.home() / "Downloads/Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf"
out = Path("benchmarks/real_activation_weight_replay_20261118")

weights = [
    json.loads(x)
    for x in (out / "weights.jsonl").read_text().splitlines()
]

reader = gguf.GGUFReader(str(model))
tensors = {t.name: t for t in reader.tensors}

print("===== GGUF READER OFFSETS =====")
print("reader.data_offset:", reader.data_offset)
print("model size:", model.stat().st_size)

with model.open("rb") as f, (out / "weights.bin").open("rb") as wf:
    for w in weights[:8]:
        t = tensors[w["tensor"]]
        expert = int(w["expert"])
        matrix_bytes = int(w["bytes"])

        expected = (
            t.data[expert].tobytes()
            if expert >= 0
            else t.data.tobytes()
        )[:32]

        replay_offset = int(w["replay_offset_bytes"])
        wf.seek(replay_offset)
        extracted = wf.read(32)

        tensor_offset = int(t.data_offset)

        candidates = {
            "t.data_offset": tensor_offset,
            "reader.data_offset + t.data_offset":
                int(reader.data_offset + tensor_offset),
        }

        if expert >= 0:
            candidates["t.data_offset + expert_stride"] = (
                tensor_offset + expert * matrix_bytes
            )
            candidates["reader.data_offset + t.data_offset + expert_stride"] = (
                int(reader.data_offset + tensor_offset)
                + expert * matrix_bytes
            )

        print("\nMATRIX:", w["weight_id"], w["tensor"], "expert:", expert)
        print("t.data_offset:", tensor_offset)
        print("reader.data_offset:", reader.data_offset)
        print("Expected ndarray bytes:", expected.hex())
        print("Extracted bytes:       ", extracted.hex())

        for label, offset in candidates.items():
            f.seek(offset)
            actual = f.read(32)
            print(
                label,
                "offset=", offset,
                "MATCH" if actual == expected else "mismatch",
                actual.hex()
            )
