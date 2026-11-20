import json
import shutil
import sys
from pathlib import Path
from collections import Counter

import gguf

MODEL = Path.home() / "Downloads/Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf"
SOURCE = Path("benchmarks/real_activations_q8k_full_20261118")
DEST = Path("benchmarks/real_activation_weight_replay_20261118")

ACTIVATION_FILE = SOURCE / "activations.f32"
ACTIVATION_MANIFEST = SOURCE / "manifest.jsonl"

RESERVE_BYTES = 5 * 1024**3

TENSOR_NAMES = {
    "attn_q": lambda l: f"blk.{l}.attn_q.weight",
    "attn_k": lambda l: f"blk.{l}.attn_k.weight",
    "attn_v": lambda l: f"blk.{l}.attn_v.weight",
    "attn_output": lambda l: f"blk.{l}.attn_output.weight",
    "moe_gate": lambda l: f"blk.{l}.ffn_gate_exps.weight",
    "moe_up": lambda l: f"blk.{l}.ffn_up_exps.weight",
    "moe_down": lambda l: f"blk.{l}.ffn_down_exps.weight",
}


def fail(message):
    print("ERROR:", message, file=sys.stderr)
    raise SystemExit(1)


def load_records():
    if not MODEL.is_file():
        fail(f"Model not found: {MODEL}")
    if not ACTIVATION_FILE.is_file():
        fail(f"Activation data not found: {ACTIVATION_FILE}")
    if not ACTIVATION_MANIFEST.is_file():
        fail(f"Activation manifest not found: {ACTIVATION_MANIFEST}")
    if DEST.exists():
        fail(f"Destination already exists; refusing to overwrite: {DEST}")

    records = [
        json.loads(line)
        for line in ACTIVATION_MANIFEST.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        fail("Activation manifest is empty")

    return records


def main():
    records = load_records()

    activation_size = ACTIVATION_FILE.stat().st_size
    disk = shutil.disk_usage(DEST.parent)

    print("===== EXTRACTION PREFLIGHT =====")
    print("Activation records:", len(records))
    print("Activation bytes:", activation_size)
    print("Available disk bytes:", disk.free)
    print("Required reserve bytes:", RESERVE_BYTES)

    reader = gguf.GGUFReader(str(MODEL))
    tensors = {t.name: t for t in reader.tensors}

    output_name = (
        "output.weight"
        if "output.weight" in tensors
        else "token_embd.weight"
    )

    matrices = {}
    replay_rows = []
    errors = []
    op_counts = Counter()
    expected_total_bytes = 0

    # First pass: validate every mapping and calculate unique storage.
    for index, r in enumerate(records):
        op = r.get("operation")
        layer = int(r.get("layer", -1))

        if op == "output_projection":
            name = output_name
        elif op in TENSOR_NAMES:
            name = TENSOR_NAMES[op](layer)
        else:
            errors.append((index, "unknown operation", op))
            continue

        t = tensors.get(name)
        if t is None:
            errors.append((index, "missing tensor", name))
            continue

        shape = [int(x) for x in t.shape]
        typ = int(t.tensor_type)

        if typ not in (13, 14) or typ != int(r["type"]):
            errors.append((index, "weight type mismatch", name, typ, r["type"]))
            continue

        k = int(r["k"])
        nb = int(r["nb"])
        ldx = int(r["ldx"])
        elements = int(r["elements"])
        act_offset = int(r["offset_bytes"])

        if len(shape) not in (2, 3):
            errors.append((index, "unexpected tensor rank", name, shape))
            continue

        if shape[0] != k:
            errors.append((index, "input dimension mismatch", name, shape[0], k))
            continue

        if nb <= 0 or k <= 0 or ldx < k:
            errors.append((index, "invalid activation dimensions", r))
            continue

        if elements != nb * k:
            errors.append((index, "activation element count mismatch",
                           elements, nb * k))
            continue

        if act_offset < 0 or act_offset + elements * 4 > activation_size:
            errors.append((index, "activation payload out of bounds", r))
            continue

        if op.startswith("moe_"):
            if len(shape) != 3:
                errors.append((index, "expert tensor is not 3D", name, shape))
                continue

            expert = int(r["expert"])
            if not 0 <= expert < shape[2]:
                errors.append((index, "expert index out of bounds", name, expert))
                continue

            matrix = t.data[expert]
            key = (name, expert)
        else:
            if len(shape) != 2:
                errors.append((index, "dense tensor is not 2D", name, shape))
                continue

            expert = -1
            matrix = t.data
            key = (name,)

        if matrix.ndim != 2 or matrix.dtype.name != "uint8":
            errors.append((index, "unexpected packed matrix representation",
                           name, matrix.shape, str(matrix.dtype)))
            continue

        matrix_rows = int(matrix.shape[0])
        row_bytes = int(matrix.shape[1])
        matrix_bytes = int(matrix.nbytes)

        if matrix_rows != shape[1] or matrix_bytes != matrix_rows * row_bytes:
            errors.append((index, "physical/logical matrix layout mismatch",
                           name, shape, matrix.shape, matrix_bytes))
            continue

        if key not in matrices:
            tensor_absolute_offset = int(t.data_offset)
            matrix_offset = (
                tensor_absolute_offset + expert * matrix_bytes
                if expert >= 0 else tensor_absolute_offset
            )

            weight_id = f"W{len(matrices):05d}"
            matrices[key] = {
                "weight_id": weight_id,
                "tensor": name,
                "expert": expert,
                "type": typ,
                "shape": shape,
                "rows": matrix_rows,
                "row_bytes": row_bytes,
                "bytes": matrix_bytes,
                "gguf_tensor_offset": tensor_absolute_offset,
                "gguf_matrix_offset": matrix_offset,
                "_matrix": matrix,
            }
            expected_total_bytes += matrix_bytes

        replay_rows.append({
            "record": int(r["record"]),
            "operation": op,
            "layer": layer,
            "expert": expert,
            "k": k,
            "nb": nb,
            "ldx_original": ldx,
            "elements": elements,
            "activation_offset_bytes": act_offset,
            "activation_bytes": elements * 4,
            "weight_id": matrices[key]["weight_id"],
            "weight_type": typ,
            "weight_rows": matrix_rows,
            "weight_row_bytes": row_bytes,
        })
        op_counts[op] += 1

    if errors:
        print("Mapping errors:", len(errors))
        for e in errors[:30]:
            print(e)
        fail("Extraction aborted because mapping validation failed")

    required = expected_total_bytes + RESERVE_BYTES
    disk = shutil.disk_usage(DEST.parent)

    print("Unique matrices:", len(matrices))
    print("Unique weight bytes:", expected_total_bytes)
    print("Unique weight GiB:", round(expected_total_bytes / 1024**3, 3))
    print("Required with reserve GiB:", round(required / 1024**3, 3))
    print("Available GiB:", round(disk.free / 1024**3, 3))

    if disk.free < required:
        fail("Insufficient disk space for extraction and safety reserve")

    # Destination is created only after all preflight checks pass.
    DEST.mkdir(parents=True, exist_ok=False)

    weights_path = DEST / "weights.bin"
    weights_manifest_path = DEST / "weights.jsonl"
    replay_manifest_path = DEST / "replay.jsonl"

    written = 0
    weight_manifest_rows = []

    try:
        with weights_path.open("xb") as wf:
            for key, item in matrices.items():
                matrix = item.pop("_matrix")
                payload = matrix.tobytes(order="C")

                if len(payload) != item["bytes"]:
                    fail(f"Byte length changed for {item['weight_id']}")

                item["replay_offset_bytes"] = written
                wf.write(payload)
                written += len(payload)

                weight_manifest_rows.append(item)

        with weights_manifest_path.open("x") as f:
            for item in weight_manifest_rows:
                f.write(json.dumps(item, separators=(",", ":")) + "\n")

        with replay_manifest_path.open("x") as f:
            for item in replay_rows:
                f.write(json.dumps(item, separators=(",", ":")) + "\n")

    except Exception:
        print(
            "\nExtraction interrupted. Partial output is preserved at:",
            DEST,
            file=sys.stderr,
        )
        raise

    actual_size = weights_path.stat().st_size
    if actual_size != expected_total_bytes or written != expected_total_bytes:
        fail(
            f"Final weight file size mismatch: "
            f"expected={expected_total_bytes}, actual={actual_size}"
        )

    # Verify all replay references resolve to an extracted matrix.
    valid_ids = {x["weight_id"] for x in weight_manifest_rows}
    unresolved = [
        r for r in replay_rows if r["weight_id"] not in valid_ids
    ]

    if unresolved:
        fail(f"Unresolved replay references: {len(unresolved)}")

    print("\n===== EXTRACTION SUMMARY =====")
    print("Activation records:", len(replay_rows))
    print("Unique matrices extracted:", len(weight_manifest_rows))
    print("Weight bytes written:", written)
    print("Weight file GiB:", round(actual_size / 1024**3, 3))
    print("Unresolved replay references:", len(unresolved))

    print("\n===== OPERATION COUNTS =====")
    for op, count in sorted(op_counts.items()):
        print(f"{op:20s}: {count}")

    print("\n===== OUTPUT FILES =====")
    for path in (
        weights_path,
        weights_manifest_path,
        replay_manifest_path,
    ):
        print(path, path.stat().st_size, "bytes")

    print("\nEXTRACTION:", "PASS")


if __name__ == "__main__":
    main()
