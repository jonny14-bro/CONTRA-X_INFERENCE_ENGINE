
# pwea/deployment/prediction_service.py

import argparse
import errno
import json
import os
import socket
import stat
import sys
from pathlib import Path

import numpy as np

from pwea.deployment.model_loader import PWEAPredictor
from pwea.deployment.protocol import recv_frame, send_frame


N_LAYER = 48
N_EXPERT = 128
PAYLOAD_BYTES = N_LAYER * N_EXPERT * 8


def error_response(request_id, message):
    return {
        "id": request_id,
        "status": "error",
        "error": str(message),
        "payload_bytes": 0,
    }


def handle_connection(conn, predictor):
    request_id = None

    try:
        raw = recv_frame(conn)
        request = json.loads(raw.decode("utf-8"))

        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")

        request_id = request.get("id")
        if (
            isinstance(request_id, bool)
            or not isinstance(request_id, (str, int))
        ):
            raise ValueError("request id must be a string or integer")

        prompt = request.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        matrix = predictor.predict(prompt)

        if matrix.shape != (N_LAYER, N_EXPERT):
            raise RuntimeError(
                f"unexpected prediction shape: {matrix.shape}"
            )

        if not np.all(np.isfinite(matrix)):
            raise RuntimeError("prediction contains non-finite values")

        if not np.all(matrix >= 0):
            raise RuntimeError("prediction contains negative values")

        if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-10):
            raise RuntimeError("prediction simplex validation failed")

        # Explicit little-endian float64, contiguous layer-major layout.
        payload = np.asarray(matrix, dtype="<f8", order="C").tobytes()

        if len(payload) != PAYLOAD_BYTES:
            raise RuntimeError("unexpected prediction payload size")

        header = {
            "id": request_id,
            "status": "ok",
            "shape": [N_LAYER, N_EXPERT],
            "dtype": "float64-le",
            "payload_bytes": len(payload),
        }

        send_frame(conn, json.dumps(header).encode("utf-8"))
        conn.sendall(payload)

    except Exception as exc:
        header = error_response(request_id, exc)

        try:
            send_frame(conn, json.dumps(header).encode("utf-8"))
        except Exception:
            pass


def prepare_socket_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(os.fsencode(path)) >= 108:
        raise ValueError("Unix socket path is too long")

    if not path.exists():
        return path

    info = path.lstat()

    if not stat.S_ISSOCK(info.st_mode):
        raise RuntimeError(
            f"Refusing to replace non-socket path: {path}"
        )

    # Do not remove a socket belonging to a running service.
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(0.25)

    try:
        probe.connect(str(path))
    except OSError as exc:
        if exc.errno not in (errno.ECONNREFUSED, errno.ENOENT):
            raise RuntimeError(
                f"Cannot establish whether socket is stale: {exc}"
            ) from exc
    else:
        raise RuntimeError(
            f"A PWEA service is already listening at {path}"
        )
    finally:
        probe.close()

    path.unlink()
    return path


def serve(socket_path):
    predictor = PWEAPredictor.load()
    print("[PWEA] Frozen deployment model loaded", file=sys.stderr)
    print(
        f"[PWEA] Model samples: "
        f"{predictor.manifest['model']['training_samples']}",
        file=sys.stderr,
    )

    path = prepare_socket_path(socket_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    old_umask = os.umask(0o177)
    try:
        server.bind(str(path))
    finally:
        os.umask(old_umask)

    os.chmod(path, 0o600)
    server.listen(8)

    print(f"[PWEA] Listening on {path}", file=sys.stderr)

    try:
        while True:
            conn, _ = server.accept()

            with conn:
                conn.settimeout(30.0)
                handle_connection(conn, predictor)

    except KeyboardInterrupt:
        print("\n[PWEA] Shutdown requested", file=sys.stderr)

    finally:
        server.close()

        try:
            path.unlink()
        except FileNotFoundError:
            pass

        print("[PWEA] Service stopped", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Persistent local PWEA prediction service"
    )
    parser.add_argument(
        "--socket",
        required=True,
        help="Unix-domain socket path",
    )

    args = parser.parse_args()
    serve(args.socket)


if __name__ == "__main__":
    main()
