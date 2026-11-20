"""Length-prefixed framing for the local PWEA prediction service."""

import struct

MAX_FRAME_BYTES = 1024 * 1024


def recv_exact(sock, size):
    if size < 0:
        raise ValueError("size cannot be negative")

    chunks = []
    remaining = size

    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("connection closed before frame was complete")
        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def send_frame(sock, payload):
    payload = bytes(payload)

    if not payload or len(payload) > MAX_FRAME_BYTES:
        raise ValueError("invalid frame size")

    sock.sendall(struct.pack("!I", len(payload)))
    sock.sendall(payload)


def recv_frame(sock):
    length = struct.unpack("!I", recv_exact(sock, 4))[0]

    if length == 0 or length > MAX_FRAME_BYTES:
        raise ValueError(f"invalid frame length: {length}")

    return recv_exact(sock, length)
