
# pwea/tests/test_prediction_service.py

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

from pwea.deployment.protocol import (
    recv_exact,
    recv_frame,
    send_frame,
)


class TestPredictionService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.socket_path = os.path.join(
            cls.tempdir.name, "pwea.sock"
        )

        cls.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "pwea.deployment.prediction_service",
                "--socket",
                cls.socket_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        deadline = time.monotonic() + 15.0

        while time.monotonic() < deadline:
            if cls.process.poll() is not None:
                raise RuntimeError(
                    "PWEA service exited during startup"
                )

            if os.path.exists(cls.socket_path):
                try:
                    probe = socket.socket(
                        socket.AF_UNIX,
                        socket.SOCK_STREAM,
                    )
                    probe.settimeout(0.2)
                    probe.connect(cls.socket_path)
                    probe.close()
                    break
                except OSError:
                    pass

            time.sleep(0.05)
        else:
            cls.process.terminate()
            raise RuntimeError("PWEA service did not become ready")

    @classmethod
    def tearDownClass(cls):
        if cls.process.poll() is None:
            cls.process.terminate()

            try:
                cls.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.process.kill()
                cls.process.wait(timeout=5)

        cls.tempdir.cleanup()

    def request(self, request):
        client = socket.socket(
            socket.AF_UNIX,
            socket.SOCK_STREAM,
        )
        client.settimeout(30.0)

        try:
            client.connect(self.socket_path)
            send_frame(
                client,
                json.dumps(request).encode("utf-8"),
            )

            header = json.loads(
                recv_frame(client).decode("utf-8")
            )

            payload_size = header.get("payload_bytes", 0)
            payload = (
                recv_exact(client, payload_size)
                if payload_size
                else b""
            )

            return header, payload
        finally:
            client.close()

    def test_prediction_response_shape_and_simplex(self):
        header, payload = self.request({
            "id": "test-001",
            "prompt": "Explain binary search in Python",
        })

        self.assertEqual(header["id"], "test-001")
        self.assertEqual(header["status"], "ok")
        self.assertEqual(header["shape"], [48, 128])
        self.assertEqual(header["dtype"], "float64-le")
        self.assertEqual(header["payload_bytes"], 48 * 128 * 8)
        self.assertEqual(len(payload), 48 * 128 * 8)

        matrix = np.frombuffer(
            payload, dtype="<f8"
        ).reshape(48, 128)

        self.assertTrue(np.all(np.isfinite(matrix)))
        self.assertTrue(np.all(matrix >= 0))
        self.assertTrue(
            np.allclose(matrix.sum(axis=1), 1.0, atol=1e-10)
        )

    def test_multiple_requests(self):
        prompts = [
            "Explain recursion",
            "Debug this Python function",
            "Convert a loop into a list comprehension",
        ]

        for index, prompt in enumerate(prompts):
            request_id = f"multi-{index}"

            header, payload = self.request({
                "id": request_id,
                "prompt": prompt,
            })

            self.assertEqual(header["id"], request_id)
            self.assertEqual(header["status"], "ok")
            self.assertEqual(len(payload), 48 * 128 * 8)

    def test_invalid_prompt_returns_error(self):
        header, payload = self.request({
            "id": "invalid-001",
            "prompt": "",
        })

        self.assertEqual(header["id"], "invalid-001")
        self.assertEqual(header["status"], "error")
        self.assertEqual(payload, b"")

    def test_malformed_json_returns_error(self):
        client = socket.socket(
            socket.AF_UNIX,
            socket.SOCK_STREAM,
        )
        client.settimeout(10.0)

        try:
            client.connect(self.socket_path)
            send_frame(client, b"{invalid-json")

            header = json.loads(
                recv_frame(client).decode("utf-8")
            )

            self.assertEqual(header["status"], "error")
            self.assertEqual(header["payload_bytes"], 0)
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
