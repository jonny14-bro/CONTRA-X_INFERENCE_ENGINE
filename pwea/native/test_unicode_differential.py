#!/usr/bin/env python3

import re
import subprocess

WORD = re.compile(r"\w")

def check_all_codepoints():
    print("Preparing Unicode test corpus...", flush=True)

    chars = [
        chr(cp)
        for cp in range(0x110000)
        if not 0xD800 <= cp <= 0xDFFF
    ]

    encoded = "".join(
        ch.encode("utf-8").hex() + "\n"
        for ch in chars
    )

    print(f"Sending {len(chars):,} codepoints...", flush=True)

    process = subprocess.run(
        ["/tmp/pwea_unicode_cli"],
        input=encoded,
        capture_output=True,
        text=True,
        check=True,
    )

    results = process.stdout.splitlines()

    if len(results) != len(chars):
        raise AssertionError(
            f"Expected {len(chars)} results, got {len(results)}"
        )

    print("Comparing results...", flush=True)

    for index, (char, result) in enumerate(zip(chars, results)):
        fields = result.split("\t")

        expected = (
            char.lower().encode("utf-8").hex(),
            str(int(bool(WORD.fullmatch(char)))),
            str(int(char.isspace())),
        )

        if len(fields) != 3 or tuple(fields) != expected:
            raise AssertionError(
                f"Mismatch U+{ord(char):04X}\n"
                f"Expected: {expected}\n"
                f"Actual:   {fields}"
            )

        if index and index % 100000 == 0:
            print(f"Checked {index:,}...", flush=True)

    print(f"PASS: {len(chars):,} Unicode scalar values", flush=True)

def check_contextual_strings():
    samples = [
        "ΟΣ", "ΟΣΑ", "ΟΣ Α", "AΣ", "AΣ\u0301",
        "AΣ\u0301A", "A\u0301Σ", "AΣ'", "AΣ'B",
        "İstanbul", "I İ ı", "ΜΆΙΟΣ",
        "Hello WORLD 123", "Café naïve",
        "नमस्ते दुनिया", "ಕನ್ನಡ ಭಾಷೆ",
        "Straße", "ẞ", "ﬃ", "AΣ\u0345", "AΣ\u200D",
    ]

    encoded = "".join(
        sample.encode("utf-8").hex() + "\n"
        for sample in samples
    )

    process = subprocess.run(
        ["/tmp/pwea_unicode_cli"],
        input=encoded,
        capture_output=True,
        text=True,
        check=True,
    )

    results = process.stdout.splitlines()

    for sample, actual in zip(samples, results):
        actual = actual.split("\t", 1)[0]
        expected = sample.lower().encode("utf-8").hex()

        if actual != expected:
            raise AssertionError(
                f"Mismatch for {sample!r}: {actual} != {expected}"
            )

    print(f"PASS: {len(samples)} contextual strings", flush=True)

def check_invalid_utf8():
    invalid = ["80", "c0af", "eda080", "f4908080", "e282", "f09f"]

    encoded = "".join(value + "\n" for value in invalid)

    process = subprocess.run(
        ["/tmp/pwea_unicode_cli"],
        input=encoded,
        capture_output=True,
        text=True,
        check=True,
    )

    results = process.stdout.splitlines()

    if results != ["ERROR"] * len(invalid):
        raise AssertionError(f"Invalid UTF-8 test failed: {results}")

    print(f"PASS: {len(invalid)} malformed UTF-8 cases", flush=True)

if __name__ == "__main__":
    check_all_codepoints()
    check_contextual_strings()
    check_invalid_utf8()
    print("\nALL DIFFERENTIAL TESTS PASSED", flush=True)
