# for cleaning
import re
import unicodedata
# for file handling
import sys
from pathlib import Path
# for tokenizing
from tokenizers import ByteLevelBPETokenizer

TIMESTAMP_LINE = re.compile(r"^\s*(?:\d{1,2}:)?\d{1,2}:\d{2}\s*$")
# above line is of regular regression for
# identifying timestamp


def normalize_text(s: str) -> str:
    # Standard normaplization for making different unicodes identical
    s = unicodedata.normalize("NFKC", s)
    # keep the line structure , just trim the end
    return s.strip()


def clean_transcript(inp: Path, out: Path) -> None:
    removed = 0
    kept_lines = []
    try:
        for line in inp.read_text(encoding='utf-8', errors="replace").splitlines():
            if TIMESTAMP_LINE.match(line):
                removed += 1
                continue
            cleaned = normalize_text(line)
            if cleaned:  # drop empty lines created by trimming
                kept_lines.append(cleaned)
        out.write_text("\n".join(kept_lines)+"\n", encoding="utf-8")
    except PermissionError:
        print("Error: Permission Denied ", file=sys.stderr)
        sys.exit(1)
    except OSError:
        print("Error: OS Error", file=sys.stderr)
        sys.exit(1)

    print(f"input: {inp}")
    print(f"output: {out}")
    print(f"Removed timestamp only lines : {removed}")
    print(f"kept lines: {len(kept_lines)}")


if __name__ == "__main__":
    inp = Path("../data_raw/sample.txt")
    out = Path("../data_cleaned/sample.cleaned.txt")
    # checks :
    if not inp.exists():
        print(f"Erorr: Input file not found : {inp}", file=sys.stderr)
        sys.exit(1)
    if not inp.is_file():
        print(f"Errpr: Input is not a file: {inp}", file=sys.stderr)
        sys.exit(1)

    # Ensire outer directory exisists
    out.parent.mkdir(parents=True, exist_ok=True)
    clean_transcript(inp, out)

    # tokenizing
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        # files=["../data_cleaned/sample.cleaned.txt"],
        files=[str(out)],
        vocab_size=5000,  # becaus I have very small corpus ATM
        min_frequency=2,
        special_tokens=["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
    )

    out_dir = Path("../tokens") / "bpe_tokenizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(str(out_dir))

    test_string = "Hello! This is a small test."
    enc = tokenizer.encode(test_string)
    print(f"test string: {test_string}")
    print(f"encoded tokens: {enc.tokens}")
    print(f"encoded ids: {enc.ids}")
    print(f"decoded text : {tokenizer.decode(enc.ids)}")
