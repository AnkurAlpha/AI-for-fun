#!/bin/env python3

import sys
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
base = Path("../tokens/") / "bpe_tokenizer_2"
vocab_path = base / "vocab.json"
merge_path = base / "merges.txt"

tokenizer = ByteLevelBPETokenizer(str(vocab_path), str(merge_path))


def print_token_analysis(s: str) -> None:
    out = tokenizer.encode(s)
    print(f"tokens: {out.tokens}")
    print(f"ids: {out.ids}")
    text_back = tokenizer.decode(out.ids)
    print(f"decoded text: {text_back}")


if __name__ == "__main__":
    print(f"argc = {len(sys.argv)}")
    print(f"argv = {sys.argv}")
    print()
    if len(sys.argv) == 1:
        print("""Error: pragram ran without argument
              run :
              .\\program <string>""", file=sys.stderr)
    text = " ".join(sys.argv[1:])
    print_token_analysis(text)
