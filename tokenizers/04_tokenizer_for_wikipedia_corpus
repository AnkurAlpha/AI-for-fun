#!/usr/bin/env python3

# for data analyzing and extraction
import csv
import pandas as pd
# for cleaning
import re
import unicodedata
# for file handling
import sys
from pathlib import Path
# for tokanization
from tokenizers import ByteLevelBPETokenizer

MANY_NEW_BLANK_LINES = re.compile(r"\n{3,}")
# if 3 or more blank lines are there then handle it


def bump_csv_limit() -> None:
    try:
        csv.field_size_limit(sys.maxsize)  # first set it to max size
    except OverflowError:
        max_size = sys.maxsize
        while True:
            try:
                csv.field_size_limit(max_size)
                break
            except OverflowError:
                max_size //= 10


def read_csv_peek(path: str) -> None:
    bump_csv_limit()
    # df = pd.read_csv(
    #     s,
    #     quotechar='"',
    #     # escapechar="\\", # use only if needed
    #     engine="python"
    # )
    print("Small peek of data : ")
    print()
    for chunk in pd.read_csv(path, quotechar='"',
                             engine="python",
                             chunksize=50_000
                             ):
        print(chunk.head(10))
        break
    total_rows = 0
    n_cols = None
    for chunk in pd.read_csv(path, engine="python",
                             quotechar='"', chunksize=50_000):
        total_rows += len(chunk)
        if n_cols is None:
            n_cols = chunk.shape[1]
    print(f"rows: {total_rows}")
    print(f"cols: {n_cols}")
    print(f"shape: {(total_rows, n_cols)}")


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # "\r\n" converted to "\n" (windows)
    # "\r" converted to "\n" (mac)
    s = MANY_NEW_BLANK_LINES.sub("\n\n", s)
    # to preven huge blank gaps
    return s.strip()  # removes all the extra white spaces


def extract_clean_text(csv_path: Path, out_txt: Path,
                       chunksize: int = 50_000) -> None:
    read_csv_peek(csv_path)
    bump_csv_limit()
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    rows_read = 0
    docs_written = 0

    with out_txt.open("w", encoding="utf-8") as f_out:
        for chunk in pd.read_csv(
                csv_path,
                engine="python",
                quotechar='"',
                usecols=["text"],
                chunksize=chunksize,
        ):
            rows_read += len(chunk)
            for s in chunk["text"].fillna(""):
                # chunk["text"]: to see all the text columns
                # .fillna(""): to fill all the values which
                # are NaN with empty string, (prevents crashing)
                s = normalize_text(s)
                if s:  # if s is not empty
                    f_out.write(s)
                    f_out.write("\n\n")  # separete documents clearly
                    docs_written += 1
            if rows_read % (chunksize * 2) == 0:
                print(f"processed rows: {  # just for giving stats
                      rows_read} | written_docks: {docs_written}")
        print(f"done. Rows read: {rows_read}, docs written: {docs_written}")
        print(f"saved cleaned text to: {out_txt}")


def ignite_training(inp: Path, out: Path) -> None:
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(inp)],
        vocab_size=30_000,
        min_frequency=2,
        special_tokens=["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
    )
    tokenizer.save_model(str(out))
    test_string = "Hello! This is small test by AnkurAlpha"
    enc = tokenizer.encode(test_string)
    print(f"test string: {test_string}")
    print(f"encoded tokens: {enc.tokens}")
    print(f"encoded ids: {enc.ids}")
    print(f"decoded text : {tokenizer.decode(enc.ids)}")
    print(f"tokenizers saved: {out}")


if __name__ == "__main__":
    print("Please download the corpus from : \
          \n\thttps://www.kaggle.com/datasets/gzdekzlkaya/wikipedia-" +
          "text-corpus-for-nlp-and-llm-projects")
    print()
    print()
    out_token_dir = Path("../tokens/") / "bpe_tokenizer_2"
    out_token_dir.mkdir(parents=True, exist_ok=True)
    # if the parent doesn't exist then create
    out_text = Path("../data_cleaned/") / "wikipedia.cleaned.txt"
    out_text.parent.mkdir(parents=True, exist_ok=True)
    inpdatadir = Path("../data_raw/")
    inpdata = inpdatadir / "wikipedia_text_corpus.csv"
    # read_csv_peek(inpdata)
    extract_clean_text(inpdata, out_text)
    final_tokens = Path("../tokens/") / "bpe_tokenizer_2"
    final_tokens.mkdir(parents=True, exist_ok=True)
    ignite_training(out_text, final_tokens)
