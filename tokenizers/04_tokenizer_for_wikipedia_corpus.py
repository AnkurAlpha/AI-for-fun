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
    print(f"shape: {(total_rows,n_cols)}")

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return s.strip()  # removes all the extra white spaces


if __name__ == "__main__":
    print("Please download the corpus from : \n\thttps://www.kaggle.com/datasets/gzdekzlkaya/wikipedia-text-corpus-for-nlp-and-llm-projects")
    print()
    print()
    outdir = Path("../tokens/") / "bpe_tokenizer"
    # if the parent doesn't exist then create
    outdir.mkdir(parents=True, exist_ok=True)
    inpdatadir = Path("../data_raw/")
    inpdata = inpdatadir / "wikipedia_text_corpus.csv"
    read_csv_peek(inpdata)
