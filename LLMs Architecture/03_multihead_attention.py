#!/bin/env python3

# default modules
import sys
# for tokenizing
from tokenizers import ByteLevelBPETokenizer
# file paths with ease and clarity
from pathlib import Path
# real AI stuff : importing modules required for building LLMs
import torch
from torch import nn as nn
from torch.nn import functional as F

# please first generate the tokens from the code in tokenizer first
base_path = Path("../tokens/") / "bpe_tokenizer_2"
vocab_path = base_path / "vocab.json"
merges_path = base_path / "merges.txt"

tokenizer = ByteLevelBPETokenizer(str(vocab_path), str(merges_path))


def print_token_analysis(s: str) -> None:
    out = tokenizer.encode(s)
    print(f"tokens: {out.tokens}")
    print(f"ids: {out.ids}")
    text_back = tokenizer.decode(out.ids)
    print(f"Decoded words: {text_back}")


# vocab + embedding_dimension
vocab_size = tokenizer.get_vocab_size()

# hyperparameter: the size of the vector used to represent each token
embedding_dimension = 128
# each token ID gets mapped to a vector of length 128

# number of tokens that is allowed to be looked by LLM while predicting next
block_size = 128

# My embedding table has:
# vocab_size * embedding_dimension parameters
# 30000 * 128 = 3,840,000 parameters just for token embeddings.


class Head (nn.Module):  # class of one single head

    def __init__(self, C: int, head_size: int, block_size: int):
        super().__init__()
        self.key = nn.Linear(C, head_size, bias=False)
        self.query = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)
        self.register_buffer("tril", torch.tril(
            torch.ones(block_size, block_size)))

    def forward(self, x):
        # define computeation here
        B, T, _ = x.shape
        k = self.key(x)             # (B,T,hs)
        q = self.query(x)           # (B,T,hs)
        v = self.value(x)           # (B,T,hs)
        # hs = head size = C/n_head = 128/8 = 16
        weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B,T,T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # masked all the future tokens
        weights = F.softmax(weights, dim=-1)

        out = weights @ v  # (B,T,hs)
        return out


class MultiheadAttention(nn.Module):
    def __init__(self, C: int, n_head: int, block_size: int):
        super().__init__()
        assert C % n_head == 0  # safety check
        head_size = C // n_head
        self.heads = nn.ModuleList(
            [Head(C, head_size, block_size) for _ in range(n_head)])
        self.proj = nn.Linear(C, C)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B,T,C)
        out = self.proj(out)
        return out


# making the first LLM component class !!!
class AttentionLLM(nn.Module):
    # small LLM
    # Goal (B,T) tokens IDs -> output (B,T,vocab_size) logits

    # it needs only two learnable parts :
    # 1. Embedding: vocab_size -> embedding_dimension
    # 2. Linear head: embedding_dimension -> vocab_size

    def __init__(self, vocab_size: int, embedding_dimension: int):
        super().__init__()
        print("Running: __init__()")
        # define layers here
        self.token_emb = nn.Embedding(vocab_size, embedding_dimension)
        # above are token embeddings, made through lookup table
        self.pos_emb = nn.Embedding(block_size, embedding_dimension)
        # above is for positional embeddings
        self.lm_head = nn.Linear(embedding_dimension, vocab_size)
        # nn.Embedding & nn.Linear stores matrix of learnable weights

        # implementing multihead attention
        self.attn = MultiheadAttention(
            embedding_dimension, n_head=8, block_size=block_size)

    def forward(self, x, targets=None):
        # define computeation here
        print("Running: forward()")
        B, T = x.shape
        assert T <= block_size
        # safety check , T -> current sequence length,
        # T = current sequence length
        tok = self.token_emb(x)         # (B,T,C)
        pos = self.pos_emb(torch.arange(T, device=x.device))  # (T,C)
        # torch.arange() Produces : [0,1,2,3, ..., T-1 ] (positional)
        # self.pos_emb is also an embedding table: nn.Embedding(block_size,C)
        h = tok + pos                   # (B,T,C)
        h = self.attn(h)
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))
        with torch.no_grad():  # for removing unnecessary graph tracking
            # for analysis:
            top = torch.topk(logits[0, -1], k=5)  # last token postion, top 5
            print(f"top5 ids: {top.indices.tolist()}")
            print(f"tpo5 logits: {top.values.tolist()}")
            decoded = tokenizer.decode(top.indices.tolist())
            print(f"Tokenizer decode from forward(): {decoded}")
            # this is just for looking inside a training process, it
            # may look really bad but it gives, interesting insights
        return logits, loss


if __name__ == "__main__":
    # print("Basic LLM")
    # print_token_analysis(str(" ".join(sys.argv[1:])))
    # get the text , join is important for converting to text
    text = str(" ".join(sys.argv[1:]))

    # small outputs to get to know from what are we dealing with
    print(f"inputted text : {text} ")
    ids = tokenizer.encode(text).ids
    x = torch.tensor([ids], dtype=torch.long)  # get all the ID's tensor
    print(f"shape: {x.shape}")
    print(f"(B,T): {torch.tensor([ids]).shape}")

    # setting up the params
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device being used : {device}")

    # finally starting for the real thing
    x = x.to(device)
    # for preventing from tensor being on CPU and model
    # on GPU both should be on CPU or GPU

    # crating targets :
    # after creating x , create targets by shifting them by 1
    x_in = x[:, :-1]
    targets = x[:, 1:]

    model = AttentionLLM(vocab_size, embedding_dimension).to(device)
    # first time run : build the neural network -> __init__

    # logits, loss = model(x)  # computeation : -> forward
    logits, loss = model(x_in, targets)
    print(f"logits: {logits.shape}")  # [B,T-1,vocab_size]
    print(f"loss: {loss}")
    # logits = [score(token0), score(token1), ..., score(token29999)]
    # it is like a score given to each 30000 ids
