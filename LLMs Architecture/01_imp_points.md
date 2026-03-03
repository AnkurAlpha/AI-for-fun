# Important terms
* **B** = *Batch size* → number of sequences/examples processed at once
* **T** = *Time / Sequence length* → number of tokens in each sequence (often called `block_size`, `context_length`)
* **C** = *Channels / Embedding size* → the size of each token’s vector (often called `n_embd`, `d_model`)
* **V** = *Vocabulary size* → number of possible tokens (like 50k, 32k, etc.)

## Important shapes :
* `x.shape = (B, T)` → token IDs
* `tok = token_emb(x)` gives `(B, T, C)` → each token ID becomes an embedding vector of length **C**
* `pos_emb(arange(T))` gives `(T, C)` → one position vector per time step
* `h = tok + pos` → broadcast add → still `(B, T, C)`
* `lm_head(h)` gives `(B, T, V)` → for **each** token position, the model outputs **V logits** (scores for every possible next token)

At every position in every sequence, the model asks: **“Which of the V
tokens comes next?”** and logits `(V)` are the scores for that.

---
## What is Cross Entropy, and what does this line do?

### The line:

```python
loss = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))
```

### First: what are logits here?

* `logits` shape = `(1, 12, 30000)`
  So for each position in the sentence, the model outputs **30,000 raw scores** (one per possible next token).


* logits = `[score(token0), score(token1), ..., score(token29999)]`

### What cross entropy loss means (simple)

At each position, you know the correct next token (target ID).
Cross entropy punishes the model if it gives low probability to the correct token.

It does this by:

1. converting logits → probabilities using **softmax**
2. taking the probability assigned to the correct target token
3. taking **negative log** of that probability

#### For one training example (one position):

If correct token is `k`, then:

* `p_k = softmax(logits)[k]`
* loss = `-log(p_k)`

So:

* if model is confident and correct: `p_k ≈ 1` ⇒ loss ≈ 0
* if model is unsure/wrong: `p_k` small ⇒ loss big
