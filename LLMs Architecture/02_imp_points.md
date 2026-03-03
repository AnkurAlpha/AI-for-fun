## What exactly is `block_size`?

**block_size = maximum sequence length (context length)** the model can handle at once.

* If `block_size = 128`, the model looks at **up to 128 tokens** as context.
* During training, we feed chunks of length ≤ 128.
* During generation, we usually keep only the **last 128 tokens** as context.

### Quick mapping of terms

* `embedding_dimension` / `n_embd` = size of vector per token (like 128, 384…)
* `block_size` = how many tokens in the context window (like 128, 256, 1024…)

