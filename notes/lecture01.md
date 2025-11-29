# Lecture-01 Introduction & Tokenizer

[TOC]

## Current Landscape

### Neural ingredinets
- [Transformer architecture (for machine translation)](https://arxiv.org/pdf/1706.03762.pdf)
- Mixture of experts, [link-moe_2017](https://arxiv.org/pdf/1701.06538.pdf)
- Model parallelism: [gpipe_2018](https://arxiv.org/pdf/1811.06965.pdf), [zero_2019](https://arxiv.org/abs/1910.02054), [megatron_lm_2019](https://arxiv.org/pdf/1909.08053.pdf), [megatron-deepspeed_2023](https://github.com/deepspeedai/Megatron-DeepSpeed)

### Open Models & Datasets

- Open-source models (e.g., [OLMo](https://arxiv.org/pdf/2402.00838.pdf)): weights and data available, paper with most details (but not necessarily the rationale, failed experiments)
- Open-weight models (e.g., DeepSeek): weights available, paper with architecture details, some training details, no data details
  - [deepseek-v3](https://arxiv.org/pdf/2412.19437.pdf)

### Today's frontier models

- [OpenAI's o3](https://openai.com/index/openai-o3-mini/)
- [xAI's Grok 3](https://x.ai/news/grok-3)
- [Anthropic's Claude Sonnet 3.7](https://www.anthropic.com/news/claude-3-7-sonnet)
- [Google's Gemini 2.5](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/)
- [Meta's Llama 3.3](https://ai.meta.com/blog/meta-llama-3/)
- [DeepSeek's r1](https://arxiv.org/pdf/2501.12948.pdf)
- [Alibaba's Qwen 2.5 Max](https://qwenlm.github.io/blog/qwen2.5-max/)
- [Tencent's Hunyuan-T1](https://tencent.github.io/llm.hunyuan.T1/README_EN.html)

## Course Overview
All information online: https://stanford-cs336.github.io/spring2025/


### 1. basics
- Goal: get a basic version of the full pipeline working
- Components: tokenization, model architecture, training
1. Tokenization
    - Tokenizers convert between strings and sequences of integers (tokens)
    - Intuition: break up string into popular segments
    - This course: Byte-Pair Encoding (BPE) tokenizer [link-sennrich_2016](https://arxiv.org/abs/1508.07909)
    - Tokenizer-free approaches:[byt5](), [megabyte](), [blt](), [tfree]()
2. Architecture & Variants
    - Starting point: original Transformer, [link-transfomer_2017](https://arxiv.org/pdf/1706.03762.pdf) ![transformer-architecture](../images/transformer-architecture.png)
    - Activation functions: ReLU, SwiGLU [link-shazeer_2020](https://arxiv.org/pdf/2002.05202.pdf)
    - Positional encodings: sinusoidal, RoPE [link-rope_2021](https://arxiv.org/pdf/2104.09864.pdf)
    - Normalization: LayerNorm, RMSNorm [link-layernorm_2016](https://arxiv.org/pdf/1607.06450.pdf), [link-rms_norm_2019](https://arxiv.org/pdf/1909.11556.pdf)
    - Placement of normalization: pre-norm versus post-norm [link-pre_post_norm_2020](https://arxiv.org/pdf/2002.04745.pdf)
    - MLP: dense, mixture of experts [link-moe_2017](https://arxiv.org/pdf/1701.06538.pdf)
    - Attention: full, sliding window, linear [link-mistral_7b](https://arxiv.org/pdf/2006.16236.pdf), **Kimi-Linear**[link-kimi_2025](https://arxiv.org/abs/2510.26692v1)
    - Lower-dimensional attention: group-query attention (GQA), multi-head latent attention (MLA) [link-gqa](https://arxiv.org/pdf/2102.11972.pdf), [link-mla](https://arxiv.org/pdf/2102.11972.pdf)
    - State-space models: Hyena [link-hyena](https://arxiv.org/pdf/2302.10866.pdf)
3. Training
    - Optimizer (e.g., [AdamW](https://arxiv.org/pdf/1711.05101.pdf), [Muon](https://arxiv.org/pdf/2002.04745.pdf), [SOAP](https://arxiv.org/pdf/2002.04745.pdf))
    - Learning rate schedule (e.g., cosine, WSD) [link-cosine_learning_rate_2017](https://arxiv.org/pdf/1608.03983.pdf), [link-wsd_2024](https://arxiv.org/pdf/2002.04745.pdf)
    - Batch size (e..g, critical batch size) [link-large_batch_training_2018](https://arxiv.org/pdf/2002.04745.pdf)
    - Regularization (e.g., dropout, weight decay)
    - Hyperparameters (number of heads, hidden dimensions)
4. Assignment-1
    - [github](https://github.com/stanford-cs336/assignment1-basics), [pdf](https://stanford-cs336.github.io/spring2025/assignments/assignment1-basics.pdf)
    - Implement BPE tokenizer
    - Implement Transformer, cross-entropy loss, AdamW optimizer, training loop
    - Train on TinyStories and OpenWebText
    - Leaderboard: minimize OpenWebText perplexity given 90 minutes on a H100, [last year's leaderboard](https://github.com/stanford-cs336/spring2024-assignment1-basics-leaderboard)
### 2. systems
- Goal : squeeze the most out of the hardware
- Components: kernels, parallelism, inference
1. Kernels
    1. Analogy: warehouse : DRAM :: factory : SRAM
    2. Trick: organize computation to maximize utilization of GPUs by minimizing data movement
    3. Write kernels in CUDA/**Triton**/CUTLASS/ThunderKittens
2. Parallelism
    - Data movement between GPUs is even slower, but same 'minimize data movement' principle holds
    - Use collective operations (e.g., gather, reduce, all-reduce)
    - Shard (parameters, activations, gradients, optimizer states) across GPUs
    - How to split computation: {data, tensor, pipeline, sequence} parallelism
3. Inference
    - Goal: generate tokens given a prompt (needed to actually use models!)
    - Inference is also needed for reinforcement learning, test-time compute, evaluation
    - Globally, inference compute (every use) exceeds training compute (one-time cost)
    - Two phases: **prefill** and **decode**
      - Prefill (similar to training): tokens are given, can process all at once (compute-bound)
      - Decode: need to generate one token at a time (memory-bound)
    - Methods to speed up decoding:
      - Use cheaper model (via model pruning, quantization, distillation)
      - Speculative decoding: use a cheaper "draft" model to generate multiple tokens, then use the full model to score in parallel (exact decoding!)
4. Assignment-2
    - [GitHub from 2024](https://github.com/stanford-cs336/spring2024-assignment2-systems), [PDF from 2024](https://github.com/stanford-cs336/spring2024-assignment2-systems/blob/master/cs336_spring2024_assignment2_systems.pdf)
    - Implement a fused RMSNorm kernel in Triton
    - Implement distributed data parallel training
    - Implement optimizer state sharding
    - **Benchmark** and profile the implementations
### 3. scaling laws
- Goal: do experiments at small scale, predict hyperparameters/loss at large scale
- Question: given a FLOPs budget ($C$), use a bigger model ($N$) or train on more tokens ($D$)?
- Compute-optimal scaling laws: [kaplan_scaling_laws_2020](),![chinchilla](../images/chinchilla-isoflop.png "chinchilla-isoflop")
- TL;DR: $D^* = 20 N^*$ (e.g., 1.4B parameter model should be trained on 28B tokens)
- But this doesn't take into account inference costs!
- Assignment-3
    - [GitHub from 2024](https://github.com/stanford-cs336/spring2024-assignment3-scaling), [PDF from 2024](https://github.com/stanford-cs336/spring2024-assignment3-scaling/blob/master/cs336_spring2024_assignment3_scaling.pdf)
    - We define a training API (hyperparameters -> loss) based on previous runs
    - Submit "training jobs" (under a FLOPs budget) and gather data points
    - Fit a scaling law to the data points
    - Submit predictions for scaled up hyperparameters
    - Leaderboard: minimize loss given FLOPs budget
### 4. data
- Question: What capabilities do we want the model to have?
- Multilingual? Code? Math
1. **Evaluation**
    1. Perplexity: textbook evaluation for language models
    2. Standardized testing (e.g., MMLU, HellaSwag, GSM8K)
    3. Instruction following (e.g., AlpacaEval, IFEval, WildBench)
    4. Scaling test-time compute: chain-of-thought, ensembling
    5. LM-as-a-judge: evaluate generative tasks
    6. Full system: RAG, agents
2. Data curation
    - Data does not just fall from the sky.
    - Sources: webpages crawled from the Internet, books, arXiv papers, GitHub code, etc.
    - Appeal to fair use to train on copyright data? [paper](https://arxiv.org/pdf/2303.15715.pdf)
    - Might have to license data (e.g., Google with Reddit data) [link](https://www.reuters.com/technology/reddit-ai-content-licensing-deal-with-google-sources-say-2024-02-22/)
    - Formats: HTML, PDF, directories (not text!)
3. Data processing
    1. Transformation: convert HTML/PDF to text (preserve content, some structure, rewriting)
    2. Filtering: keep high quality data, remove harmful content (via classifiers)
    3. Deduplication: save compute, avoid memorization; use Bloom filters or MinHash
4. Assignment-4
    1. [GitHub from 2024](https://github.com/stanford-cs336/spring2024-assignment4-data), [PDF from 2024](https://github.com/stanford-cs336/spring2024-assignment4-data/blob/master/cs336_spring2024_assignment4_data.pdf)
    2. Convert Common Crawl HTML to text
    3. Train classifiers to filter for quality and harmful content
    4. Deduplication using MinHash
    5. Leaderboard: minimize perplexity given token budget
### 5. alignment
1. So far, a **base model** is raw potential, very good at completing the next token. Alignment makes the model actually useful.
2. Goals of alignment:
    1. Get the language model to follow instructions
    2. Tune the style (format, length, tone, etc.)
    3. Incorporate safety (e.g., refusals to answer harmful questions)
3. Two phases:
    1. **supervised_finetuning**
        - Intuition: base model already has the skills, just need few examples to surface them.[link-lima](https://arxiv.org/pdf/2305.11206.pdf")
        - Supervised learning: fine-tune model to maximize p(response | prompt).
        - Instruction data: (prompt, response) pairs & Example
        - Example:
            ```json
            [
                {"role" : "system", "content" : "You are a helpful assistant."},
                {"role" : "user", "content" : "What is the capital of France?"}, 
                {"role" :  "assistant", "content" : "Paris."}
            ]
            ```
    2. **learning_from_feedback**
        - Preference data: 
          - Data: generate multiple responses using model (e.g., [A, B]) to a given prompt. 
          - User provides preferences (e.g., A < B or A > B).
          - Example:
            ```json
            {
                "messages": [
                    {"role" : "system", "content" : "You are a helpful assistant."},
                    {"role" : "user", "content" : "What is the capital of France?"}
                ],
                "reponse": {
                    "a": "You shouldb'use a large dataset and train for a long time.", 
                    "b": "You should use a small dataset and train for a short time."
                }, 
                "choice": "a"
            }
            ```
        - Verifiers:
          - Formal verifiers (e.g., for code, math)
          - Learned verifiers: train against an LM-as-a-judge
        - Algorithms:
          - Proximal Policy Optimization (PPO) from reinforcement learning, [ppo_2017](https://arxiv.org/pdf/1707.06347.pdf), [instruct_gpt](https://arxiv.org/pdf/2203.02155.pdf)
          - Direct Policy Optimization (DPO): for preference data, simpler[link-dpo](https://arxiv.org/pdf/2305.18290.pdf)
          - Group Relative Preference Optimization (GRPO): remove value function[link-grpo](https://arxiv.org/pdf/2402.03300.pdf)
4. Assignment-5
    1. [GitHub from 2024](https://github.com/stanford-cs336/spring2024-assignment5-alignment), [PDF from 2024](https://github.com/stanford-cs336/spring2024-assignment5-alignment/blob/master/cs336_spring2024_assignment5_alignment.pdf)
    2. Implement supervised fine-tuning
    3. Implement Direct Preference Optimization (DPO)
    4. Implement Group Relative Preference Optimization (GRPO)


### Assignments

- 5 assignments (basics, systems, scaling laws, data, alignment).
- No scaffolding code, but we provide unit tests and adapter interfaces to help you check correctness.
- Implement locally to test for correctness, then run on cluster for benchmarking (accuracy and speed).
- Leaderboard for some assignments (minimize perplexity given training budget).
- AI tools (e.g., CoPilot, Cursor) can take away from learning, so use at your own risk.


## Tokenization

BPE tokenizer pipeline:
1. Tokenize into subwords
2. Merge most frequent pairs of subwords
3. Repeat until desired vocabulary size




