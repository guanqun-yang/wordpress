---
title: Research Notes | Transformer from Scratch
tags: 
categories:
- Research
---

# Overview

This post aims to implements the the transformer model and its variant from scratch. It is based on the following posts:

1.   [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) (Harvard NLP)

2.   [The Illustrated Transformer – Jay Alammar – Visualizing machine learning one concept at a time.](https://jalammar.github.io/illustrated-transformer/)

3.   [GitHub - karpathy/minGPT: A minimal PyTorch re-implementation of the OpenAI GPT (Generative Pretrained Transformer) training](https://github.com/karpathy/minGPT): Anderj Karpathy also creates a 2-hour [video](https://www.youtube.com/watch?v=kCc8FmEb1nY) describing the process he creates the model.

     [GitHub - karpathy/nanoGPT: The simplest, fastest repository for training/finetuning medium-sized GPTs.](https://github.com/karpathy/nanoGPT): This is the optimized version of `minGPT` that is able to reproduce some of the mid-sized models, including a 1.3B GPT-2. Pretraining a 124M GPT-2 took 4 days on 8 A100 GPUs (each 40 GB).

4.   [GitHub - nlp-with-transformers/notebooks: Jupyter notebooks for the Natural Language Processing with Transformers book](https://github.com/nlp-with-transformers/notebooks)