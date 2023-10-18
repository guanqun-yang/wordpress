---
title: Reading Notes | Text Embeddings Reveal (Almost) As Much As Text
tags: 
- Embeddings
categories:
- Reading
---

> [Semantic Scholar] - [[Code](https://github.com/jxmorris12/vec2text)] - [Tweet] - [Video] - [Website] - [Slide] 
>
> Change Logs:
>
> - 2023-10-18: First draft. This paper appears at EMNLP 2024. This paper is a work by John X. Morris. It comes with an easy-to-use library that could revert the OpenAI embeddings.

# Overview

The authors assume an attacker has access to (1) a compromised vector database, and (2) a black-box embedding model $\phi(\cdot)$ (for example, OpenAI's embedding API). The attacker starts from an embedding  and an empty string to reconstruct the original text corresponding to that string; the method proposed in the paper manage to recover a string up to 32 tokens.

The main motivation of this paper is privacy.

# Method





# Reference

1. [[2211.00053] Generating Sequences by Learning to Self-Correct](https://arxiv.org/abs/2211.00053) (Welleck et al.): This is the main inspiration of the main paper.

    > This method relates to other recent work generating text through iterative editing (Lee et al., 2018; Ghazvininejad et al., 2019). Especially relevant
    > is Welleck et al. (2022), which proposes to train a text-to-text ‘self-correction’ module to improve language model generations with feedback.

2. [Decoding a Neural Retriever’s Latent Space for Query Suggestion](https://aclanthology.org/2022.emnlp-main.601) (Adolphs et al., EMNLP 2022)