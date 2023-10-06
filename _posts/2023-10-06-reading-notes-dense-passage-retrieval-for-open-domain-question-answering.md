---
title: Reading Notes | Dense Passage Retrieval for Open-Domain Question Answering
tags: 
- DPR
categories:
- Reading
---

> [[Semantic Scholar](https://www.semanticscholar.org/paper/Dense-Passage-Retrieval-for-Open-Domain-Question-Karpukhin-O%C4%9Fuz/79cd9f77e5258f62c0e15d11534aea6393ef73fe)] - [Code] - [Tweet] - [Video] - [Website] - [Slide]
>
> Change Logs:
>
> - 2023-10-05: First draft. This paper appears at EMNLP 2020.

# Overview

- Dense Passage Retrieval is a familiar thing proposed in this paper. The issue is that previous solutions underperform BM25. The contribution of this paper is discovering an engineering feasible solution that learns a DPR model effectively without many examples; it improves upon the BM25 by a large margin.

# Method

The training goal of DPR is to learn a metric where the distance between the query $q$ and relevant documents $p^+$ smaller than that of irrelevant documents $p^-$ in the high-dimensional space. That is, we want to **minimize** the loss below:
$$
L(q _ i, p _ i ^ +, p _ {i1} ^ -, \cdots, p _ {in}^-) := -\log \frac{ \exp(q _ i^T p _ i^+)}{\exp(q_i^T p _ i ^  +) + \sum _ {j=1}^n \exp(q _ i ^ T p _ {ij}^-)}
$$
The authors find that using the "in-batch negatives" is a simple and effective negative sampling strategy (see "Gold" with and without "IB"). Specifically, within a batch of $B$ examples, any answer that is not associated with the current query is considered a negative. If **one answer** (see the bottom block) retrieved from BM25 is added as a hard negative, the performance will improve more.

![image-20231006000405936](https://raw.githubusercontent.com/guanqun-yang/remote-images/master/2023/10/upgit_20231006_1696565045.png)

The retrieval model has been trained for 40 epochs for larger datasets and 100 epochs for smaller ones with a learning rate `1e-5`

# Additional Notes

-   The dual-encoder + cross-encoder design is a classic; they are not necessarily end-to-end differentiable. For example, in this work, after fine-tuning the dual-encoder for retrieval, the authors separately fine-tuned a QA model. This could be a favorable design due to better performance:

    >   This approach obtains a score of 39.8 EM, which suggests that our strategy of training a strong retriever and reader in isolation can leverage effectively available supervision, while outperforming a comparable joint training approach with a simpler design.

-   The inner product of unit vectors is indeed the cosine similarity.

# Code

HuggingFace provides [classes](https://huggingface.co/docs/transformers/v4.34.0/model_doc/dpr) for DPR. The Retrieval Augmented Generation (RAG) is one [example](https://github.com/huggingface/transformers/tree/main/examples/research_projects/rag) that fine-tunes using DPR to improve knowledge-intense text generation.
