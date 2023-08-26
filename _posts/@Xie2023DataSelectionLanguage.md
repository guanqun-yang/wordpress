---
title: Reading Notes | Data Selection for Language Models via Importance Resampling
tags: 
- LLM
- Pretraining
- Data Selection
categories:
---
# Notes
> [[Semantic Scholar](https://www.semanticscholar.org/paper/Data-Selection-for-Language-Models-via-Importance-Xie-Santurkar/a008cc894024329d832d2c9c489d57440e3fa234)] - [Code] - [Tweet] - [Video]
> - 2023-08-25: First draft. This paper is a preprint.

## Problem Settings

The paper tries to solve the data selection problem for language model pretraining. Suppose a collection of high-quality samples $x_1' \cdots, x_n'$ have a distribution $p$, with another collection of $N$ low-quality samples available; how could we sample a high-quality subset $x_1, \cdots, x_k\quad (k \ll N)$ with an underlying distribution $q$ that approximates $p$?

## Heuristic Classification

This is the approach used by GPT-3, EleutherAI's Pile dataset, PaLM to select training corpus. Specifically,

- Training a fasttext regression model $f: \mathcal{X} \rightarrow [0, 1]$ using unigrams and bigrams from high-quality datasets.

- Applying the trained classifier to the low-quality data collection and sampling $x_i$ if `np.random.pareto(alpha) > 1 - score`. `alpha` is chosen as 9 in the GPT-3 paper:

  > We chose $\alpha=9$ in order to take mostly documents the classifier scored highly, but still include some documents that were out of distribution. 
  
  Samples from the Pareto distributions are small; this makes most of the included samples high-quality.

## Data Selection with Importance Resampling (DSIR)

The issue of the heuristic classification is that it does not explicitly model the underlying distribution $q$. Instead, the authors of DSIR explicitly model the distribution of a corpus as follows:

$$
p(z; \gamma)=\prod_{j=1}^{10000} \gamma_j^{z_j}
$$

where

- $z$ is a 10000-dimensional vector; its entries represent the index of the hashed unigrams and bigrams (with potential collisions).
- $\gamma$ is the parameter to learn.

After learning the distributions $p$ and $q$, we could assign a weight to each sample of the pool $w_i = \frac{p(z_i)}{q(z_i)},\ i =1, \cdots, N$. We could then sample the pool with weights $w_1, \cdots, w_N$ until we have collected $k$ samples. The authors sample the data without replacement and explain this choice theoretically. They could have explained it better, as deduplication is one key aspect for language model pretraining ([Falcon paper](https://arxiv.org/pdf/2306.01116.pdf)).

## Experiments

- The average KL reduction strongly correlates ($r = 0.89$) with the accuracies on the downstream task (i.e., GLUE). The DSIR significantly improves the downstream accuracies. This correlation is a post-hoc justification of dataset modeling $p(z;\gamma)$.

  The KL reduction is defined as following, where $\hat{p}$ is target distribution, $\hat{q}$ is the raw distribution, and $p'$ is the distribution of doing some data selection, including random selection, expert curation, and sampling with the proposed algorithm.

  $$
  \frac{1}{\vert \mathcal{T}\vert} \sum_{\hat{p} \sim \mathcal{T}} \mathrm{KL}(\hat{p} \parallel \hat{q}) - \mathrm{KL}(\hat{p} \parallel p'),\quad \mathrm{KL}(p\parallel q)= H(p, q) - H(p)
  $$

  There is $\mathcal{T}$ because the authors are trying to evaluate the data selection methods; these methods could be applied to many different models. Therefore, there will be $n$ scores for $n$ data selection algorithms.

- Continued pretraining on RoBERTa on domain-specific datasets sampled using the DSIR algorithm improves upon the model fine-tuned with datasets sampled with the baseline methods (Table 1, 2).
- Training BERT from scratch using data sampled with the different sampling approaches and fine-tuning on GLUE shows the proposed selection algorithm's advantages over the heuristic classification and random selection (Table 4).
- It is important to make sure the domain of the pretraining dataset matches the deployment domain as (1) performance typically drops when the domain is different (Table 3), and (2) domain transfer is hard to predict (Figure 3).

