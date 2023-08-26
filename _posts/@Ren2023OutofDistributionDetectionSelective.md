---
title: Reading Notes | Out-of-Distribution Detection and Selective Generation for Conditional Language Models
tags: 
- Seq2Seq
- Selective Generation
- OOD
categories:
- Reading
---

# Notes

> [[Semantic Scholar](https://www.semanticscholar.org/paper/Out-of-Distribution-Detection-and-Selective-for-Ren-Luo/94b6f6822f364cf7b1a3a9984667c009e2ec6a65)] - [Code] - [Tweet] - [Video] - [[Slide](https://iclr.cc/media/iclr-2023/Slides/11478.pdf)]
>
> - 2023-08-26: First draft. This paper appears at ICLR '23.

## Overview

The paper proposes to teach the encoder-decoder language models (the Transformers model on translation task and the PEGASUS model on summarization task) to abstain when receiving sentences substantial different from the training distribution; abstaining from generating some contents (more metaphorically, saying "I don't know" when "I" really do not know) is indicates that the system is trustworthy; this practice improves the system safety.

## Method

Given a domain-specific dataset $\mathcal{D}_1 = \{ (x_1, y_1), \cdots, (x_N, y_N)\}$ and a general-domain dataset (for example, C4) $\mathcal{D}_0$, the authors fit 4 Mahalanobis distance metric defined as $\mathrm{MD}(\mathbf{x}; \mu, \Sigma) = (\mathbf{x} - \mu )^T \Sigma^{-1} (\mathbf{x} - \mu )$.

- Background:
- Input:
- Output
- Decoded output given ground-truth input:

## Experiments

- Perplexity should not be used for OOD detection **alone** because
  - The fitted PDFs of perplexities on different datasets (i.e., domains) mostly overlap (Figure 1).
  - When the averaged OOD scores increase, the Kentall's $\tau$ between perplexity and quality measure is (1) low and (2) decreases (Figure 4). If perplexity is a good measure, then the curve should be mostly flat.
  - It could be combined with the proposed metric (Section 4.3, 4.4).
- The distance between domains could be quantitatively measured with the Jaccard similarity of $n$-grams (1 through 4 in the paper) (Table A.10). This is used to quantify the task difficulties as the authors define "near OOD" and "far OOD" domains (Table 1).



## References

1. [[1705.08500] Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500)

2. [[1612.01474] Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/abs/1612.01474): This paper works on predictive uncertainty of deep classification models. Their proposed approach tries to approximate the state-of-the-art Bayesian NNs while being easy to implement and parallelize.

3. [[2106.09022] A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection](https://arxiv.org/abs/2106.09022): For a classification problem of $K$ classes, we could fit $K$ class-dependent Gaussian and 1 background Gaussian. Then we could use these $(K+1)$ Gaussians to detect anomalies: a negative score in class $k$ indicates that the sample is in the domain $k$ and a positive score means it is OOD; a more positive score shows that the sample deviates more from that domain.

