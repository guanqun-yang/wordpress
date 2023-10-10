---
title: Reading Notes | Understanding Dataset Difficulty with V-Usable Information
tags: 
- Data Selection
categories:
- Reading
---

> [[Semantic Scholar](https://www.semanticscholar.org/paper/Understanding-Dataset-Difficulty-with-V-Usable-Ethayarajh-Choi/39d05ffbc06fdca54ea6a90cd6d7fca202809aaa)] - [[Code](https://github.com/kawine/dataset_difficulty)] - [[Tweet](https://twitter.com/ethayarajh/status/1449203922057400329)] - [[Video](https://icml.cc/virtual/2022/oral/16634)] - [Website] - [Slide]
>
> Change Logs:
>
> - 2023-09-19: First draft. This paper appears as one of the **outstanding papers** at ICML 2022.

# Overview

The main contribution of the paper is a metric to evaluate the difficulty of the aggregate and sample-wise difficulty of a dataset for a model family $\mathcal{V}$: a lower score indicates a more difficult dataset. This metric is appealing because it gives an estimate of each sample's difficulty; this is not possible by using accuracy or F1 score.

# Method

Despite a lot of theoretical construct in Section 2, the way to compute the proposed metric is indeed fairly straightforward. 

Suppose we have a dataset $\mathcal{D} _ \text{train}$ and $\mathcal{D} _ \text{test}$ of a task, such as NLI, the proposed metric requires fine-tuning on $\mathcal{D} _ \text{train}$ two models from the same base model $\mathcal{V}$ and collecting measurements on $\mathcal{D} _ \text{test}$ (Algorithm 1):

- Step 1: Fine-tuning a model $g'$ on $\mathcal{D} _ \text{train} = \{ (x_1, y_1), \cdots, (x_m, y_m)  \}$ and another model $g$ on $\{ (\phi, y_1), \cdots, (\phi, y_m) \}$, where $\phi$ is an empty string; both $g'$ and $g$ are the model initialized from the same base model, such as `bert-base-uncased`.

- Step 2: For each test sample, the sample-wise difficulty (aka. PVI) is defined as $\mathrm{PVI}(x_i \rightarrow y_i) := -\log_2 g(y_i\vert \phi) + \log_2 g'(y_i\vert x_i)$; the aggregate difficulty is its average $\hat{I} _ \mathcal{V}(X \rightarrow Y) = \frac{1}{n}\sum _ i \mathrm{PVI}(x_i \rightarrow y_i)$.

    If the input and output are independent, the metric is provably and 0; it will be empirically close to 0.

Note that:

- The method requires a reasonably large dataset $\mathcal{D} _ \text{train}$. However, the exact size is not known in advance unless we train many models and wait to see when the curve plateaus, which is not feasible in practice. The authors use 80% of the SNLI dataset for estimation (Appendix A).
- The specific choice of models, hyperparameters, and random initializations does not influence the results a lot (Section 3.2).

# Applications

There are several applications when we use the proposed metric to rank the samples in a dataset:

- Identifying the annotation errors (Section 3).
- Using the metric to select challenging samples for data selection, including training data selection, data augmentation, and TCP (Section 4).
- Guiding the creation of new specifications as it is possible to compute the token-wise metric (Section 4.3).

# Additional Notes

- It is quite surprising that the CoLA dataset is more difficult than SNLI and MNLI according to the authors' measure. 

# Reference

1. [Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics](https://aclanthology.org/2020.emnlp-main.746) (Swayamdipta et al., EMNLP 2020): The method in the main paper and this paper both requires training a model.
2. [[2002.10689] A Theory of Usable Information Under Computational Constraints](https://arxiv.org/abs/2002.10689) (Xu et al., ICLR 2020).
