---
title: Research Notes | Training Data Optimization
tags: 
- DataSelection
- NoisyLabel
categories:
- Research
---

# Problem Statement

Suppose we have a collection of datasets from $K$ sources $\mathcal{D} _ 1, \cdots, \mathcal{D} _ K$. These $K$ datasets have been unified regarding input and output spaces.

Now we split each $\mathcal{D} _ i$ into train, validation, and test splits $\mathcal{D} _ i ^ \text{train},\ \mathcal{D} _ i ^ \text{val}$ and $\mathcal{D} _ i ^ \text{test}$ and form the aggregated train, validation, and test sets as $\mathcal{D}^\text{train} := \cup _ {i=1}^ K D _ i^\text{train}$, $\mathcal{D}^\text{val} := \cup _ {i=1}^ K D _ i^\text{val}$, and $\mathcal{D}^\text{test} := \cup _ {i=1}^ K D _ i^\text{test}$ .

The learning problem could vary depending the quality of the datasets after (1) dataset collection and annotation by authors of different datasets, and (2) dataset unification when merging $K$ datasets into one. This is because:

- If labels are reliable, then this is dataset selection problem. The argument is to save computation resources when training on $\mathcal{D} \subseteq \mathcal{D} ^ \text{train}$ while maintaining the performance as a model trained in (1) each $\mathcal{D}_i,\ i \in [K]$, (2)  $\mathcal{D} ^ \text{train}$, and (3) $\mathrm{Sample}(\mathcal{D} ^ \text{train})$ that matches the size of $\mathcal{D}$.

    In some special cases, another motivation for dataset selection is that we know the size of a sampled dataset (for example, the dataset statistics described in a paper) but we are not sure what are exactly these samples.

- If labels are not reliable, then the argument is to prevent the low-quality labels from offsetting the benefits of a larger training dataset (rather than distilling a smaller dataset to save compute). We have three options:

| Index | Method                                                       | Type    |
| ----- | ------------------------------------------------------------ | ------- |
| 1     | Reannotating the entire dataset. This could be reduced as a dataset distillation problem as now we have more confidence on the filtered datasets. | Offline |
| 2     | Identifying and removing unreliable labels and optionally using these samples as an unsupervised dataset. This is also reducible to a dataset selection problem as 1. | Offline |
| 3     | Learning with the noisy labels (LNL as described in 1) they are; this requires the learning algorithm to explicitly consider the variablity in the label quality. | Online  |

Note that there is a easily topic called "dataset distillation" that one may easily confused with. The goal of dataset distillation is to create **synthetic dataset** in the **feature space** based on the original one to match the performance on the test set. Previous show that it is possible to attain the original performance on MNIST ([3]) and IMDB ([4]) with a synthetic dataset of size (surprisingly) 10 and 20.

# Adaptive Data Selection

With the test sets finalized, we could now work on sampling training sets, i.e., choosing one specific $\mathrm{Sample}(\cdot)$ function described above. The goal here is to sample the training set so that the scores on the test sets are maximized:

- [DSIR](https://arxiv.org/abs/2302.03169): Suppose we need to sample $B$ batches of samples totaling $K$, then we could start by randomly sampling the 1st batch and then calling the DSIR algorithm in the future batches until we have collected $K$ samples. This should be done for each label.

# Reference

1. [NoisywikiHow: A Benchmark for Learning with Real-world Noisy Labels in Natural Language Processing](https://aclanthology.org/2023.findings-acl.299) (Wu et al., Findings 2023)

2. [[2202.01327] Adaptive Sampling Strategies to Construct Equitable Training Datasets](https://arxiv.org/abs/2202.01327) (Cai et al., FAccT 2023)

3. [[2301.04272] Data Distillation: A Survey](https://arxiv.org/abs/2301.04272) (Sachdeva and McAuley, JMLR).

4. [[1811.10959] Dataset Distillation](https://arxiv.org/abs/1811.10959) (Wang et al.)

5. [[1910.02551] Soft-Label Dataset Distillation and Text Dataset Distillation](https://arxiv.org/abs/1910.02551) (Sucholutsky and Schonlau, IJCNN 2020). This is the **only** paper referenced in 3 describing the dataset distillation for texts. This paper is based on the **very original** data distillation objective proposed in 4.

6. [[2302.03169] Data Selection for Language Models via Importance Resampling](https://arxiv.org/abs/2302.03169) (Xie et al.)

7. [[2305.10429] DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining](https://arxiv.org/abs/2305.10429) (Xie et al.)

8. [[2306.11670] GIO: Gradient Information Optimization for Training Dataset Selection](https://arxiv.org/abs/2306.11670) (Everaert and Potts): This paper has similar settings as the DSIR paper [6]: we are selecting new samples by minimizing their KL divergence with an existing set of unlabeled samples. The paper claims an advantage over the DSIR as the proposed algorithm requires fewer samples:

    > Like GIO, these heuristic methods aim to select a subset of data that is higher quality and more relevant. However, they are either highly tailored to their particular tasks or they require very large numbers of examples (to develop classifiers or construct target probabilities). By contrast, GIO is task- and domain-agnostic, it can be applied plug-and-play to a new task and dataset, and it requires comparatively few gold examples X to serve as the target distribution.
