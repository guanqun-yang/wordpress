---
title: Research Notes | Training Data Optimization
tags: 
- Data Selection
- Learning with Noisy Labels
categories:
- Research
---

# Problem Statement

Suppose we have a collection of datasets from $K$ sources $\mathcal{D} _ 1, \cdots, \mathcal{D} _ K$. These $K$ datasets have been unified regarding input and output spaces.

Now we split each $\mathcal{D} _ i$ into train, validation, and test splits $\mathcal{D} _ i ^ \text{train},\ \mathcal{D} _ i ^ \text{val}$ and $\mathcal{D} _ i ^ \text{test}$ and form the aggregated train, validation, and test sets as $\mathcal{D}^\text{train} := \cup _ {i=1}^ K D _ i^\text{train}$, $\mathcal{D}^\text{val} := \cup _ {i=1}^ K D _ i^\text{val}$, and $\mathcal{D}^\text{test} := \cup _ {i=1}^ K D _ i^\text{test}$ .

The learning problem could vary depending the quality of the datasets after (1) dataset collection and annotation by authors of different datasets, and (2) dataset unification when merging $K$ datasets into one. This is because:

- If labels are reliable, then this is dataset distillation problem. The argument is to save computation resourcces when training on $\mathcal{D} \subseteq \mathcal{D} ^ \text{train}$ while maintaining the performance as a model trained in (1) each $\mathcal{D}_i,\ i \in [K]$, (2)  $\mathcal{D} ^ \text{train}$, and (3) $\mathrm{Sample}(\mathcal{D} ^ \text{train})$ that matches the size of $\mathcal{D}$.

- If labels are not reliable, then the argument is to prevent the low-quality labels from offsetting the benefits of a larger training dataset (rather than distilling a smaller dataset to save compute). We have three options:

| Index | Method                                                       | Type    |
| ----- | ------------------------------------------------------------ | ------- |
| 1     | Reannotating the entire dataset. This could be reduced as a dataset distillation problem as now we have more confidence on the filtered datasets. | Offline |
| 2     | Identifying and removing unreliable labels and optionally using these samples as an unsupervised dataset. This is also reducible to a dataset distillation problem as 1. | Offline |
| 3     | Learning with the noisy labels (LNL as described in 1) they are; this requires the learning algorithm to explicitly consider the variablity in the label quality. | Online  |

# Reference

1. [NoisywikiHow: A Benchmark for Learning with Real-world Noisy Labels in Natural Language Processing](https://aclanthology.org/2023.findings-acl.299) (Wu et al., Findings 2023)
