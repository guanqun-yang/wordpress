---
title: Basics | A Quick Reference of the Evaluation Metrics in NLP
tags: 
- Evaluation Metrics
categories:
- Basics
---

# Overview

This set of evaluation metrics I discuss in this post is organized based on the typical tasks in NLP (see [1]); they are:

- Sequence Classification
- Token Classification (Tagging)
- Generation: This category includes all tasks whose outputs are a sequence of tokens. For example, question answering, machine translation, text summarization and text simplification, and paraphrasing.
- Retrieval
- Regression

# Sequence Classification

## Confusion Matrix

The $\mathbf{C}_ {ij}$ means the number of samples of class $i$ receive the prediction $j$; the rows are the true classes while the columns are the predictions.

- When there are only two classes, we could define $\mathrm{TN} = \mathbf{C}_ {11}, \mathrm{FP}=\mathbf{C}_ {12}, \mathrm{FN}=\mathbf{C}_ {21}$, and $\mathrm{TP}=\mathbf{C}_ {22}$:
    
    Bases on these 4 numbers, we could define

    - $\mathrm{TPR}$, $\mathrm{FPR}$, $\mathrm{FNR}$, and $\mathrm{TNR}$: they are the normalized version of the confusion matrix on the true number of samples in each class. 
    - Precision $P$ and Recall $R$: we could compute these two numbers for each class; they are important in diagnosing a classifier's performance.


| Notation       | Formula                                       |
| :------------- | :-------------------------------------------- | 
| $\mathrm{TNR}$ | $\frac{\mathrm{TN}}{\mathrm{N}}$              |   
| $\mathrm{FNR}$ | $\frac{\mathrm{FN}}{\mathrm{N}}$              |   
| $\mathrm{FPR}$ | $\frac{\mathrm{FP}}{\mathrm{P}}$              | 
| $\mathrm{TPR}$ | $\frac{\mathrm{TP}}{\mathrm{P}}$              |
| $P$            | $\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}$ |
| $R$            | $\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}$ |


```python
import numpy as np
from sklearn.metrics import confusion_ matrix

y_ true = np.array([1, 0, 1, 0, 1])
y_ pred = np.array([1, 0, 1, 1, 0])

# raw counts
tn, fp, fn, tp = confusion_ matrix(
    y_ true=y_ true,
    y_ pred=y_ pred
).ravel()

print(tn, fp, fn, tp)
# expected output: (1, 1, 1, 2)
# actual output: array([1, 1, 1, 2])

# tnr, fpr, fnr, tpr
tnr, fpr, fnr, tpr = confusion_ matrix(
    y_ true=y_ true,
    y_ pred=y_ pred,
    normalize="true",
).ravel()

print(tnr, fpr, fnr, tpr)
# expected output: (1/2, 1/2, 1/3, 2/3)
# actual output: 0.5 0.5 0.3333333333333333 0.6666666666666666

```

- We could use the code below to visualize a confusion matrix. Following the example above, we have:

```python
import pandas as pd

df = pd.DataFrame(
    confusion_ matrix(y_ true, y_ pred),
    index=labels
    columns=labels,
)

sns.plot(df)

```

# Reference

1. [[2107.13586] Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](https://arxiv.org/abs/2107.13586): This survey gives clear categories of NLP tasks: `GEN` , `CLS`, and `TAG`.
2. [Confusion matrix - Wikipedia](https://en.wikipedia.org/wiki/Confusion_ matrix): A comprehensive overview of a list of related metrics.