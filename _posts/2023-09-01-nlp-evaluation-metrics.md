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

The $\mathbf{C}_{ij}$ means the number of samples of class $i$ receive the prediction $j$; the rows are the true classes while the columns are the predictions.

- When there are only two classes, we could define $\mathrm{TN} = \mathbf{C} _ {11}, \mathrm{FP}=\mathbf{C} _ {12}, \mathrm{FN}=\mathbf{C} _ {21}$, and $\mathrm{TP}=\mathbf{C} _ {22}$:
  
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
from sklearn.metrics import confusion_matrix

y_true = np.array([1, 0, 1, 0, 1])
y_pred = np.array([1, 0, 1, 1, 0])

# raw counts
tn, fp, fn, tp = confusion_matrix(
    y_true=y_true,
    y_pred=y_pred
).ravel()

print(tn, fp, fn, tp)
# expected output: (1, 1, 1, 2)
# actual output: array([1, 1, 1, 2])

# tnr, fpr, fnr, tpr
tnr, fpr, fnr, tpr = confusion_matrix(
    y_true=y_true,
    y_pred=y_pred,
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
    confusion_matrix(y_true, y_pred),
    index=labels
    columns=labels,
)

sns.plot(df)

```

## Matthews Correlation Coefficient (MCC)

The Matthews Correlation Coefficient (MCC) (also called $\phi$ statistic) is a special case of Pearson's correlation for two boolean distributions with value $\{-1, 1\}$.

The range of MCC is $[-1, 1]$, where 1 - perfect predictions, 0 - random prediction, and -1 - inverse prediction. It is better than F1 score (and therefore accuracy) as it does not have similar majority-class bias by additionally considering $TN$  (remember that $P=\frac{TP}{TP+FP}$, $R=\frac{TP}{TP+FN}$, and $F1=\frac{2PR}{P+R}=\frac{2TP}{2TP+FP+FN}$ and $TN$ is not considered in $F1$).

Concretely, consider the following examples

- Example 1: Consider a dataset with 10 samples and $y =(1, 1, \cdots, 1, 0)$ and $\hat{y} = (1, 1, \cdots , 1, 1)$, then the $F1=0.9474, MCC=0$.
- Example 2: Consider the use case A1 detailed in [3], given a dataset of 91 positive samples and 9 negative samples, suppose all 9 negative samples are misclassified and 1 positive sample is misclassified (i.e. $TP=90, TN=0, FP=9, FN=1$), then $F1=0.95$ while $MCC=-0.03$.

A potential issue when computing the $MCC$ is that the metric is undefined when $y$ is all 0 or all 1, and this will make either $TP$ or $TN$ undefined. 
This issue also come with F1 score as it could be written as $F1=\frac{2P\cdot R}{P+R}=\frac{2TP}{2TP+FP+FN}$ and it could not be computed when $TP$ is undefined when all labels are 0.

To have a better comparison between $MCC$ and $F1$, consider a dataset with 100 samples, plot the $F1 \sim MCC$ curve given different combinations of $TP, FP, TN$ and $FN$ using simulation. The dots colored red are those (1) achieve an more than 95% F1 score, and (2) correspond to a dataset where there are more than 95% positive samples. We could see that their $MCC$ is relatively low.

![](https://raw.githubusercontent.com/guanqun-yang/remote-images/master/2023/09/upgit_20230921_1695311325.png)

# Reference

1. [[2107.13586] Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](https://arxiv.org/abs/2107.13586): This survey gives clear categories of NLP tasks: `GEN` , `CLS`, and `TAG`.
2. [Confusion matrix - Wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix): A comprehensive overview of a list of related metrics.
3. [The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation | BMC Genomics | Full Text](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7) (Chicco and Jurman).

