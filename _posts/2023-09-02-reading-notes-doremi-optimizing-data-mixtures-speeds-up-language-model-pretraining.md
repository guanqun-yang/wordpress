---
title: Reading Notes | DoReMi - Optimizing Data Mixtures Speeds Up Language Model Pretraining
tags:
  - LLM
  - DataSelection
categories:
  - Reading
---

# Overview



# Other Information

- The ratios of domains should be counted using number of tokens rather than number of documents, even though different tokenizers may return slightly different ratios.

# Reference

1. [[2110.10372] Distributionally Robust Classifiers in Sentiment Analysis](https://arxiv.org/abs/2110.10372) (Stanford Course Project Report).

2. [Distributionally Robust Finetuning BERT for Covariate Drift in Spoken Language Understanding](https://aclanthology.org/2022.acl-long.139) (Broscheit et al., ACL 2022): This paper is one of few papers I could find that applies DRO to an NLP model; the problem the authors addressing here is mitigating the spurious correlation (or improving robustness) of a **cascade** of text and token classification models.

    The standard ERM (aka. MLE) assumes a single distribution and therefore all losses are equally important. However, the DRO tries to minimize the maximum (i.e., the worse case) of a set of distributions; this set of distributions is modeled by prior knowledge. 

3. [[1810.08750] Learning Models with Uniform Performance via Distributionally Robust Optimization](https://arxiv.org/abs/1810.08750)

4. [Distributionally Robust Language Modeling](https://aclanthology.org/D19-1432) (Oren et al., EMNLP-IJCNLP 2019): The main paper extensively cites this paper. The goal of this paper is to train a language model on a dataset mixture of $K$ sources $\cup _ {i=1}^K\mathcal{D} _ i$ without degrading the perform on each domain's test set; it is a **practical application** of [3] in language modeling.

    This setting may be useful because (1) each $\mathcal{D} _ i$ may not be large enough to train the model, and (2) the authors observe that training on data mixture degrades the performance on the each domain's test set than using a smaller dataset.

5. [[1911.08731] Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization](https://arxiv.org/abs/1911.08731) (ICLR '20; 1K citations). This paper fine-tunes BERT using DRO on the MNLI dataset; the paper also experiments on the image datasets.
