---
title: Reading Notes | Retrieval Enhanced Data Augmentation for Question Answering on Privacy Policies
tags: 
- Retrieval
- Data Augmentation
categories:
- Reading
---

> [[Semantic Scholar](https://www.semanticscholar.org/paper/Retrieval-Enhanced-Data-Augmentation-for-Question-Parvez-Chi/a7972d8f9f1ed21293a355e925006536fe6fe4df#citing-papers)] - [Code] - [Tweet] - [[Video](https://aclanthology.org/2023.eacl-main.16.mp4)] - [Slide] 
>
> Change Logs:
>
> - 2023-10-05: First draft. This paper appears at EACL 2023; it is dated 2204.08952. The code is not released.

# Overview

Paraphrasing and back-translation methods are only applicable for texts that are **not** sensitive to changes in texts. However, the privacy policies could convey wildly different meanings for small differences in the texts; this makes these two techniques less applicable to the problem being studied.

# Method

The authors propose a coarse-to-fine architecture for retrieval-based data augmentation. It consists of an ensemble of retrieval and filter models; these models include (1) regular BERT, (2) PBERT, a BERT fine-tuned with MLM objective on the privacy policies, and (3) the PBERT fine-tuned with SimCSE. 

- Retrieval Model (Bi-Encoder): This is a typical structure proposed in [1]. 
- Filter Model (Cross-Encoder): This is indeed a text classification model that takes the query, retrieved sentence pair and return a binary decision.

Note that

- The retrieval model and filter model are trained separately; they are not jointly trained in this work.
- The ensemble here is more like three systems working in parallel and aggregating the collected sentences altogether at last. 

During inference, the top-$k$ retrieved samples are filtered by the trained filter model. The aggregated retrieved texts are combined with original dataset to fine-tune the privacy QA model.

# Reference

1. [Dense Passage Retrieval for Open-Domain Question Answering](https://aclanthology.org/2020.emnlp-main.550) (Karpukhin et al., EMNLP 2020) and [HuggingFace](https://huggingface.co/docs/transformers/model_doc/dpr).

