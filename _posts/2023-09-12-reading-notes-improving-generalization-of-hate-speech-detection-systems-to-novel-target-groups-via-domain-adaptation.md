---
title: Reading Notes | Improving Generalization of Hate Speech Detection Systems to Novel Target Groups via Domain Adaptation
tags: 
- Generalization
- Hate Speech Detection
categories:
- Reading
---

> [[Semantic Scholar](https://www.semanticscholar.org/paper/Improving-Generalization-of-Hate-Speech-Detection-Ludwig-Dolos/bb774e72ee7ba8a3bb4db5ae76c4f2937b7437e9)] - [Code] - [Tweet] - [[Video](https://aclanthology.org/2022.woah-1.4.mp4)] - [Website] - [Slide]
>
> Change Logs:
>
> - 2023-09-11: First draft. This paper appears at WOAH '22.

The paper studies the generalization to new hate target groups on the single HateXplain dataset; they authors do so by comparing three existing methods, including (1) Unsupervised Domain Adaptation (UDA, this method is also used in paper [1]), (2) MixUp regularization, (3) curriculum labeling, and (4) DANN.

The paper also considers the back translation approach (specifically `(en, fr)`, `(en, de)`, and `(en, es)`) for data augmentation. 

# Experiments

- `Zero`: Directly apply a model trained on $\mathcal{D}_A$ to a new domain $\mathcal{D}_B$.
- `Zero+`: Augmenting $\mathcal{D}_A$ using back-translation.
- `ZeroB+`: Applying back-translation-based data augmentation while making sure that the each batch is class-balanced.

# Reference

1. [Unsupervised Domain Adaptation in Cross-corpora Abusive Language Detection](https://aclanthology.org/2021.socialnlp-1.10) (Bose et al., SocialNLP 2021): This paper considers the setting of training on dataset $\mathcal{D}_A$ and testing on another dataset $\mathcal{D}_B$, where $A, B$ are [HateEval](https://aclanthology.org/S19-2007/), [Waseem](https://aclanthology.org/N16-2013/), and [Davidson](https://arxiv.org/abs/1703.04009), resulting in 6 pairs. They use several existing methods to improve the test scores on $\mathcal{D}_B$.

2. [[2012.10289] HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection](https://arxiv.org/abs/2012.10289) (Mathew et al., AAAI 2021): This used to be the **only** dataset that provides the target groups of both hateful and non-hateful contents.

3. d: Data augmentation could happen in symbol space via rules, [word replacement through BERT](https://arxiv.org/abs/1812.06705),  text-generation models or feature space. However, the main paper chooses to use the back translation for data augmentation. 

    Here are two libraries on data augmentation in NLP:

    - [GitHub - makcedward/nlpaug: Data augmentation for NLP](https://github.com/makcedward/nlpaug) (4.1K stars)
    - [GitHub - GEM-benchmark/NL-Augmenter: NL-Augmenter 🦎 → 🐍 A Collaborative Repository of Natural Language Transformations](https://github.com/GEM-benchmark/NL-Augmenter) (700 stars; Continuous Updated).

