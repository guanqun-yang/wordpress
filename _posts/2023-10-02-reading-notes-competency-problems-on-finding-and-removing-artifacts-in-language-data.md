---
title: Reading Notes | Competency Problems - On Finding and Removing Artifacts in Language Data
tags: 
- Generalization
- Spurious Correlation
categories:
- Reading
---

> [[Semantic Scholar](https://www.semanticscholar.org/paper/Competency-Problems%3A-On-Finding-and-Removing-in-Gardner-Merrill/023fc86c932fbc36702a6ad11c94ba419e1d8d88)] - [Code] - [Tweet] - [[Video](https://aclanthology.org/2021.emnlp-main.135.mp4)] - [Website] - [Slide]
>
> Change Logs:
>
> - 2023-09-26: First draft. The paper appears at EMNLP 2021.

- The following is the main claim of the paper, as is summarized in [1]:

> [...] all correlations between labels and individual "input features" are spurious.

- Spurious correlation is useful in the training data but unreliable in general [1].

# Reference

1. [Informativeness and Invariance: Two Perspectives on Spurious Correlations in Natural Language](https://aclanthology.org/2022.naacl-main.321) (Eisenstein, NAACL 2022): This paper updates the claim in the main paper **theoretically**: feature-label correlation is not related to whether label is invariant to the the interventions on the feature. 

    **Practically**, the paper suggests the partial invariance (whether independent or not) for real-world datasets; for example, the sentiment of a movie review is invariant to the actor names. The paper also suggest the following options to improve model robustness:

    > data augmentation, causally-motivated regularizers, stress tests, and “worst-subgroup” performance metrics (and associated robust optimizers) can be seen as enforcing or testing task-specific invariance properties that provide robustness against known distributional shifts (e.g., Lu et al., 2020; Ribeiro et al., 2020; Kaushik et al., 2021; Koh et al., 2021; Veitch et al., 2021). Such approaches generally require domain knowledge about the linguistic and causal properties of the task at hand — or to put it more positively, they make it possible for such domain knowledge to be brought to bear. Indeed, the **central argument of this paper is that no meaningful definition of spuriousness or robustness can be obtained without such domain knowledge**.

    - [[1807.11714] Gender Bias in Neural Natural Language Processing](https://arxiv.org/abs/1807.11714) (Lu et al.)
    - [[2010.02114] Explaining The Efficacy of Counterfactually Augmented Data](https://arxiv.org/abs/2010.02114) (Kaushik et al., ICLR 2021)
    - [[2106.00545] Counterfactual Invariance to Spurious Correlations: Why and How to Pass Stress Tests](https://arxiv.org/abs/2106.00545) (Veitch et al., NeurIPS 2021)
    - Koh et al. and Ribeiro et al. correspond to the WILDS paper and the CheckList paper.

2. [On the Limitations of Dataset Balancing: The Lost Battle Against Spurious Correlations](https://aclanthology.org/2022.findings-naacl.168) (Schwartz & Stanovsky, Findings 2022): This paper shows that creating a truly balanced dataset devoid of the issues mentioned in the main paper will also throw the useful signals encoded in the texts ("throw the baby out with the bathwater").

