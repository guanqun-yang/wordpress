---
title: Research Notes | Generalizable Hate Speech Detection
tags: 
- HateSpeech
- Generalization
- Benchmark
categories:
- Research
---

# Overview

This post is the summary of the following methods; they rank top on the [CivilComments-WILDS benchmark](https://wilds.stanford.edu/leaderboard/#civilcomments):

| Rank  | Method      | Paper                                                        |
| ----- | ----------- | ------------------------------------------------------------ |
| 1     | FISH        | [[2104.09937] Gradient Matching for Domain Generalization](https://arxiv.org/abs/2104.09937) (Shi et al., ICLR 2022 |
| 2, 3  | IRMX        | [[2206.07766] Pareto Invariant Risk Minimization: Towards Mitigating the Optimization Dilemma in Out-of-Distribution Generalization](https://arxiv.org/abs/2206.07766) (Chen et al., ICLR 2023) |
| 4     | LISA        | [[2201.00299] Improving Out-of-Distribution Robustness via Selective Augmentation](https://arxiv.org/abs/2201.00299) (Yao et al., ICML 2022) |
| 5     | DFR         | [[2204.02937] Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations](https://arxiv.org/abs/2204.02937) (Kirichenko et al., ICLR 2023) |
| 6, 8  | Group DRO   |                                                              |
| 7, 12 | Reweighting | [[1901.05555] Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555) (Cui et al., CVPR 2019) is one example that uses this method; the reweighting method could date back to much earlier works. |

# Reweighting, IRM, and CORAL

IRM [2] and CORAL [3] are two extensions of the basic reweighting method by adding an additional penalty term on top of the reweighting loss; this term is based on some measures of the data representations from different domains to encourage the data distribution of different domains to be similar.





# Reference

1. [[2012.07421] WILDS: A Benchmark of in-the-Wild Distribution Shifts](https://arxiv.org/abs/2012.07421)
2. [[1907.02893] Invariant Risk Minimization](https://arxiv.org/abs/1907.02893) (Arjovsky et al.)
3. [[2007.01434] In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434) (Gulrajani and Lopez-Paz)