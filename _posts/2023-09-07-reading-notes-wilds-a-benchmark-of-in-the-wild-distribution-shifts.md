---
title: Reading Notes | WILDS - A Benchmark of in-the-Wild Distribution Shifts
tags: 
- Benchmark
- Generalization
categories:
- Reading
---

> [Semantic Scholar] - [Code] - [Tweet] - [Video] - [[Website](https://wilds.stanford.edu/datasets/)] - [Slide] - [[Leaderboard](https://wilds.stanford.edu/leaderboard/)]
>
> Change Logs:
>
> - 2023-09-06: First draft. 

Distribution shifts happen when test conditions are "newer" or "smaller" compared to training conditions. The paper defines them as

- Newer: Domain Generalization

    The test distribution is related to but distinct (aka. **new** or unseen during trainig) to the training distributions. Note that the test conditions are not necessarily a superset of the training conditions; they are not "larger" compared to the "smaller" case described below. 

    Here are two typical examples of domain generalization described in the paper:

    - Training a model based on patient information from some hospitals and expect the model to generalize to many more hospitals; these hospitals may or may not be  a superset of the hospitals we collect training data from.
    - Training an animal recognition model on images taken from some existing cameras and expect the model to work on images taken on the newer cameras.

- Smaller: Subpopulation Shift

    The test distribution is a **subpopulation** of the training distributions. For example, degraded facial recognition accuracy on the underrepresented demographic groups ([3] and [4]).

The dataset includes regular and medical image, graph, and text datasets; there are 3 out of 10 are text datasets, where the less familiar Py150 is a code completion dataset. Note that the authors fail to cleanly define why there are subpopulation shifts for Amazon Reviews and Py150 datasets as the authors acknowledge below:

> However, it is not always possible to cleanly define a problem as one or the other; for example, a test domain might be present in the training set but at a very low frequency.

For Amazon Reviews dataset, one viable explanation on why there is subpopulation shift is uneven distribution of reviews on the **same** product in the train, validation, and test set.

| Name           | Domain Generalization                                        | Subpopulation Shift                                          | Notes                                           |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------- |
| CivilComments  | No; this is because the demographic information of the writers are unknown; if such information is known, we could also create a version with domain generation concern. | Yes; this is because the mentions of 8 target demographic groups are available. | The only dataset with only subpopulation shift. |
| Amazon Reviews | Yes; due to disjoint users in the train, OOD validation, and OOD test set; there are also ID validation and ID test set from same users as the training set. | Yes                                                          |                                                 |
| Py150          | Yes; due to disjoint repositories in train, OOD validation, and OOD test set; there are also ID validation and ID test set from same repositories as the training set. | Yes                                                          |                                                 |

Importantly, the authors note that performance drop is a necessary condition of distribution shifts: distribution shifts do not lead to performance drop on the test set. However, if we observe that degraded performance, we could consider either domain generalization or subpopulation shift as potential causes. Here are the examples:

- Time Shift in Amazon Review Dataset:
- Time and User Shift in Yelp Dataset:

# Additional Notes

- Deciding Type of Distribution Shift

    As long as there are no clearly disjoint train, validation, and test set as in Amazon Reviews and Py150 datasets, then there is no domain generalization issue; presence of a few unseen users in the validation or test set should not be considered as the domain generalization case.

- Challenge Sets vs. Distribution Shifts

    The CheckList-style challenge sets, such as HANS, PAWS, CheckList, and counterfactually-augmented datasets like [5], are intentionally created different from the training set.



# Reference

1. [[2201.00299] Improving Out-of-Distribution Robustness via Selective Augmentation](https://arxiv.org/abs/2201.00299) (Yao et al.): This paper proposes the LISA method that performs best on the Amazon dataset according to the [leaderboard](https://wilds.stanford.edu/leaderboard/#amazon).
2. [[2104.09937] Gradient Matching for Domain Generalization](https://arxiv.org/abs/2104.09937) (Shi et al.): This paper proposes the FISH method that performs best on the CivilComments dataset on the [leaderboard](https://wilds.stanford.edu/leaderboard/#civilcomments).
3. [Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification](https://proceedings.mlr.press/v81/buolamwini18a/buolamwini18a.pdf) (Buolamwini and Gebru, FAccT '18)
4. [Racial Disparities in Automated Speech Recognition](https://www.pnas.org/doi/full/10.1073/pnas.1915768117) (Koenecke et al., PNAS)
5. [[2010.02114] Explaining The Efficacy of Counterfactually Augmented Data](https://arxiv.org/abs/2010.02114) (Kaushik et al., ICLR '20)
6. [[2004.14444] The Effect of Natural Distribution Shift on Question Answering Models](https://arxiv.org/abs/2004.14444) (Miller et al.): This paper trains 100+ QA models and tests them across different domains.
7. [Selective Question Answering under Domain Shift](https://aclanthology.org/2020.acl-main.503) (Kamath et al., ACL 2020): This paper creates a test set of mixture of ID and OOD domains.