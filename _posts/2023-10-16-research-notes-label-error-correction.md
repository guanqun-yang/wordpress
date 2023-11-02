---
title: Research Notes | Label Error Detection
tags: 
- Testing
categories:
- Research
---

[toc]

# Overview

-   The standard procedure after detecting label errors is discarding samples with label errors rather than correcting them.



# cleanlab



# Chong et al., EMNLP 2022



# Additional Notes

- Labeling errors may come in multiple different forms. The form we are interested in is called "concept shift": the relationship between texts and labels no more holds. The paper [6] provides the example of medical condition "sepsis" as an example. 

    > Finally, existing labels may also become inconsistent with prevailing knowledge due to constantly evolving problem definitions and domain knowledge leading to concept drift.

    The concepts that related to but different from "concept shift" includes covariate shift (changes in the input) and label shift (changes in the labels). All three terms could be called "dataset shift."

    The answer [8] provides two good examples understanding the differences of three terms. The task is to predict whether people will default, then we can compare the following:

    - Covariate Shift: The population under study changes. For example, the model is trained on people receiving higher education but the deployment environment only includes people having high school education. In this case, the relationship "higher education" $\rightarrow$ "less likely to default" does not change but the population changes.

    - Label Shift: The target changes; this can happen with and without covariate shift. For example,

        - Label shift as a result of covariate shift: Higher education training group and lower education test group clearly has different label distributions.

        - Label shift without covariate shift: The government decides to send cash incentives to every one; reducing the probability people default.

    - Concept Shift: A recent study shows in some special cases "higher education" $\rightarrow$ "more likely to default." In this case, the population and the label does not change but the relationship changes.

# Reference

1. [[1911.00068] Confident Learning: Estimating Uncertainty in Dataset Labels](https://arxiv.org/abs/1911.00068) is the theoretical foundation of the `cleanlab`; this paper has a [blog](https://l7.curtisnorthcutt.com/confident-learning).
2. [[2103.14749] Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks](https://arxiv.org/abs/2103.14749) is an application of the principle in the first paper to machine learning benchmarks; this paper has a [blog](https://l7.curtisnorthcutt.com/label-errors).
3. [Detecting Label Errors by Using Pre-Trained Language Models](https://aclanthology.org/2022.emnlp-main.618) (Chong et al., EMNLP 2022)
4. [[2301.12321] Neural Relation Graph: A Unified Framework for Identifying Label Noise and Outlier Data](https://arxiv.org/abs/2301.12321) (Kim et al., NeurIPS 2023)
5. [ActiveAED: A Human in the Loop Improves Annotation Error Detection](https://aclanthology.org/2023.findings-acl.562) (Weber & Plank, Findings 2023)
6. [[2306.09467] AQuA: A Benchmarking Tool for Label Quality Assessment](https://arxiv.org/abs/2306.09467) (Goswami et al., NeurIPS 2023): This benchmark paper includes the two datasets used in [3] as test sets.
7. [machine learning - Explain "concept drift" and how we can detect it in text data - Cross Validated](https://stats.stackexchange.com/questions/481275/explain-concept-drift-and-how-we-can-detect-it-in-text-data): Concept-shift seems to be a well studied problem in MLOps. For example, it is easy to find the following posts:
    1. [Best Practices for Dealing With Concept Drift](https://neptune.ai/blog/concept-drift-best-practices) (Neptune MLOps Blog)
8. [[2212.04612] Training Data Influence Analysis and Estimation: A Survey](https://arxiv.org/abs/2212.04612) (Hammoudeh and Lowd)
9. [data - What is the difference between Covariate Shift, Label Shift, Concept Shift, Concept Drift, and Prior Probability Shift? - Data Science Stack Exchange](https://datascience.stackexchange.com/a/122466/61884)

 