---
title: Reading Notes | NoisywikiHow - A Benchmark for Learning with Real-world Noisy Labels in Natural Language Processing
tags:
  - Generalization
  - NoisyLabel
categories:
  - Reading
---

> [[Semantic Scholar](https://www.semanticscholar.org/paper/NoisywikiHow%3A-A-Benchmark-for-Learning-with-Noisy-Wu-Ding/35225d37ec210becb5af7c07a4b2af715f5b7e6c)] - [[Code](https://github.com/tangminji/NoisywikiHow)] - [Tweet] - [Video] - [Website] - [Slide]
>
> Change Logs:
>
> - 2023-09-14: First draft. The paper appears at ACL 2023. The code base has very detailed [instructions](https://github.com/tangminji/NoisywikiHow/blob/main/TORUN.md) on how to reproduce their results.

# Method

- The authors find that the labeling errors are both annotator-dependent and instance-dependent.

# Experiments

- The best performing LNL method on the benchmark is SEAL [1]: one could also consider MixUp regularization [2]. All other LNL methods have almost **indistinguishable** difference as the base models, i.e., not doing any intervention on the training process.

# Additional Note

# Comments

- The reason why creating a new dataset is necessary is that the users could customize the noise level to compare performances of different algorithms in a controlled setting. 

# Reference

1. [[2012.05458] Beyond Class-Conditional Assumption: A Primary Attempt to Combat Instance-Dependent Label Noise](https://arxiv.org/abs/2012.05458) (Chen et al. AAAI 2021).
2. [[1710.09412] mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412) (Zhang et al., ICLR 2018, 7.6K citations).
3. [Nonlinear Mixup: Out-of-Manifold Data Augmentation for Text Classification](https://ojs.aaai.org/index.php/AAAI/article/view/5822) (Guo, AAAI 2020). One application of MixUp regularization in NLP. It is based on a CNN classifier and the improvement is quite marginal.
4. [[2006.06049] On Mixup Regularization](https://arxiv.org/abs/2006.06049) (Carratino et al., JMLR): A theoretical analysis of MixUp regularization.
5. [Learning with Noisy Labels](https://proceedings.neurips.cc/paper_files/paper/2013/file/3871bd64012152bfb53fdf04b401193f-Paper.pdf) (Natarajan et al., NIPS 2013): This paper is the first paper that (theoretically) studies LNL. It considers the binary classification problem where labels are **randomly** flipped, which is theoretically appealing but less relevant empirically according to the main paper.