---
title: Talk Notes |  Data-Centric AI
tags: 
- Data-Centric AI
categories:
- Talk
---



[toc]

# Overview

The following notes are the data-centric AI IAP course notes from MIT; [Independent Activities Period (IAP)](https://elo.mit.edu/iap/) is a special four-week semester of MIT.

# Lecture 1 - Data-Centric AI vs. Model-Centric AI

-   It is not hard to design fancy models and apply various tricks on the well curated data. However, these models and tricks do not work for real-world data if we do not explicitly consider the real-world complexities and take them into account. Therefore, it is important to focus on data rather than model.

    It turns out there are pervasive label errors on the most cited test sets of different modalities, including text, image, and audio. They could be explored in [`labelerrors.com`](https://labelerrors.com/).

-   To understand why data is important, we could think about kNN algorithm. The accuracy of kNN is purely based on the quality of datasets. However, the kNN is not a data-centric algorithm because it does not modify the labels. 

-   Two Goals of Data-Centric AI

    Rather than modifying loss function, doing HPO, or changing the model itself, we do either of the following:

    -   Designing an algorithm that tries to understand the data and using that information to improve the model. One such example is curriculum learning by Yoshua Bengio; in curriculum learning, the data is not changed but its order is shuffled.
    -   Modifying the dataset itself to improve the models. For example, the confident learning (i.e., **removing** wrong labels before training the model) studied by Curtis Northcutt.

-   What are NOT Data-Centric AI and Data-Centric AI Counterpart

    -   Hand-picking data points you think you will improve a model. $\rightarrow$ Coreset Selection.
    -   Doubling the size of dataset. $\rightarrow$ Data Augmentation. For example, back-translation for texts, rotation and cropping for images. However, we need to first fix label errors before augmenting the data.

-   Typical Examples of Data-Centric AI

    Curtis Northcutt cites Andrew Ng and other sources on the importance of data in machine learning ([1] through [3]). Here are some examples of data-centric AI:

    -   Outlier Detection and Removal. However, this process relies on a validation process on which threshold to choose.
    -   Label Error Detection and Correction
    -   Data Augmentation
    -   Feature Engineering and Selection. For example, solving XOR problem by adding a new column.
    -   Establishing Consensus Labels during Crowd-sourcing.
    -   Active Learning. I want to improve 5% accuracy on the test set but I could afford as little new annotated data as possible.
    -   Curriculum Learning.

-   Data-Centric AI Algorithms are Often Superior to Model-Centric Algorithms

    The model centric approaches (i.e., training less on what a model believes are the bad subset of data) is a much worse idea than the data-centric approach (i.e., confident learning).

![image-20231106162934431](https://raw.githubusercontent.com/guanqun-yang/remote-images/master/2023/11/upgit_20231106_1699306174.png)

-   Root-Causing the Issues - Models or Data
    -   The model should perform well on the slices of data. Slicing means not only sampling data to a smaller number but also reducing the number of classes from a large number to a very small number. For example, rather than classifying images to 1000 classes, we only focus on performance on two classes.
    -   The model should perform similarly on similar datasets (for example, MNIST datasets and other digits dataset).

# Lecture 2 - Label Errors


## Notation

| Notation                                   | Meaning                                                      | Note |
| ------------------------------------------ | ------------------------------------------------------------ | ---- |
| $\tilde{y}$                                | Noisy observed label                                         |      |
| $y ^ *$                                    | True underlying label                                        |      |
| $\mathbf{X} _ {\tilde{y} =i, y ^ {*} = j}$ | A set of examples whose true label is $j$ but they are mislabeled as $i$. |      |
| $\mathbf{C} _ {\tilde{y} =i, y ^ {*} = j}$ | The size of the dataset above.                               |      |
| $p(\tilde{y} =i, y ^ {*} = j)$             | The joint probability of label $i$ and label $j$; it could be estimated by normalizing $\mathbf{C}$. |      |
| $p(\tilde{y} =i\vert y ^ {*} = j)$         | The transition probability that the label $j$ flips to label $i$; it could also be called flipping rate. |      |

## Categories of Label Errors

When comparing the consensus crowd-sourcing labels and the final label in the dataset, there are 4 types of label errors:

-   Correctable: The given label is wrong and it could be corrected with crowd-sourcing. This is the type of label the lecture **focus** on detecting.
-   Multi-label: The given label and the consensus label are both right. However, more than one label in $\mathcal{Y}$ could be used to label the samples. For example, an image with co-existence of laptop and humans that is incorrectly labeled as "laptop."
-   Neither: The given label and the consensus label are both wrong.
-   Non-agreement: There is no way to tell whether the given label or the consensus label is correct.

There are also two categories of the label errors the presenter does not focus on:

-   Uniform Random Flipping $p(\hat{y} = i \vert y ^ * = j) = \epsilon, \forall i\neq j$: This will show as a symmetric $\mathbf{X}$ matrix. It is **easy** to solve and this type of errors are unlikely to happen in the real world. 
-   Instance-Dependent Label Noise $p(\hat{y} = i \vert y ^ * = j, \mathbf{x})$: This will require a lot of assumptions on the data distribution. Importantly, this type of label errors seldom happen in the real world.

## Uncertainty

There are two sources of uncertainty:

-   Aleatoric Uncertainty: Label noise. It is the difficulty of an sample. This difficulty could come from incorrect label $y$ or the strange distribution of $\mathbf{x}$.
-   Epistemic Uncertainty: Model noise. It is the model's inability to understand the example. For example, the model has never seen similar examples before or the model class is too simple.

## Confident Learning

The focus of the lecture is the correctable errors and the matrix $\mathbf{X}$ is **non-symmetric**.

-   Assumption: Class-Conditional Label Noise
    $$
    p(\hat{y} \vert y ^ *; \mathbf{x} ) = p(\hat{y} \vert y ^ *)
    $$

    -   Interpretation: Given the true label, there is a constant flipping rate for the samples under that true label to other labels.
    -   Rationale: A pig image often confused with a boar image but not other items such as "missiles" and "keyboards." This tendency has nothing to do with what exactly a pig look like in an image but the similarities of the classes.
    -   Motivation: This assumption is made because the LHS couples the aleatoric uncertainy and epistemic uncertainty and this assumption decouples these two uncertainties.

    

-   The reason why we need confident learning is that we could not find a loss threshold and claim the samples above this threshold are label errors.

-   

    







# Reference

1.   [Why it’s time for 'data-centric artificial intelligence' | MIT Sloan](https://mitsloan.mit.edu/ideas-made-to-matter/why-its-time-data-centric-artificial-intelligence)
2.   [Bad Data Costs the U.S. $3 Trillion Per Year](https://hbr.org/2016/09/bad-data-costs-the-u-s-3-trillion-per-year) (Harvard Business Review)
3.   [Bad Data: The $3 Trillion-Per-Year Problem That's Actually Solvable | Entrepreneur](https://www.entrepreneur.com/science-technology/bad-data-the-3-trillion-per-year-problem-thats-actually/393161)

