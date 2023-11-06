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
    -   Establishing Concensus Labels during Crowdsourcing.
    -   Active Learning. I want to improve 5% accuracy on the test set but I could afford as little new annotated data as possible.
    -   Curriculum Learing.

-   Data-Centric AI Algorithms are Often Superior to Model-Centric Algorithms

    The model centric approaches (i.e., training less on what a model believes are the bad subset of data) is a much worse idea than the data-centric approach (i.e., confident learning).

![image-20231106162934431](https://raw.githubusercontent.com/guanqun-yang/remote-images/master/2023/11/upgit_20231106_1699306174.png)

-   Root-Causing the Issues - Models or Data
    -   The model should perform well on the slices of data. Slicing means not only sampling data to a smaller number but also reducing the number of classes from a large number to a very small number. For example, rather than classifying images to 1000 classes, we only focus on performance on two classes.

# Lecture 2 - Label Errors



# Reference

1.   [Why it’s time for 'data-centric artificial intelligence' | MIT Sloan](https://mitsloan.mit.edu/ideas-made-to-matter/why-its-time-data-centric-artificial-intelligence)
2.   [Bad Data Costs the U.S. $3 Trillion Per Year](https://hbr.org/2016/09/bad-data-costs-the-u-s-3-trillion-per-year) (Harvard Business Review)
3.   [Bad Data: The $3 Trillion-Per-Year Problem That's Actually Solvable | Entrepreneur](https://www.entrepreneur.com/science-technology/bad-data-the-3-trillion-per-year-problem-thats-actually/393161)

