---
title: Research Notes | Research Questions
tags: 
categories:
- Research
---

# Overview

Here I document a list of general research questions that warrants searching, reading, thinking, and rethinking. 

# General Topics

- Model Capacity

    - What is the broadly applicable measure of model capacity similar to [a hardware performance benchmark](https://www.maxon.net/en/cinebench) that helps practitioners pick up the suitable model to start building their applications? 
        - Note: Model capacity mostly determines the performance upper bound of a model. The actual model performance may also related to how the model is trained with what set of hyperparameters.
        - Hypothesis: A straightforward choice is the number of parameters a model has. However, one may question the correlation between the parameter count and this measure, i.e., the parameter count may need to be a valid proxy for the model capacity.

- Generalization

    - Existence of Universal Generalization

        Specifically, suppose there are $K$ texts existing in the world at time $t$, and they are all labeled by an oracle; if we fine-tune a `bert-base-uncased` with $k \ll K$ samples as a classification model, is there any hope that this fine-tuned model perform *reasonably well* (needs more precise definition) on all $(K-k)$ samples.

        - Experiment: We could only approximate the oracle by some knowingly most capable models like GPT-4. We therefore have two datasets, one from (a) original annotations and (b) the other from oracle (approximated by GPT-4) annotations. Could the model fine-tuned on dataset (b) generalize better than (a)?
        - Question: Despite the generalization, could the fine-tuned model also inherit the *bias*  (needs more precise definition) of GPT-4?
    
    - Text Classification, Annotation Bias, and Spurious Correlation
    
        Does text classification work by relying on the spurious correlation (likely due to annotation bias where the annotators seek to take the shortcuts to complete their assigned tasks as soon as possible) between a limited number of words in the input text and output label? Therefore, is the better model indeed the model that better exploits the spurious correlation?
    
        - Hypothesis: If the $K$ samples are all annotated by an oracle, then any *reasonably capable* model (needs more precise definition) can generalize well. 
        - Tentative Experiment: If we replace the words in the list with their hypernyms, will the system performance drop?
    
    - Life Long and Continual Learning
    
        Suppose we have a generalizable model at time $t$ if we want the model to serve the users indefinitely. What are the strategies that make the model generalize well across time?
    
- Data Distribution

    In machine learning theory, we often encounter concepts such as i.i.d. Understanding "distribution" for tabular data is straightforward, where the list of variables forms a joint distribution that predicts the label y. However, what could be considered a "distribution" for texts is less clear. Note that this is possible for images, for example, modeling the gray-scale images as a Dirichlet multimodal distribution that predicts digits from 0 to 9.

# Special Issues



# References

