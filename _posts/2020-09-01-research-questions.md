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

        > - Note: Model capacity mostly determines the performance upper bound of a model. The actual model performance may also related to how the model is trained with what set of hyperparameters.
        > - Hypothesis: A straightforward choice is the number of parameters a model has. However, one may question the correlation between the parameter count and this measure, i.e., the parameter count may need to be a valid proxy for the model capacity.

- Generalization

    - Existence of Universal Generalization

        Specifically, suppose there are $K$ texts existing in the world at time $t$, and they are all labeled by an oracle; if we fine-tune a `bert-base-uncased` with $k \ll K$ samples as a classification model, is there any hope that this fine-tuned model perform *reasonably well* (needs more precise definition) on all $(K-k)$ samples.

        > - Experiment: We could only approximate the oracle by some knowingly most capable models like GPT-4. We therefore have two datasets, one from (a) original annotations and (b) the other from oracle (approximated by GPT-4) annotations. Could the model fine-tuned on dataset (b) generalize better than (a)?
        > - Question: Despite the generalization, could the fine-tuned model also inherit the *bias*  (needs more precise definition) of GPT-4?
    
    - Text Classification, Annotation Bias, and Spurious Correlation
    
        Does text classification work by relying on the spurious correlation (likely due to annotation bias where the annotators seek to take the shortcuts to complete their assigned tasks as soon as possible) between a limited number of words in the input text and output label? Therefore, is the better model indeed the model that better exploits the spurious correlation?
    
        > - Hypothesis: If the $K$ samples are all annotated by an oracle, then any *reasonably capable* model (needs more precise definition) can generalize well. 
        > - Tentative Experiment: If we replace the words in the list with their hypernyms, will the system performance drop?
    
    - Life Long and Continual Learning
    
        Suppose we have a generalizable model at time $t$ if we want the model to serve the users indefinitely. What are the strategies that make the model generalize well across time?
    
- Data Distribution

    In machine learning theory, we often encounter concepts such as "i.i.d." Understanding "distribution" for tabular data is straightforward, where the list of variables forms a joint distribution that predicts the label y. However, what could be considered a "distribution" for texts is less clear. Note that this is possible for images, for example, modeling the gray-scale images as a Dirichlet multimodal distribution that predicts digits from 0 to 9.
    
- Data Annotation

    The labels in the NLP tasks have different levels of subjectivity. For example, the grammatical error correction is less subjective, sentiment classification is moderately subjective, and the topics like hate speech, suicidal ideation [1], and empathy [2] are either extremely subjective or requires expert knowledge.

    The difficulty here it to mitigate the ambiguity during data annotation and make sure the information available in texts matches well with the label. Ideally, if we know the true underlying label of a text, we could fine-tune any reasonably capable model to generalize well.
    
- Data Selection and Data Pruning for Classification Tasks

    As of 2023-10-11, I have not seen a single work on the data selection for classification tasks; there are plenty of works on optimizing data mixture for language model pretraining. One likely reason why this happens is that the quality of classification datasets depends on **both** texts and labels; investigating the label quality is hard.

# Special Issues

- Improving Machine Learning Models with Specifications

    The difference of testing machine learning models versus testing traditional software is that the action items for the latter is generally known after testing and could be executed with high precision, while for the former, we do not know how we will improve the model.

    Suppose the model could already achieve high accuracy on the standard test set (it is indeed the train-to-train setting if we follow the WILDS paper), which means the model architecture, training objective, and hyperparameters are not responsible for the lower performance on the artificially created challenging set, the most straightforward way to improve the model performance is data augmentation. The naive way is to **blindly** collect more data that are **wishfully relevant** as an augmentation so we **expect** the performance could improve.

    However, this blindness hampers the efficiency of improving the models: the only feedback signal is the single scalar (i.e., failure rate) **after** we have trained and evaluated the model; we should have a feedback signal **before** we train the model. 

    > - Unverified Hypothesis: the feedback signal is highly (inversely) correlated with the failure rate on the benchmark.

    Formally, we have a list of specifications in the format of $(s_1, D _ 1, D _ 1 ^ \text{heldout}), (s _ 2, D _ 2, D _ 2 ^ \text{heldout}), \cdots$, the model $\mathcal{M}_0$ trained on $D _ \text{train}$ does well on $D _ \text{train} ^\text{heldout}$ but poorly on $D _ 1 \cup D _  2 \cup D _ 3 \cdots$ as indicated by failure rate $\mathrm{FR}$. We additionally have a new labeled dataset $D _ \text{unused}$. The goal is to sample $D _ \text{unused}$ using $(s_1,D _ 1 ^ \text{heldout}), (s _ 2, D _ 2 ^ \text{heldout}), \cdots$: $\mathrm{Sample}(D _ \text{unused})$; we also have a random sample with same size $\mathrm{RandomSample}(D _ \text{unused})$ as baseline.

    > - Note: The $D _ i$ and $D _ i ^ \text{heldout}$ are completely different. For example, if the specification $s _ i$ is operationalized through templates, these two sets are disjoint in terms of templates. What we are certain about  $D _ i$ and $D _ i ^ \text{heldout}$ is that they are ideally sufficient and necessary with respect to $s _ i$; practically, the semantic underspecification of them are low [3].
    
    There are a lot of things we could do with $\mathrm{RandomSample}(D _ \text{unused})$ and $\mathrm{Sample}(D _ \text{unused})$. For example
    
    - Fine-tuning a model from scratch using $\mathrm{RandomSample}(D _ \text{unused}) \cup \mathcal{D} _ \text{train}$.
    - Patching the model using constrained fine-tuning [4] and related approaches.
    
    Whichever method we choose, if we denote the intervention with $\mathrm{RandomSample}(D _ \text{unused})$ as $\mathcal{M} _ 1$ and $\mathrm{Sample}(D _ \text{unused})$ as $\mathcal{M} _ 2$. We expect the following conditions will hold:
    
    - $D _ \text{train} ^ \text{heldout}$: $\mathcal{M} _ 0 \approx \mathcal{M} _ 1 \approx \mathcal{M} _ 2$.
    - $D _ 1 \cup D _  2 \cup D _ 3 \cdots$: $\mathcal{M} _ 2 \ll \mathcal{M} _ 0$, $\mathcal{M} _ 2 \ll \mathcal{M} _ 1$. That is, the specification-following data selection improves over random selection on the specification-based benchmarks.
    
    > - Assumption: The samples $x _ {ij}$ is fully specified by the specification $s _ i$.
    > - Note: If the annotations of a dataset strictly follows the annotation codebook, then the machine learning learns the specifications in the codebook. The process described above is a reverse process: we have a model that is already trained by others; we want to use the model in a new application but do not want to or can not afford to relabel the entire dataset, what is the minimal intervention we could apply to the dataset so that the model could quickly meets my specifications?

- Detecting Inconsistent Labels with Specifications

# References

1. [ScAN: Suicide Attempt and Ideation Events Dataset](https://aclanthology.org/2022.naacl-main.75) (Rawat et al., NAACL 2022)

2. [A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support](https://aclanthology.org/2020.emnlp-main.425) (Sharma et al., EMNLP 2020)

3. [Dealing with Semantic Underspecification in Multimodal NLP](https://aclanthology.org/2023.acl-long.675) (Pezzelle, ACL 2023)

4. [[2012.00363] Modifying Memories in Transformer Models](https://arxiv.org/abs/2012.00363) (Zhu et al.)

5. `clearnlab`

    1. [[1911.00068] Confident Learning: Estimating Uncertainty in Dataset Labels](https://arxiv.org/abs/1911.00068) is the theoretical foundation of the `cleanlab`; this paper has a [blog](https://l7.curtisnorthcutt.com/confident-learning).

    2. [[2103.14749] Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks](https://arxiv.org/abs/2103.14749) is an application of the principle in the first paper to machine learning benchmarks; this paper has a [blog](https://l7.curtisnorthcutt.com/label-errors).

        
