---
title: Research Notes | Debugging Machine Learning Models
tags: 
- Testing
- Debugging
categories:
- Research
---

[toc]

# Overview

The edited knowledge in this paper is in the form of triplets. Given the prompt `Eiffel Tower is located in the city of`, the original model will output `Paris` as expected. However, after model editing, the output could be other tokens with high probability. For example, `Seattle`.

Suppose we have an input $x$ and its original output is $y := \mathcal{M}(x)$, if we apply some intervention to $\mathcal{M}(\cdot)$ and expect the future output to be $y'$, we require the editing to be reliable, local, and general:

- Reliable: The edited model should output $y'$ with a high probability.
- Local: The output of anything semantically different from $x$ should not change.
- General (or Consistent): The output of anything semantically equivalent to $x$ should also change.

# KnowledgeEditor

# MEND

# MEMIT

The architecture studied in this paper (and ROME) is limited to GPT models.

# Experiments

## Datasets
The canonical tasks of model editing includes fact-checking on FEVER and QA with the zsRE datasets.

- For FEVER, the editing dataset is based on the original input and flipped label.
- For zsRE, the editing dataset is based on the original input and an answer that is not top-1.


| Paper                       | Fact Checking                                         | QA                                                      | Generation | Note                                                         |
| --------------------------- | ----------------------------------------------------- | ------------------------------------------------------- | ---------- | ------------------------------------------------------------ |
| MEMIT [1]                   | N/A                                                   | zsRE and CounterFact                                    | N/A        | There are two intermediate works ROME and SEARC. But they are omitted as the best model is MEMIT. |
| MEND [5]                    | [Binary FEVER](https://huggingface.co/datasets/fever) | [zsRE](https://huggingface.co/datasets/qa_zre)          | Wikitext   | The first two tasks are chosen same as De Cao et al.; Wikitext is an additional dataset. |
| KnowledgeEditor [4]         | Binary FEVER                                          | zsRE                                                    | N/A        |                                                              |
| Constrained Fine-Tuning [3] | N/A                                                   | zsRE and [T-REx](https://aclanthology.org/L18-1544.pdf) | N/A        |                                                              |
| ENN [4]                     | N/A                                                   | N/A                                                     | N/A        | This early work experiments on CIFAR-10 and MT tasks.        |


# Additional Notes

-   The RDF triplet may be the most unambiguous way to express instances of a specification; it is a classical way to represent knowledge and could be bidirectionally converted from and to a SQL database ([Wikipedia](https://en.wikipedia.org/wiki/Semantic_triple)).
-   The overarching research field is called "mechanistic interpretibility."
-   Knowledge editing is thought to be difficult because now knowledge is stored distributionally rather than symbols. However, the paper [2] finds that the localization is quite concentrated in MLPs; the authors focus on MLPs because they believe the attention is too complicated to study.

> - MLPs are storing information while attention is gathering information: the information "Seattle" is in one specific location of GPT-2 before the "the space needle is located at" is asked.


# Reference

[Kevin Meng](https://mengk.me/) and [David Bau](https://baulab.info/) have published a series of works ([1] and [2]) on knowledge editing for transformers. [3] through [6] are the predecessors to the proposed work; they could at most scale to 75 edits.

1. [[2210.07229] Mass-Editing Memory in a Transformer](https://arxiv.org/abs/2210.07229) (MEMIT system).

2. [[2202.05262] Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) (ROME system).

3. [[2012.00363] Modifying Memories in Transformer Models](https://arxiv.org/abs/2012.00363): This paper is the first to study the problem of fact editing transformers. The authors propose to fine-tune the models' first and last transformer block on the modified facts $\mathcal{D} _ M$ while constraining the parameter within a small space.
    $$
    \min _ {\theta \in \Theta} \frac{1}{m} \sum _ {x \in \mathcal{D}_M} L(x;\theta)\quad s.t. \Vert \theta - \theta_0 \Vert \leq \delta
    $$

4. [[2004.00345] Editable Neural Networks](https://arxiv.org/abs/2004.00345) (Sinitsin et al., ICLR 2020) (ENN system): This paper is the first to apply meta-learning to model editing; it is a precursor to follow-up works [5], [6], and [7]. Besides, it mentions the following important observations:

    - The goal of model editing is quickly patching critical mistakes made by a neural model. The problem precludes (1) retraining with augmented dataset because it is slow, and (2) manual cache as it does not adapt to diverse input changes.

5. [Editing Factual Knowledge in Language Models](https://aclanthology.org/2021.emnlp-main.522) (De Cao et al., EMNLP 2021) (KnowledgeEditor system): The authors observe that the previous methods [3] and [4] have following limitations in their edited models:

    - Unreliable Edits: For sentences that are different from $x$, the behaviors **should not have** changed.
    - Inconsistent Edits: For sentences that are semantically equivalent to $x$, the behaviors **should have** changed.

    Furthermore, the method [4] also requires expensive retraining.

6. [[2110.11309] Fast Model Editing at Scale](https://arxiv.org/abs/2110.11309) (Mitchell et al.) (MEND system): This paper improves the De Cao et al. in editing models with a scale of 10B parameter. On smaller models, the ENN model is **better** than KnowledgeEditor.

7. [[2206.06520] Memory-Based Model Editing at Scale](https://arxiv.org/abs/2206.06520) (Mitchell et al.) (SEARC system): The authors do not release code for SEARC.

8. [Transformer Feed-Forward Layers Are Key-Value Memories](https://aclanthology.org/2021.emnlp-main.446) (Geva et al., EMNLP 2021): This paper helps the main paper constrain the editing target to the MLP layers.

9. [[2308.07269] EasyEdit: An Easy-to-use Knowledge Editing Framework for Large Language Models](https://arxiv.org/abs/2308.07269) (Wang et al.)

The following are other useful references:

- [semantic web - Translating a complex Sentence into set of SPO triple (RDF) (maybe with reification) - Stack Overflow](https://stackoverflow.com/a/57732900/7784797): The user notes that it is difficult to convert the natural language into a standard structure in a definitive way. Some of the approximations include dependency parsing, constituency parsing, knowledge graph, and First-Order Logic (FOL).
- [ROME: Locating and Editing Factual Associations in GPT (Paper Explained & Author Interview) - YouTube](https://www.youtube.com/watch?v=_NMQyOu2HTo)
