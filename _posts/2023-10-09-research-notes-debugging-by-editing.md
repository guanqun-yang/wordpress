---
title: Research Notes | Debugging Machine Learning Models
tags: 
- Testing
categories:
- Research
---

# Additional Notes

-   The RDF triplet may be the most unambiguous way to express instances of a specification; it is a classical way to represent knowledge and could be bidirectionally converted from and to a SQL database ([Wikipedia](https://en.wikipedia.org/wiki/Semantic_triple)).

# Reference

[Kevin Meng](https://mengk.me/) and [David Bau](https://baulab.info/) have published a series of works on knowledge editing for transformers:

1.   [[2210.07229] Mass-Editing Memory in a Transformer](https://arxiv.org/abs/2210.07229) (MEMIT system).
2.   [[2202.05262] Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) (ROME system).

The following are other predecessors to the proposed approach:

3. [[2012.00363] Modifying Memories in Transformer Models](https://arxiv.org/abs/2012.00363): This paper is the first to study the problem of fact editing transformers. The authors propose to fine-tune the models' first and last transformer block on the modified facts $\mathcal{D} _ M$ while constraining the parameter within a small space.
    $$
    \min _ {\theta \in \Theta} \frac{1}{m} \sum _ {x \in \mathcal{D}_M} L(x;\theta)\quad s.t. \Vert \theta - \theta_0 \Vert \leq \delta
    $$

The following are other useful references:

4. [semantic web - Translating a complex Sentence into set of SPO triple (RDF) (maybe with reification) - Stack Overflow](https://stackoverflow.com/a/57732900/7784797): The user notes that it is difficult to convert the natural language into a standard structure in a definitive way. Some of the approximations include dependency parsing, constituency parsing, knowledge graph, and First-Order Logic (FOL).
