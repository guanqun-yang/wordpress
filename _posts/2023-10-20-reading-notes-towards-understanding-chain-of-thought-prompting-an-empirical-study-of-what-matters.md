---
title: Reading Notes | Towards Understanding Chain-of-Thought Prompting - An Empirical Study of What Matters
tags: 
- LLM
- Prompt Engineering
categories:
- Reading
---

> [[Semantic Scholar](https://www.semanticscholar.org/paper/Towards-Understanding-Chain-of-Thought-Prompting%3A-Wang-Min/35922cd0d6b17e45320917338e9f98cb5c1a4f6f)] - [Code] - [Tweet] - [Video] - [Website] - [Slide] - [Poster]
>
> Change Logs:
>
> - 2023-10-20: First draft. The paper appears at ACL 2023 as the best paper honorable mention.

# Method


- The experiments of this paper was done on `text-davinci-002` with greedy decoding with temperature 0. The datasets they work on is quite small due to manual efforts required.

- The paper focus on QA and arithmetic reasoning tasks; the authors introduce two concepts:
  
  
  - Bridging Objects
  - Language Template
  
- The authors define the intermediate F1 scores for bridging objects. It is likely that the authors only accept generations that satisfy the predefined template and compute these metrics.

- Observations:


  - The correctness of reasoning during CoT is not important.
  - Query should be (1) relevant and (2) follow the order of reasoning steps.

- Additional Observations:


  - CoT does not make LLMs better; it unlocks the ability already learned by LLMs during pre-training. For example, the conclusions drawn on `text-davinci-002` does not apply to Flan-PaLM; this is because Flan-PaLM has been fine-tuned on the two tasks.

    Given limited resources and an ability to fine-tune the model, we should include more and more data to pre-training or instruction tuning to improve the model rather than focusing the specific prompt engineering tricks.



# Experiment

# Additional Notes

# Reference

