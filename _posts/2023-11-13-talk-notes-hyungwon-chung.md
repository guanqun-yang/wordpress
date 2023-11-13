---
title: Talk Notes |  LLM and RLHF
tags: 
- LLM
- RLHF
categories:
- Talk
---

>   [Talk on LLM] - [[Talk on RLHF](https://www.youtube.com/watch?v=zjrM-MW-0y0)]
>
>   -   The presenter Hyungwon Chung is a research engineer at OpenAI; he was with Google. He was doing mechanical engineering during Ph.D. that is completely irrelevant (aka. pressure-retarded osmosis) from machine learning.

-   Instruction tuning is the technique that will **almost universally** beneficial to decoder only model and encoder-decoder model to improve their performances: the answer to "should I try instruction tuning" is almost always "yes."

-   There are two flavors of instruction tuning

    -   Using Mixture of Academic Datasets: Flan and T0. 
    -   Using User Traffic: For example, InstructGPT and ChatGPT.

-   Task Diversity and Model Size are Important

    -   The T0 by the presenter collects 1836 tasks; it is still the largest collections as of November 2023. The authors show the linear scaling law of model size and normalized performance on the held-out tasks. Further, when the number of tasks increase, the line is lifted upwards with a **double digit gain**. Further, it is important to have combine the non-CoT and CoT data together.

    -   However, the performance quickly plateaus even when there are more tasks. This is likely due to limited diversity of academic datastes.

# Reference

1.   [[2210.11416] Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) (Chung et al., including Jason Wei)
2.   [[2301.13688] The Flan Collection: Designing Data and Methods for Effective Instruction Tuning](https://arxiv.org/abs/2301.13688) (Longpre et al.)
