---
title: Research Notes | Benchmarking LLM Safety
tags: 
- LLM
- Safety
categories:
- Research
---

[toc]

# Problem Description

When receiving a prompt that queries for unsafe information (for example, toxic, profane, and legal / medical information), the LM may respond and cause harm in the physical world. There are several ways to diagnose LM weakness: 

-   Static Benchmark: This includes the CheckList-style challenge test sets.
    -   Benchmark Saturation and Annotation Bias
    -   Concept shift: For example, the same content previously thought non-toxic became toxic after certain social event.
    -   Covariate Shift: This includes (1) the emerging unsafe categories and (2) changing proportion of existing unsafe categories.

-   Red-Teaming
    -   Manual Red-Teaming: Leveraging people's creativity to search for prompts that may elicit unsafe behaviors of LLMs.
    -   Automated Red-Teaming

Note that

-   The description above only considers the language model itself. There may be external input / output filters that assist the detection and mitigation of unsafe behaviors; these external filters should be studies separately.
-   The LM itself may or may not go through the process of enhancing safety. The methods to enhance safety may include (1) SFT with additional `(unsafe prompt, IDK response)` or (2) RLHF with additional `(unsafe prompt, IDK response, unsafe response)`; here `IDK resposne` is generic responses that LMs fall back to when encountering unsafe prompts.

# Red Teaming



# Resources

-   An comprehensive [wiki](https://alignmentsurvey.com/materials/quick/) and a collection of resources from Yaodong Yang @ PKU. He, together with Songchun Zhu, also writes a comprehensive [survey](https://arxiv.org/abs/2310.19852) on AI alignment; it has [a Chinese version](https://alignmentsurvey.com/uploads/AI-Alignment-A-Comprehensive-Survey-CN.pdf).

# Reference

## Safety Alignment

1.   [[2307.04657] BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset](https://arxiv.org/abs/2307.04657) (PKU-Alignment): This work provides a SFT dataset, a RLHF dataset, and the resulting model that fine-tunes upon `LLaMA2`; the authors claim a better safety than the base model.

## Safety Benchmark

1.   [[2308.01263] XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models](https://arxiv.org/abs/2308.01263) (Röttger et al.): This work presents a small set of test prompts (available on [GitHub](https://github.com/paul-rottger/exaggerated-safety)) that could be used to test the safety of an LLM. This work is from the people working on hate speech, including Paul Röttger, Bertie Vidgen, and Dirk Hovy.
2.   [[2308.09662] Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment](https://arxiv.org/abs/2308.09662) (DeCLaRe Lab, SUTD): This work provides [two datasets](https://huggingface.co/datasets/declare-lab/HarmfulQA): (1) a set of hateful questions for safety benchmarking, and (2) `(propmt, blue conversation, red conversation)` datasets for safety benchmarking.
3.   [[2309.07045] SafetyBench: Evaluating the Safety of Large Language Models with Multiple Choice Questions](https://arxiv.org/abs/2309.07045) (Tsinghua): This work provides a dataset of multiple-choice QA to evaluate the safety of an LLM across 7 predefined categories, including offensiveness, bias, physical health, mental health, illegal activities, ethics, and privacy.

## Red Teaming

1.   [[2209.07858] Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://arxiv.org/abs/2209.07858) (Ganguli et al., Anthropic).
2.   [[2202.03286] Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286) (Perez et al., DeepMind and NYU)