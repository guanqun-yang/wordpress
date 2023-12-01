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

-   The description above only considers the language model itself. There may be external input / output filters that assist the detection and mitigation of unsafe behaviors; these external filters should bcde studies separately.
-   The LM itself may or may not go through the process of enhancing safety. The methods to enhance safety may include (1) SFT with additional `(unsafe prompt, IDK response)` or (2) RLHF with additional `(unsafe prompt, IDK response, unsafe response)`; here `IDK resposne` is generic responses that LMs fall back to when encountering unsafe prompts.

# Red Teaming



# Resources

-   An comprehensive [wiki](https://alignmentsurvey.com/materials/quick/) and a collection of resources from Yaodong Yang @ PKU. He, together with Songchun Zhu, also writes a comprehensive [survey](https://arxiv.org/abs/2310.19852) on AI alignment; it has [a Chinese version](https://alignmentsurvey.com/uploads/AI-Alignment-A-Comprehensive-Survey-CN.pdf).

# Reference

## Safety Alignment

1.   [[2310.12773] Safe RLHF: Safe Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2310.12773)

2.   [[2307.04657] BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset](https://arxiv.org/abs/2307.04657) (PKU-Alignment)

     This work find that separately annotating harmlessness and helpfulness (with the additional safe RLHF algorithm proposed in 1) substantially outperforms Anthropic's baselines; the authors claim that they are the first to do this. The author also open-source the datasets (1) [a SFT (or classification) dataset](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) that is used to train safety classifier and (2) [a RLHF dataset](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) that is used to fine-tune an LM (Alpaca in the paper).

     <img src="https://raw.githubusercontent.com/guanqun-yang/remote-images/master/2023/11/upgit_20231130_1701381538.png" alt="image-20231130165858750" style="zoom:67%;" />

     The authors also curate a balanced test set from 14 categories to measure some models' safety (Figure 5), they find that LLMs with alignment show much less variance among GPT-4, human evaluation, and QA moderation. Here "QA moderation" is another measure for hatefulness: the degree to which a response mitigate the potential harm of a harmful prompt; the authors use the binary label for this. Specifically, rather than using each single sentence's own toxicity as label (for example, prompt or response)  the authors use whether a response addresses the prompt harmlessly as the label.

     <img src="https://raw.githubusercontent.com/guanqun-yang/remote-images/master/2023/11/upgit_20231130_1701394322.png" alt="image-20231130203202302" style="zoom: 67%;" /><img src="https://raw.githubusercontent.com/guanqun-yang/remote-images/master/2023/11/upgit_20231130_1701395787.png" alt="image-20231130205627290" style="zoom: 67%;" />

     Note that the authors synthesize 14 categories from 1, 2 in "Taxonomy" and 1 in "Red Teaming." The authors acknowledge that these categories are **not** MECE.

     The authors release their models and datasets on HuggingFace hub:
     
     | Model | Name                                                         | Note                                                         |
     | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
     | 1     | [`PKU-Alignment/alpaca-7b-reproduced`](https://huggingface.co/PKU-Alignment/alpaca-7b-reproduced) | The reproduced Alpaca model.                                 |
     | 2     | [`PKU-Alignment/beaver-dam-7b`](https://huggingface.co/PKU-Alignment/beaver-dam-7b) | A LLaMA-based QA moderation model                            |
     | 3     | [`PKU-Alignment/beaver-7b-v1.0-reward`](https://huggingface.co/PKU-Alignment/beaver-7b-v1.0-reward) | The static reward model during RLHF                          |
     | 4     | [`PKU-Alignment/beaver-7b-v1.0-cost`](https://huggingface.co/PKU-Alignment/beaver-7b-v1.0-cost) | The static cost model during RLHF                            |
     | 5     | [`PKU-Alignment/beaver-7b-v1.0`](https://huggingface.co/PKU-Alignment/beaver-7b-v1.0) | The Alpaca model that goes through the safe RLHF process based on 1 |
     
     | Dataset | Name                                                         | Note                                                         |
     | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
     | 1       | [`PKU-Alignment/BeaverTails`](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | A classification dataset with `prompt`, `response`, `category`, and `is_safe` columns. |
     | 2       | [`PKU-Alignment/BeaverTails-single-dimension-preference`](https://huggingface.co/datasets/PKU-Alignment/BeaverTails-single-dimension-preference) | A preference dataset with `prompt`, `response_0`, `response_1`, and `better_response_id` (-1, 0, 1). |
     | 3       | [`PKU-Alignment/BeaverTails-Evaluation`](https://huggingface.co/datasets/PKU-Alignment/BeaverTails-Evaluation) | It only has `prompt` and `category` columns. It is not the test split of the dataset 1 and 2. |
     | 4       | [`PKU-Alignment/PKU-SafeRLHF`](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) | A preference and classification dataset (N=330K) with `prompt`, `response_0`, `response_1`, `is_response_0_safe`, `is_response_1_safe`, `better_response_id`, `safer_response_id`; it has both training and test split. |
     | 5       | [`PKU-Alignment/PKU-SafeRLHF-30K`](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-30K) | Sampled version of 4 with both training and test split.      |
     | 6       | [`PKU-Alignment/PKU-SafeRLHF-10K`](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-10K) | A further sampled version of 4 with only training split available. |
     | 7       | [`PKU-Alignment/processed-hh-rlhf`](https://huggingface.co/datasets/PKU-Alignment/processed-hh-rlhf?) | A reformatted version of the Anthropic dataset for the ease of use; the [original dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf) is formatted in plain text. |
     
     

## Safety Benchmark

1.   [[2308.01263] XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models](https://arxiv.org/abs/2308.01263) (Röttger et al.): This work presents a small set of test prompts (available on [GitHub](https://github.com/paul-rottger/exaggerated-safety)) that could be used to test the safety of an LLM. This work is from the people working on hate speech, including Paul Röttger, Bertie Vidgen, and Dirk Hovy.
2.   [[2308.09662] Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment](https://arxiv.org/abs/2308.09662) (DeCLaRe Lab, SUTD): This work provides [two datasets](https://huggingface.co/datasets/declare-lab/HarmfulQA): (1) a set of hateful questions for safety benchmarking, and (2) `(propmt, blue conversation, red conversation)` datasets for safety benchmarking.
3.   [[2309.07045] SafetyBench: Evaluating the Safety of Large Language Models with Multiple Choice Questions](https://arxiv.org/abs/2309.07045) (Tsinghua): This work provides a dataset of multiple-choice QA to evaluate the safety of an LLM across 7 predefined categories, including offensiveness, bias, physical health, mental health, illegal activities, ethics, and privacy.

## Red Teaming

1.   [[2209.07858] Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://arxiv.org/abs/2209.07858) (Ganguli et al., Anthropic).
2.   [[2202.03286] Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286) (Perez et al., DeepMind and NYU)

## Taxonomy

1.   [[2206.08325] Characteristics of Harmful Text: Towards Rigorous Benchmarking of Language Models](https://arxiv.org/abs/2206.08325) (Rauh et al., DeepMind)
2.   [BBQ: A hand-built bias benchmark for question answering](https://aclanthology.org/2022.findings-acl.165) (Parrish et al., Findings 2022, NYU)