---
title: Reading Notes | Direct Preference Optimization - Your Language Model is Secretly a Reward Model
tags: 
- RLHF
- LLM
categories:
- Reading
---

>   [Semantic Scholar] - [Code] - [Tweet] - [Video]
>
>   Change Logs:
>
>   - 2023-12-04: First draft.

# Overview

-   DPO belongs to a larger family of algorithms of directly optimizing human preferences. The algorithm assumes there are always a winning solution and a losing solution; this is different from PPO as now the label becomes discrete.
-   Using DPO will alleviate the need for a dedicated library such as `trl`. The only change need to be made is a loss function.

# Reference

1.   [DPO Debate: Is RL needed for RLHF? - YouTube](https://www.youtube.com/watch?v=YJMCSVLRUNs): This is an advanced video by Nathan Lambert.

2.   [[2310.12036] A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036) (DeepMind)

     This is a theoretical paper that reveals the limitations of DPO. 

     It shows two assumptions of RLHF: (1) pairwise comparison could be substituted with pointwise rewards, and (2) an LM trained on the pointwise rewards will generalize from collected data to OOD data.

     

     