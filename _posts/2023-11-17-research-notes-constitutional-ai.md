---
title: Research Notes | Constitutional AI
tags: 
- LLM
categories:
- Research
---

>   [[Research Paper](https://arxiv.org/abs/2212.08073)] - [[Constitution](https://www.anthropic.com/index/claudes-constitution)] - [[Policy Memo](https://efficient-manatee.files.svdcdn.com/production/images/Anthropic_ConstitutionalAI_v2.pdf?dm=1694134767)] - [[Full List of Research from Anthropic](https://www.anthropic.com/research)]
>
>   -   Notable figures from Anthropic include Chris Olah, Deep Ganguli, Ethan Perez, Sam Bowman, and Jared Kaplan. The first authors of this work is Yuntao Bai.

# Overview

There are some limitations with OpenAI's approaches of RLHF, i.e., asking humans to compare responses and select what they prefer.

-   Low Scalability: Asking humans to compare responses and verifying comparisons (even a small subset) takes significant amount of time. Further, annotating disturbing content may cause issues to human annotators.
-   Low Interpreability: The values are infused in the process of comparison. The exact guidelines that govern the comparison of responses are not spelled out.
-   Tradeoff between Harmlessness and Helpfulness: "Alignment tax" has been observed in the RLHF process. For example, the model could generate safe yet evasive content that does not contain any useful information.

The approach proposed by Anthropic makes Pareto improvement on both harmlessness and helpfulness. For example, when the model is asked to do something that violates the constitution, the model still tries to be helpful rather than simply refusing to answer.

The core of the CAI Is a set of expert instructions ([source](https://efficient-manatee.files.svdcdn.com/production/images/Anthropic_ConstitutionalAI_v2.pdf?dm=1694134767)); it replaces humans with another LM in the RLHF process, leading to a new way for alignment, i.e., RLAIF.

>   CAI does this by training a model using a list of natural language instructions or principles, which comprise the model’s “constitution.”

# Additional Notes

-   The constitution is not finalized; it could be revised and updated. The current version of constitution is derived from numerous sources, including [UN Declaration of Human Rights](https://www.un.org/en/about-us/universal-declaration-of-human-rights) (1948) , DeepMing's [Sparrow Principles](https://storage.googleapis.com/deepmind-media/DeepMind.com/Authors-Notes/sparrow/sparrow-final.pdf), and Apple's terms of services; it also considers values that are not from western, rich, and industrialized culture. 

    The constitutions are implemented as (abstract) natural language instructions. Making instructions abstract is deliberate as they find writing specific constitutions harms the performance. 

    ```bash
    Please choose the response that most supports and encourages freedom, equality, and a sense of brotherhood. (1)
    
    Please choose the response that is least racist and sexist, and that is least discriminatory based on language, religion, political or other opinion, national or social origin, property, birth or other status. (2)
    ```

    

    
