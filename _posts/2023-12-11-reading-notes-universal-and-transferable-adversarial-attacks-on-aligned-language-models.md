---
title: Reading Notes | Universal and Transferable Adversarial Attacks on Aligned Language Models
tags:
  - LLM
  - Safety
categories:
  - Reading
---
> [Semantic Scholar] - [Code] - [Tweet] - [Video] - [Website] - [Slide] - [HuggingFace]
>
> Change Logs:
>
> - 2023-12-07: First Draft.

# Overview
This work tries to add an automatically optimized suffix to the input instruction so that the LM will follow the unsafe instruction to generate unsafe content. 

Specifically, suppose the length of `{affirmation}` is $a$, the algorithm do the following:
- Iterating over $1, 2, \cdots, t$:
	- Forward pass with model using string, the model will output logits of length $(a, \vert \mathcal{V}\vert)$.
	- Compute the cross-entropy loss of these logits and true token IDs (think of it of a $\vert \mathcal{V}\vert$-class classification problem).
	- Backprogation loss back to the tokens in `{suffix i-1}`, we could select tokens with highest gradient to replace and obtain `{suffix i}`.

Finally, we use the optimized `{suffix t}` to put it into test and hope that it will generate the `{affirmation}` token.
```bash
# train
BEGINNING OF CONVERSATION: USER: {in} {suffix 0} ASSISTANT: {affirmation}
BEGINNING OF CONVERSATION: USER: {in} {suffix 1} ASSISTANT: {affirmation}
BEGINNING OF CONVERSATION: USER: {in} {suffix 2} ASSISTANT: {affirmation}
...
BEGINNING OF CONVERSATION: USER: {in} {suffix 2} ASSISTANT: {affirmation}

# test
BEGINNING OF CONVERSATION: USER: {in} {suffix t} ASSISTANT:
```

# Basics

PPO is the extension of the classical policy gradient algorithm (therefore, on-policy) by updating multiple steps rather than only one step. Suppose we have a reward model $r _ \theta(x, y)$, the PPO tries to update the LM parameters $\phi$ so that the **cumulative** reward is **maximized**

The following is the steps of PPO (taken from Hyunwon Chung's talk):

- Step 1: Obtaining a SFT model using the standard LM loss.
- Step 2: Repeat the following:
	- Sampling: Sampling prompts from the datasets.
	- Rollout: Generating responses with the current version of LM $\pi _ \phi ^ \mathrm{RL}$.
	- Evaluation: Using the (fixed) reward model $r _ \theta$ to score each of the response in the last step.
	- Optimization: Using the `(prompt, continuation, score)` triplets as a dataset to optimize the parameter (i.e., $\phi$) of the LM.

These steps are written concisely (yet confusingly) in the original paper as follows. The first boxed term is used to prevent overfitting to the reward function; the second boxed term is to reduce the performance regression on the standard benchmarks.
$$
\mathbb{E} _ {x, y \sim D _ {\pi ^ \mathrm{RL}}} \left[ r _ \theta(x, y) - \boxed{\beta \cdot \log \frac{\pi _ \phi ^ \mathrm{RL}(y\vert x)}{\pi^\mathrm{SFT}(y\vert x)}}\right] + \boxed{\gamma \mathbb{E} _ {x \sim D} \left[ \log \pi _ \phi ^ \mathrm{RL}(x) \right]}
$$

# Method
This paper is motivated by the observation that the aligned LM will still generate unsafe content if we could make the first few words of the LM response something like "Sure, here is how $UNSAFE_CONTENT".

Therefore, the idea is to disguise the input prompt with an automatically optimized suffix so that an aligned LM has a similar loss as 

Note that selecting replacement by loss makes sense because the RLHF maximizes the reward **while** staying as close as the original SFT model.
# Code Anatomy

The codebase is designed for chat models that involve different "roles" in the format of tuples. It is necessary to adapt the codebase to make it work with plain text models.

The most complicated part of the codebase is how the authors handle different prompt template in various language models; these messy details are all included in `llm_attacks.minimal_gcg.string_utils.SuffixManager`. What makes things more complicated is the fact that these string processing utilities in turn depend on `fastchat` library.

Three key variables in the demo specific to LLaMA2-Chat model are `manager._control_slice`, `manager._loss_slice`, and `manager._target_slice`. These three variables are derived from hidden variables `self._assistant_role_slice` and `self._user_role_slice`; they are **fixed** throughout the attack.


```python
d
```

The attack discussed in the paper works best with greedy decoding (the default approach in `model.generate()`). One may develop special decoding methods geared towards safety.