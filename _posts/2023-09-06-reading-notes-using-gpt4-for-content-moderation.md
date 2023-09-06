---
title: Reading Notes | Using GPT4 for Content Moderation
tags: 
- LLM
- Hate Speech Detection
- Customized LLM
categories:
- Reading
---

# Method

This blog post illustrates an idea of human-AI collaboration in revising an existing content policy. Specifically,

- Based on an initial policy $P_0$, a human expert may disagree with a moderation decision of GPT-4. 
- The human expert elicit suggestions from GPT-4 to revise the policy $P_0$ into $P_1$ until the human expert agrees with the decision from GPT-4.

The blog post does not clearly explain how either step is done. For example, (1) what prompt is used to turn the general purpose GPT-4 into a content moderator, (2) what prompt is used to ask the feedback from GPT-4, and (3) how human experts ingest GPT-4 feedback into concrete policy revisions.