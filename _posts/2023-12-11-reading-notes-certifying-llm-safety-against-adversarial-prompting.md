---
title: Reading Notes | Certifying LLM Safety against Adversarial Prompting
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
> - 2023-12-07: First draft. The corresponding authors include Soheil Feizi (UMD) and Himabindu Lakkaraju (Harvard).

- The intuition of the proposed certified LLM safety is quite simple: the complete sequence is safe if all of its subsequences are also safe. 

  However, one issue with this notion of safety is that it relies on the capability of the safety classifier: if the classifier systematically fail, then the certificate is broken.




