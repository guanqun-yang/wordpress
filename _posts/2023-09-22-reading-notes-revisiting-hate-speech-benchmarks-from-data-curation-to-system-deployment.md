---
title: Reading Notes | Revisiting Hate Speech Benchmarks - From Data Curation to System Deployment
tags: 
- Hate Speech Detection
- MoE
categories:
- Reading
---

> [[Semantic Scholar](https://www.semanticscholar.org/paper/Revisiting-Hate-Speech-Benchmarks%3A-From-Data-to-Kulkarni-Masud/faf46bd0d88c582177d9bf9d0b3c3e30fd5a763e)] - [Code] - [Tweet] - [Video] - [Website] - [Slide]
>
> Change Logs:
>
> - 2023-09-21: First draft. This paper appears at KDD 2023. The co-lead author - [Sarah Musud](https://sara-02.github.io/) - has published numerous papers on hate speech detection.

# Additional Notes

- Measuring Dataset Difficulty

    The authors compare different datasets' difficulty using the JS divergence between Laplician smoothed unigram distributions of texts under different label pairs; the lower the divergence, the closer the unigram distributions and this makes texts under a label pair more difficult to distinguish. 

    For example, the proposed datasets have 4 labels, this will lead to $\binom{4}{2} = 6$ divergence measures.

- Matthews Correlation Coefficient (MCC)

    

# Reference

