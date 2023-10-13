---
title: Talk Notes |  Training State-of-the-Art Text Embedding & Neural Search Models
tags: 
- DPR
categories:
- Talk
---

> [[YouTube](https://www.youtube.com/watch?v=XHY-3FzaLGc)] - [[Personal Website](https://www.nils-reimers.de/)]
>
> -   The presenter of this tutorial is Nils Remiers; he is the author of `sentence_transformers` and he is a researcher at HuggingFace.

-   Dense representations are interesting as they allow for zero-shot classification in the embedding space. This not only works for text embeddings, but multi-lingual and multi-modal as well.

![image-20231012114840698](https://raw.githubusercontent.com/guanqun-yang/remote-images/master/2023/10/upgit_20231012_1697125720.png)

-   Using out-of-the-box embeddings (for example, averaging BERT embeddings or using GPT-3 embeddings) does not work (see [1], [2]).

-   Vector Space

    The contrastive or triplet loss may only optimize the local structure. A good embedding model should both optimize global and local structures.

    -   Global Structure: Relation of two random sentences.
    -   Local Structure: Relation of two similar sentences.

# Reference

1.   [OpenAI GPT-3 Text Embeddings - Really a new state-of-the-art in dense text embeddings? | by Nils Reimers | Medium](https://medium.com/@nils_reimers/openai-gpt-3-text-embeddings-really-a-new-state-of-the-art-in-dense-text-embeddings-6571fe3ec9d9): This benchmarking was done in late December 2021, when the embedding endpoint was released not long.
2.   [MTEB Leaderboard - a Hugging Face Space by mteb](https://huggingface.co/spaces/mteb/leaderboard): As of 2023-10-12, the `text-embedding-ada-002` ranks 14 in the benchmark. All of the first 13 models that rank higher are open-source models.
