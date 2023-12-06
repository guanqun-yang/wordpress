---
title: Basics | Learning to Rank
tags: 
- LTR
categories:
- Basics
---
Table of Contents

1. [[#Overview|Overview]]
1. [[#Rank Aggregation|Rank Aggregation]]
1. [[#Reference|Reference]]


# Overview

This note is mostly based on three books below. When necessary, I provide additional references in the last section.

1. [Li, Hang. “Learning to Rank for Information Retrieval and Natural Language Processing, Second Edition.” *Learning to Rank for Information Retrieval and Natural Language Processing, Second Edition* (2014).](https://link.springer.com/book/10.1007/978-3-031-02155-8)
2. [Liu, Tie-Yan. “Learning to rank for information retrieval.” *Proceedings of the 33rd international ACM SIGIR conference on Research and development in information retrieval* (2009): n. pag.](https://link.springer.com/book/10.1007/978-3-642-14267-3)
3. [[2010.06467] Pretrained Transformers for Text Ranking: BERT and Beyond](https://arxiv.org/abs/2010.06467) (Lin et al.)

# Rank Aggregation

Suppose there are $M$ queries and $N$ documents, there will be a ranking list for each of $n$ queries. The goal is to aggregate these $n$ ranking lists into one ranking list.

The simplest rank aggregation method is called Borda count. The Borda count algorithm operates on the ranking lists by the following steps:

- Step 1: Aligning ranking list **of ranks** by document indexes.
- Step 2: Using the total document number $N$ to subtract each entry in the aligned ranking list.
- Step 3: Summing up the transformed ranking lists and generating a ranking based on this summed ranking list.

For example, the lists `A, B, C`, `A, C, B` and `B, A, C`:

- Step 1: After alignment by index `A`, `B`, and `C`, the ranking lists **of ranks** become `1, 2, 3`, `1, 3, 2`, and `2, 1, 3`.
- Step 2: Using $N=3$ to subtract each entry gives us `2, 1, 0`, `2, 0, 1`, and `1, 2, 0`.
- Step 3: The summed ranking list of ranks is `5, 3, 1`. Therefore, the initial 3 ranking lists is converted to one single ranking list: `A, B, C`.

This could be easily implemented in Python as following: 

```python
from collections import defaultdict

def borda_count(votes):
    N = len(votes[0])
    score_dict = defaultdict(int)

    for vote in votes:
        for rank, candidate in enumerate(vote):
            score_dict[candidate] += N - rank

    aggregated_ranks = sorted(score_dict.keys(), key=score_dict.get, reverse=True)
    return aggregated_ranks


votes = [["A", "B", "C"], ["A", "C", "B"], ["B", "A", "C"]]
print(borda_count(votes))
```



# Reference

