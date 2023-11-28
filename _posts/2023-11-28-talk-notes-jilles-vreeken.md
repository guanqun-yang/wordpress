---
title: Talk Notes | Causality
tags:
- Causality
categories:
- Talk
---

[toc]

# Overview

Causality Ladder (Judea Pearl): Seeing $\rightarrow$ Intervening $\rightarrow$ Imagining
-   Seeing: This is where the traditional ML happens.
-   Invervening
-   Imaging: This requires structural causal model (SCM). This is not discussed in the talk.

# Assumptions

-   Ingredients

    Besides, we need to assume (1) we have **magically** measured all factors; there are no confounders, and (2) iid.

    -   Data: Assumes to be faithful to the graph.
    -   Causal Graph: Assumes to satisfy Markov condition.

# Identifying Causality

-   Intuition (Janzing 2012)

    If $X$ causes $Y$, then the noise pattern from $X$ is $Y$ is simpler than the other way around.

-   Operationalizing the Intuition

    -   Kolmogorov Complexity: The shortest program (in any programming language) that computes a PDF. Then if $X \rightarrow Y$, then $K(P(X)) + K(P(Y\vert X)) \leq K(P(Y)) + K(P(X\vert Y))$.
    -   The formula above could be realized in practice with some assumptions in systems called SLOOPY, HECI (Xu et al. 2022 and Marx & Vreeken 2017, 2019) based on relatively simple regressions.

-   These systems could be evaluated using radar plot of some established datasets.









