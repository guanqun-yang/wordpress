---
title: Talk Notes |  ACL 2023 Tutorial - Retrieval-based Language Models and Applications
tags: 
- RAG
categories:
- Talk
---

>   [[Zoom Recording](https://us06web.zoom.us/rec/play/6fqU9YDLoFtWqpk8w8I7oFrszHKW6JkbPVGgHsdPBxa69ecgCxbmfP33asLU3DJ74q5BXqDGR2ycOTFk.93teqylfi_uiViNK?canPlayFromShare=true&from=share_recording_detail&continueMode=true&componentName=rec-play&originRequestUrl=https%3A%2F%2Fus06web.zoom.us%2Frec%2Fshare%2FNrYheXPtE5zOlbogmdBg653RIu7RBO1uAsYH2CZt_hacD1jOHksRahGlERHc_Ybs.KGX1cRVtJBQtJf0o)] - [[Website and Slides](https://acl2023-retrieval-lm.github.io/)] - [[Proposal](https://aclanthology.org/2023.acl-tutorials.6.pdf)] - [[Q&A](https://app.sli.do/event/ok8R2jMMvNjp9uMkxi63Qi/live/questions)] - [[Backup Recording](https://youtu.be/7_0R5JMIogM)]
>
>   - This tutorial is given by Akari Asai, Sewon Min, Zexuan Zhong, and Danqi Chen
>   - Timestamp: 1:14:44

# Overview

RALM is an LM that uses external databases **in the test time**.

The motivation of RALM is the pessimism about the current editing-based approaches. If we are able to edit the LLMs as fast as CRUD operations on databases, the retrieval-based approaches and editing-based approaches are comparable.

>   Moreover, these parametric LMs are **fundamentally incapable** of adapting over time, often hallucinate, and may leak private data from the training corpus. [...] Retrieval-based LMs can outperform LMs without retrieval by a large margin with **much fewer parameters**, can update their knowledge by replacing their retrieval corpora, and provide citations for users to easily verify and evaluate the predictions.

Besides ease of updating knowledge, the retrieval-based approaches also have following advantages:

-   Traceability, Verifiability, Interpretability, and Controllability: They mean the same thing for RALMs.
-   Privacy and Copyright: The LMs are only responsible for making inference. The more relevant documents are stored in the databases. 

# Architecture

Three elements of RALMs:

-   *What* to Retrieve? What is the minimal unit for retrieval. The choices could be token, document, or text chunk.
-   *How* to Use the Retrieval Results? Input, output, or somewhere in between?
-   *When* to Retrieve? Only once, every token, or somewhere in between?

![image-20231102223928224](https://raw.githubusercontent.com/guanqun-yang/remote-images/master/2023/11/upgit_20231102_1698979168.png)

-   REALM is one of the first works in RALM. The goal is improving MLM LM pretraining. The followup papers include DPR, RAG, Atlas; they all focus on knowledge intensive tasks:

    -   DPR
    -   RAG
    -   Atlas

-   Retrieval-in-context LM

    -   REPLUG

    -   Ram et al.: We do not have to use the entire prompt as query. It may be better to use more recent tokens (due to higher relevance to the tokens to generate) as long as they are not too short.

        Further, we may want to retrieve more often. For example, after we have already generated some tokens, it is time to retrieve again for the next batch of new tokens.

-   RETRO

    -   Used in the intermediate layer. Specifically, each prompt is chunked into multiple pieces and sent to retrieve some results; these results are sent into the LM through a specially design attention mechanism. The authors also consider some parallelization techniques for the sake of efficiency.
    -   Another orthogonal finding of this paper is that the scale of the datastores are important.

-   kNN-LM

    -   Using the test prefix to query for the prefixes that are already continued with new tokens. The token to generate will be an linear interpolation between the actual generated new tokens and the new tokens that belong to the best match prefix.

    -   The motivation of this design is that the text representations of the **entire** sentences could be quite different even though the same word appears in it. For example, the word "torch" and "cheap."

        >   Comment: The motivation of this design is dubious. Furthermore, the interaction between the input and the retrieved results is limited.

-   Other Extensions
    -   Adaptive Retrieval. Whether the retrieval is enabled depends on the confidence of the outputs. For example, 
        -   FLARE and He et al. More specifically, the $\lambda$ in kNN-LM could be a function of confidence.
        -   Alon et al.





| Paper                                    | What   | When             | How          | Note |
| ---------------------------------------- | ------ | ---------------- | ------------ | ---- |
| REALM                                    | Chunks | Once             | Input        |      |
| Retrieve-In-Context LM                   | Chunks | Every $n$ Tokens | Input        |      |
| RETRO                                    | Chunks | Every $n$ Tokens | Intermediate |      |
| kNN-LM                                   | Tokens | Every Token      | Output       |      |
| FLARE                                    | Chunks | Adaptive         | Input        |      |
| Adaptive kNN-LM (He et al., Alon et al.) | Tokens | Adaptive         | Output       |      |
|                                          |        |                  |              |      |
|                                          |        |                  |              |      |
|                                          |        |                  |              |      |







# Additional Notes

-   The backup video is downloaded based on the instruction documented [here](https://michaelabrahamsen.com/posts/how-to-download-zoom-recordings/). Basically, we just need to replace the `<cookie content>` and `<request url>` with the content we obtain after `F5` $\rightarrow$ `F12` in the browser.

```bash
youtube-dl -o video.mp4 --referer "https://zoom.us/" --add-header "Cookie: COOKIE_CONTENT" 'REQUEST_URL'
```



# Reference

1.   [Building Scalable, Explainable, and Adaptive NLP Models with Retrieval | SAIL Blog](https://ai.stanford.edu/blog/retrieval-based-NLP/)
