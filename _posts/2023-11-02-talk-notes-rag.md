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

# Overview

The motivation of RALM is the pessimism about the editing-based approaches for updating knowledge:

>   Moreover, these parametric LMs are **fundamentally incapable** of adapting over time, often hallucinate, and may leak private data from the training corpus. [...] Retrieval-based LMs can outperform LMs without retrieval by a large margin with **much fewer parameters**, can update their knowledge by replacing their retrieval corpora, and provide citations for users to easily verify and evaluate the predictions.

# Additional Notes

-   The backup video is downloaded based on the instruction documented [here](https://michaelabrahamsen.com/posts/how-to-download-zoom-recordings/). Basically, we just need to replace the `<cookie content>` and `<request url>` with the content we obtain after `F5` $\rightarrow$ `F12` in the browser.

    ```bash
    youtube-dl -o video.mp4 --referer "https://zoom.us/" --add-header "Cookie: COOKIE_CONTENT" 'REQUEST_URL'
    ```



