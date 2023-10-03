---
title: Reading Notes | Wild-Time - A Benchmark of in-the-Wild Distribution Shift over Time
tags:
- Benchmark
- Generalization
categories:
- Reading
---

> [[Semantic Scholar](https://www.semanticscholar.org/paper/Wild-Time%3A-A-Benchmark-of-in-the-Wild-Distribution-Yao-Choi/629677e4ff2aceb951ccfdf3763956c75974c8e5)] - [[Code](https://github.com/huaxiuyao/Wild-Time)] - [Tweet] - [Video] - [[Website and Leaderboard](https://wild-time.github.io/)] - [Slide] - [[Lead Author](https://www.huaxiuyao.io/)]
>
> Change Logs:
>
> - 2023-10-03: First draft. The authors provide 5 datasets (2 of them are text classification datasets, the others include 2 image classification datasets and 1 EHR dataset) and more than 10 mitigation methods for distribution shift.

# Experiments

- The authors find that most of the mitigation methods are **not** effective compared to the standard ERM on the proposed benchmark. Note that SimCLR and SwaV methods are only applicable to image classification tasks.

    ![image-20231003120134331](https://raw.githubusercontent.com/guanqun-yang/remote-images/master/2023/10/upgit_20231003_1696348894.png)

![image-20231003120318062](https://raw.githubusercontent.com/guanqun-yang/remote-images/master/2023/10/upgit_20231003_1696348998.png)



# Additional Notes

From the content below, we could see that:

> To address this challenge, we adapt the above invariant learning approaches to the temporal distribution shift setting. We leverage timestamp metadata to create a temporal robustness set consisting of substreams of data, where each substream is treated as one domain. Specifically, as shown in Figure 3, we define a sliding window G with length L. For a data stream with T timestamps, we apply the sliding window G to obtain $T − L + 1$ substreams. We treat each substream as a “domain" and apply the above invariant algorithms on the robustness set. We name the adapted CORAL, GroupDRO and IRM as CORAL-T, GroupDRO-T, IRM-T, respectively. Note that we do not adapt LISA since the intra-label LISA performs well without domain information, which is also mentioned in the original paper.

- The way the authors apply the group algorithms look questionable: it does not make sense to create artificial domains by grouping data from some consecutive timestamps. This may be the reason why the authors do not observe the performance gains.
- The [LISA](https://arxiv.org/abs/2201.00299), which is the same author's work, seems to be a good approach as it does not require the domain labels while performing competitively.

