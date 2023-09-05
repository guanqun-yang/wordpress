---
title: Reading Notes | Robust Hate Speech Detection in Social Media - A Cross-Dataset Empirical Evaluation
tags: 
- Hate Speech Detection
- Data Selection
categories:
- Reading
---

> [[Semantic Scholar](https://www.semanticscholar.org/paper/Robust-Hate-Speech-Detection-in-Social-Media%3A-A-Antypas-Camacho-Collados/92ebe45b5422c7b78fdab7520c7f2bce3e713733)] - [Code] - [Tweet] - [Video] - [Website] - [Slide] - [[HuggingFace](https://huggingface.co/cardiffnlp/twitter-roberta-base-hate-latest)]
>
> Change Logs:
>
> - 2023-09-04: First draft. This paper appears at WOAH '23. The provided models on HuggingFace have more than 40K downloads thanks to their easy-to-use `tweetnlp` package; the best-performing binary and multi-class classification models are `cardiffnlp/twitter-roberta-base-hate-latest` and `cardiffnlp/twitter-roberta-base-hate-multiclass-latest` respectively.

## Method

### Datasets

The authors **manually** select and unify 13 hate speech datasets for binary and multi-class classification settings. The authors do not provide the rationale on why they choose these 13 datasets.

For the multi-class classification setting, the authors devise 7 classes: racism, sexism, disability, sexual orientation, religion, other, and non-hate. This category is similar to yet smaller than the MHS dataset, including gender, race, sexuality, religion, origin, politics, age, and disability (see [1]).

For all 13 datasets, the authors apply a 7:1:2 ratio of data splitting; they also create a small external test set (i.e., `Indep`). With test sets kept untouched, the authors consider 3 ways of preparing data:

1. Training on the single dataset.
2. Training on an aggregation of 13 datasets.
3. Training on a sampled dataset from the aggregation in 2. Specifically, the authors (1) find the dataset size that leads to the highest score in 1, (2) sample the dataset proportionally by the each of 13 datasets' sizes and the the ratio of hate versus non-hate to exactly 1:1.

The processed datasets are not provided by the authors. We need to follow the guides below to obtain them; the index of the datasets is kept consistent with the [HuggingFace model hub](https://huggingface.co/cardiffnlp/twitter-roberta-base-hate-latest) and their names follow the main paper.

| Index | Dataset Name | Source                                                       | Notes      |
| ----- | ------------ | ------------------------------------------------------------ | ---------- |
| 1     | HatE         | [Link](http://hatespeech.di.unito.it/hateval.html) that requires filling in a Google form. |            |
| 2     | MHS          | [`ucberkeley-dlab/measuring-hate-speech`](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) |            |
| 3     | DEAP         | [Zenodo](https://zenodo.org/record/3816667)                  |            |
| 4     | CMS          | [Link](https://search.gesis.org/research_data/SDN-10.7802-2251?doi=10.7802/2251) that requires registration and email verification. |            |
| 5     | Offense      | [Link](https://sites.google.com/site/offensevalsharedtask/olid); this dataset is also called OLID. |            |
| 6     | HateX        | [`hatexplain`](https://huggingface.co/datasets/hatexplain) and [GitHub](https://github.com/hate-alert/HateXplain) |            |
| 7     | LSC          | [GitHub](https://github.com/ENCASEH2020/hatespeech-twitter.git) | Dehydrated |
| 8     | MMHS         | [`nedjmaou/MLMA_hate_speech`](https://huggingface.co/datasets/nedjmaou/MLMA_hate_speech) and [GitHub](https://github.com/HKUST-KnowComp/MLMA_hate_speech) |            |
| 9     | HASOC        | [Link](https://hasocfire.github.io/hasoc/2020/dataset.html) that requires uploading a signed agreement; this agreement takes up to 15 days to approve. |            |
| 10    | AYR          | [GitHub](https://github.com/zeeraktalat/hatespeech)          | Dehydrated |
| 11    | AHSD         | [GitHub](https://github.com/t-davidson/hate-speech-and-offensive-language) |            |
| 12    | HTPO         | [Link](https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/stance-hof/) |            |
| 13    | HSHP         | [GitHub](https://github.com/zeeraktalat/hatespeech)          | Dehydrated |

The following are the papers that correspond to the list of datasets:

1. [SemEval-2019 Task 5: Multilingual Detection of Hate Speech Against Immigrants and Women in Twitter](https://aclanthology.org/S19-2007) (Basile et al., SemEval 2019)
2. [The Measuring Hate Speech Corpus: Leveraging Rasch Measurement Theory for Data Perspectivism](https://aclanthology.org/2022.nlperspectives-1.11) (Sachdeva et al., NLPerspectives 2022)
3. [Detecting East Asian Prejudice on Social Media](https://aclanthology.org/2020.alw-1.19) (Vidgen et al., ALW 2020)
4. [[2004.12764] "Call me sexist, but...": Revisiting Sexism Detection Using Psychological Scales and Adversarial Samples](https://arxiv.org/abs/2004.12764) (Samory et al.)
5. [Predicting the Type and Target of Offensive Posts in Social Media](https://aclanthology.org/N19-1144) (Zampieri et al., NAACL 2019)
6. [[2012.10289] HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection](https://arxiv.org/abs/2012.10289) (Mathew et al.)
7. [[1802.00393] Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior](https://arxiv.org/abs/1802.00393) (Founta et al.)
8. [Multilingual and Multi-Aspect Hate Speech Analysis](https://aclanthology.org/D19-1474) (Ousidhoum et al., EMNLP-IJCNLP 2019)
9. [[2108.05927] Overview of the HASOC track at FIRE 2020: Hate Speech and Offensive Content Identification in Indo-European Languages](https://arxiv.org/abs/2108.05927) (Mandal et al.)
10. [Are You a Racist or Am I Seeing Things? Annotator Influence on Hate Speech Detection on Twitter](https://aclanthology.org/W16-5618) (Waseem, NLP+CSS 2016)
11. [[1703.04009] Automated Hate Speech Detection and the Problem of Offensive Language](https://arxiv.org/abs/1703.04009) (Davidson et al.)
12. [Hate Towards the Political Opponent: A Twitter Corpus Study of the 2020 US Elections on the Basis of Offensive Speech and Stance Detection](https://aclanthology.org/2021.wassa-1.18) (Grimminger & Klinger, WASSA 2021)
13. [Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitter](https://aclanthology.org/N16-2013) (Waseem & Hovy, NAACL 2016)

### Models and Fine-Tuning

The authors start from the `bert-base-uncased`, `roberta-base`, and two models specifically customized to Twitter (see [2], [3]). The authors carry out the HPO on learning rates, warmup rates, number of epochs, and batch size using `hyperopt`.

## Experiments

- The data preparation method 3 (`All*`) performs better than the method 1 (`MHS`, `AYR`, etc). It also achieves the highest scores on the `Indep` test set (Table 3).

    ![image-20230904122703515](https://raw.githubusercontent.com/guanqun-yang/remote-images/master/2023/09/upgit_20230904_1693844823.png)

## Other Information

- Language classification tasks could be done with `fasttext` models ([doc](https://fasttext.cc/docs/en/language-identification.html)).

## Comments

- Ill-Defined Data Collection Goal

    We can read the sentence like following from the paper:

    > - For example both CMS and AYR datasets deal with sexism but the models trained only on CMS perform poorly when evaluated on AYR (e.g. BERTweetCSM achieves 87% F1 on CSM, but only 52% on AYR).
    > - This may be due to the scope of the dataset, dealing with East Asian Prejudice during the COVID-19 pandemic, which is probably not well captured in the rest of the datasets. 

    The issue is that there is not quantitative measure of the underlying theme of a dataset (for example, CMS and AYR). The dataset curators may have some general ideas on what the dataset should be about; they often do not have a clearly defined measure to quantify how much one sample aligns with their data collection goals.

    I wish to see some quantitative measures on topics  and distributions of an NLP dataset.

## Reference

1. [Targeted Identity Group Prediction in Hate Speech Corpora](https://aclanthology.org/2022.woah-1.22) (Sachdeva et al., WOAH 2022)
2. [BERTweet: A pre-trained language model for English Tweets](https://aclanthology.org/2020.emnlp-demos.2) (Nguyen et al., EMNLP 2020)
3. [TimeLMs: Diachronic Language Models from Twitter](https://aclanthology.org/2022.acl-demo.25) (Loureiro et al., ACL 2022): This paper also comes from Cardiff NLP. It considers the time axis of the language modeling through continual learning. It tries to achieve OOD generalization (in terms of time) without degrading the performance on the static benchmark.
