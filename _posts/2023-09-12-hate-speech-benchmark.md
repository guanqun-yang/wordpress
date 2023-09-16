---
title: Research Notes | A Benchmark for Hate Speech Detection
tags: 
- Hate Speech Detection
- Datasets
- Benchmark
categories:
- Research
---

# Existing Datasets

The current data aggregation includes [1] through [5], where the [5] only includes hate speech. 

1. [Detecting East Asian Prejudice on Social Media](https://aclanthology.org/2020.alw-1.19) (Vidgen et al., ALW 2020)
2. [[2005.12423] Racism is a Virus: Anti-Asian Hate and Counterspeech in Social Media during the COVID-19 Crisis](https://arxiv.org/abs/2005.12423) (He et al.)
3. [[2108.12521] TweetBLM: A Hate Speech Dataset and Analysis of Black Lives Matter-related Microblogs on Twitter](https://arxiv.org/abs/2108.12521) (Kumar et al.)
4. [Hate Towards the Political Opponent: A Twitter Corpus Study of the 2020 US Elections on the Basis of Offensive Speech and Stance Detection](https://aclanthology.org/2021.wassa-1.18) (Grimminger & Klinger, WASSA 2021)
5. [Latent Hatred: A Benchmark for Understanding Implicit Hate Speech](https://aclanthology.org/2021.emnlp-main.29) (ElSherief et al., EMNLP 2021)

# cardiffnlp/twitter-roberta-base-hate-latest Collection

The follow are the datasets used for the model [`cardiffnlp/twitter-roberta-base-hate-latest`](cardiffnlp/twitter-roberta-base-hate-latest) or the paper below:

> [Robust Hate Speech Detection in Social Media: A Cross-Dataset Empirical Evaluation](https://aclanthology.org/2023.woah-1.25) (Antypas & Camacho-Collados, WOAH 2023)

| Index | Dataset Name | Source                                                       | Notes         |
| ----- | ------------ | ------------------------------------------------------------ | ------------- |
| 1     | HatE         | [Link](http://hatespeech.di.unito.it/hateval.html) that requires filling in a Google form. |               |
| 2     | MHS          | [`ucberkeley-dlab/measuring-hate-speech`](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) |               |
| 3     | DEAP         | [Zenodo](https://zenodo.org/record/3816667)                  |               |
| 4     | CMS          | [Link](https://search.gesis.org/research_data/SDN-10.7802-2251?doi=10.7802/2251) that requires registration and email verification. |               |
| 5     | Offense      | [Link](https://sites.google.com/site/offensevalsharedtask/olid); this dataset is also called OLID. |               |
| 6     | HateX        | [`hatexplain`](https://huggingface.co/datasets/hatexplain) and [GitHub](https://github.com/hate-alert/HateXplain) |               |
| 7     | LSC          | [GitHub](https://github.com/ENCASEH2020/hatespeech-twitter.git) | Dehydrated    |
| 8     | MMHS         | [`nedjmaou/MLMA_hate_speech`](https://huggingface.co/datasets/nedjmaou/MLMA_hate_speech) and [GitHub](https://github.com/HKUST-KnowComp/MLMA_hate_speech) |               |
| 9     | HASOC        | [Link](https://hasocfire.github.io/hasoc/2020/dataset.html) that requires uploading a signed agreement; this agreement takes up to 15 days to approve. | Not Available |
| 10    | AYR          | [GitHub](https://github.com/zeeraktalat/hatespeech)          | Dehydrated    |
| 11    | AHSD         | [GitHub](https://github.com/t-davidson/hate-speech-and-offensive-language) |               |
| 12    | HTPO         | [Link](https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/stance-hof/) |               |
| 13    | HSHP         | [GitHub](https://github.com/zeeraktalat/hatespeech)          | Dehydrated    |

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

It is possible to approximate a subset of the original training mixture (8 of 12 datasets excluding the MMHS dataset, which only includes hate speech) following the Table 2 of the original paper. Something to note is that:

- AYR, HASOC, HSHP, and LSC are not usable.
- Offense does not exactly match the sizes in Table 2.
- We **disregard** any splits and try to match the number in Table 2. When matching number is not possible, we try to make sure the ratio of `on-hate` versus `hate` is same.



# Additional Datasets from hatespeechdata.com

The following the the additional datasets from [`hatespeechdata.com`](https://hatespeechdata.com/) that are not included in the above mentioned sources. The dataset names are either available from the original paper or created here for easy reference.

| Index | Dataset Name        | Source                                                       | Notes                                                        |
| ----- | ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1     | AbuseEval           | [GitHub](https://github.com/tommasoc80/AbuseEval/tree/master) | The Offense dataset above reannotated for non-hate, implicit, and explicit hate; only IDs are available. Around 87% of the hate/non-hate labels are same as the previous Offense dataset. |
| 2     | SWAD                | [GitHub](https://github.com/dadangewp/SWAD-Repository)       |                                                              |
| 3     | ALONE               |                                                              | Not usable. Requires contacting authors.                     |
| 4     | HatefulUsersTwitter | [GitHub](https://github.com/manoelhortaribeiro/HatefulUsersTwitter) and [Kaggle](https://www.kaggle.com/datasets/manoelribeiro/hateful-users-on-twitter) | Available but not relevant. This dataset is about detecting whether a user is hateful or neutral on the Tweet network; it does **not** come with annotated hateful/benign texts. |
| 5     | MMHS150K            | [Website](https://gombru.github.io/2019/10/09/MMHS/)         | Not usable. Multimodal datasets.                             |
| 6     | HarassmentLexicon   | [GitHub](https://github.com/Mrezvan94/Harassment-Corpus)     | Not usable. Lexicons only.                                   |
| 7     | P2PHate             | [GitHub](https://github.com/melsherief/hate_speech_icwsm18)  | Not usable. Dehydrated.                                      |
| 8     | Golbeck             |                                                              | Not usable. Requires contacting `jgolbeck@umd.edu`           |
| 9     | SurgeAI             | [Website](https://app.surgehq.ai/datasets/twitter-hate-speech) | Hateful content only.                                        |
| 10    | TSA                 | [Kaggle](https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech) | Dataset is provided by Analytics Vidhya. The `test.csv` does not come with labels. |

1. [I Feel Offended, Don’t Be Abusive! Implicit/Explicit Messages in Offensive and Abusive Language](https://aclanthology.org/2020.lrec-1.760) (Caselli et al., LREC 2020): The dataset from this paper is also called AbuseEval v1.0.
2. [Do You Really Want to Hurt Me? Predicting Abusive Swearing in Social Media](https://aclanthology.org/2020.lrec-1.765) (Pamungkas et al., LREC 2020)
3. [[2008.06465] ALONE: A Dataset for Toxic Behavior among Adolescents on Twitter](https://arxiv.org/abs/2008.06465) (Wijesiriwardene et. al.)
4. [[1803.08977] Characterizing and Detecting Hateful Users on Twitter](https://arxiv.org/abs/1803.08977) (Ribeiro et al., ICWSM 2018)
5. [[1910.03814] Exploring Hate Speech Detection in Multimodal Publications](https://arxiv.org/abs/1910.03814) (Gomez et al., WACV 2020)
6. [[1802.09416] A Quality Type-aware Annotated Corpus and Lexicon for Harassment Research](https://arxiv.org/abs/1802.09416) (Rezvan et al.)
7. [[1804.04649] Peer to Peer Hate: Hate Speech Instigators and Their Targets](https://arxiv.org/abs/1804.04649) (ElSherief et al.)
8. [A Large Labeled Corpus for Online Harassment Research](https://dl.acm.org/doi/10.1145/3091478.3091509) (Golbeck et al., WebSci 2017)
9. [Twitter Hate Speech Dataset](https://app.surgehq.ai/datasets/twitter-hate-speech) (Surge AI)
10. [Twitter Sentiment Analysis](https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech) (Kaggle)

