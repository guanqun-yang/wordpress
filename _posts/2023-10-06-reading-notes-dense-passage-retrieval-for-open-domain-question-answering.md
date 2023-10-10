---
title: Reading Notes | Dense Passage Retrieval for Open-Domain Question Answering
tags: 
- DPR
categories:
- Reading
---

> [[Semantic Scholar](https://www.semanticscholar.org/paper/Dense-Passage-Retrieval-for-Open-Domain-Question-Karpukhin-O%C4%9Fuz/79cd9f77e5258f62c0e15d11534aea6393ef73fe)] - [Code] - [Tweet] - [[Video](https://slideslive.com/38939151/dense-passage-retrieval-for-opendomain-question-answering)] - [Website] - [Slide]
>
> Change Logs:
>
> - 2023-10-05: First draft. This paper appears at EMNLP 2020.

# Overview

- Dense Passage Retrieval is a familiar thing proposed in this paper. The issue is that previous solutions underperform BM25. The contribution of this paper is discovering an engineering feasible solution that learns a DPR model effectively without many examples; it improves upon the BM25 by a large margin.

# Method

The training goal of DPR is to learn a metric where the distance between the query $q$ and relevant documents $p^+$ smaller than that of irrelevant documents $p^-$ in the high-dimensional space. That is, we want to **minimize** the loss below:
$$
L(q _ i, p _ i ^ +, p _ {i1} ^ -, \cdots, p _ {in}^-) := -\log \frac{ \exp(q _ i^T p _ i^+)}{\exp(q_i^T p _ i ^  +) + \sum _ {j=1}^n \exp(q _ i ^ T p _ {ij}^-)}
$$
The authors find that using the "in-batch negatives" is a simple and effective negative sampling strategy (see "Gold" with and without "IB"). Specifically, within a batch of $B$ examples, any answer that is not associated with the current query is considered a negative. If **one answer** (see the bottom block) retrieved from BM25 is added as a hard negative, the performance will improve more.

![image-20231006000405936](https://raw.githubusercontent.com/guanqun-yang/remote-images/master/2023/10/upgit_20231006_1696565045.png)

The retrieval model has been trained for 40 epochs for larger datasets ("NQ", "TriviaQA", "SQuAD") and 100 epochs for smaller ones ("WQ", "TREC") with a learning rate `1e-5`. Note that the datasets the authors use to fine-tune the models are large. For example, [`natural_questions`](https://huggingface.co/datasets/natural_questions/viewer/default/train) is 143 GB.

![image-20231009181325527](https://raw.githubusercontent.com/guanqun-yang/remote-images/master/2023/10/upgit_20231009_1696889605.png)

# Additional Notes

-   The dual-encoder + cross-encoder design is a classic; they are not necessarily end-to-end differentiable. For example, in this work, after fine-tuning the dual-encoder for retrieval, the authors separately fine-tuned a QA model. This could be a favorable design due to better performance:

    >   This approach obtains a score of 39.8 EM, which suggests that our strategy of training a strong retriever and reader in isolation can leverage effectively available supervision, while outperforming a comparable joint training approach with a simpler design.

-   The inner product of unit vectors is indeed the cosine similarity.

# Code

- HuggingFace provides [classes](https://huggingface.co/docs/transformers/v4.34.0/model_doc/dpr) for DPR. The Retrieval Augmented Generation (RAG) is one [example](https://github.com/huggingface/transformers/tree/main/examples/research_projects/rag) that fine-tunes using DPR to improve knowledge-intense text generation.

- `simpletransformers` provides easy-to-use [interfaces](https://simpletransformers.ai/docs/retrieval-model/) to train DPR models; it even provides a routine to select hard negatives. The following is a minimal working example:

    ```python
    import os
    import logging
    
    os.environ["WANDB_DISABLED"] = "false"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from simpletransformers.retrieval import (
        RetrievalModel,
        RetrievalArgs,
    )
    
    from datasets import (
        Dataset,
        DatasetDict,
    )
    
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    
    # trec_train.pkl and trec_dev.pkl are prepared from the original repository
    # see: https://github.com/facebookresearch/DPR/blob/main/README.md
    df = pd.read_pickle("../datasets/trec_train.pkl")
    train_df, eval_df = train_test_split(df, test_size=0.2)
    test_df = pd.read_pickle("../datasets/trec_dev.pkl")
    
    columns = ["query_text", "gold_passage", "title"]
    
    train_data = train_df[columns]
    eval_data = eval_df[columns]
    test_data = test_df[columns]
    
    # Configure the model
    model_args = RetrievalArgs()
    
    model_args.num_train_epochs = 40
    model_args.include_title = False
    
    # see full list of configurations:
    # https://simpletransformers.ai/docs/usage/#configuring-a-simple-transformers-model
    # critical settings
    model_args.learning_rate = 1e-5
    model_args.num_train_epochs = 40
    model_args.train_batch_size = 32
    model_args.eval_batch_size = 32
    model_args.gradient_accumulation_steps = 1
    model_args.fp16 = False
    model_args.max_seq_length = 128
    model_args.n_gpu = 1
    model_args.use_multiprocessing = False
    model_args.use_multiprocessing_for_evaluation = False
    
    # saving settings
    model_args.no_save = False
    model_args.overwrite_output_dir = True
    model_args.output_dir = "outputs/"
    model_args.best_model_dir = "{}/best_model".format(model_args.output_dir)
    model_args.save_model_every_epoch = False
    model_args.save_best_model = True
    model_args.save_steps = 2000
    
    # evaluation settings
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = 100
    
    # logging settings
    model_args.silent = False
    model_args.logging_steps = 50
    model_args.wandb_project = "HateGLUE"
    model_args.wandb_kwargs = {
        "name": "DPR"
    }
    
    model_type = "dpr"
    context_encoder_name = "facebook/dpr-ctx_encoder-single-nq-base"
    question_encoder_name = "facebook/dpr-question_encoder-single-nq-base"
    
    model = RetrievalModel(
        model_type=model_type,
        context_encoder_name=context_encoder_name,
        query_encoder_name=question_encoder_name,
        use_cuda=True,
        cuda_device=0,
        args=model_args
    )
    
    # Train the model
    model.train_model(train_data, eval_data=eval_data)
    result = model.eval_model(eval_data)
    ```

    
