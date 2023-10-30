---
title: Coding Notes | HuggingFace Reference
tags: 
categories:
- Coding
---

[toc]

# Basics

## Hyperparameters

- The hyperparameters are specified through [`TrainingArguments`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) and [`Seq2SeqTrainingArguments`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments).
- `model_name_or_path` and `output_dir` are the only two required arguments. However, we should also set other critical hyperparameters, including `num_train_epochs`, `per_device_train_batch_size`, `per_device_eval_batch_size`, `learning_rate`.

## Evaluation, Logging, and Saving

- It is better to set `logging_steps` to `1` and `logging_strategy` to `step` as logging is beneficial whatsoever yet does not cause significant overhead.
- It is better to specify`eval_steps` as  `1 / n`  and `eval_strategy` to `"steps"`, where `n` is number evaluations. This will help collect enough samples even if we have fewer training steps or training epochs.
- `load_best_model_at_end=True` has to pair with the following configurations ([answer](https://discuss.huggingface.co/t/save-only-best-model-in-trainer/8442/5)). It will save the best checkpoints according to the evaluations done throughout the training process:
    - After setting `eval_steps` to a decimal number, `save_strategy` has to be set to `"steps"` since `save_steps` has to be multiple of `eval_steps`. As saving larger models will take long time, we need to set `save_steps` to a reasonable number. For example, if we would like to evaluate the model for 10 times (i.e., `eval_steps` is set to 0.1), we should save twice (i.e., `save_steps` is set to 0.5).

    - `save_total_limit` governs the saving of the latest models; it is likely to save `k+1` checkpoints even if `save_total_limit=k` as the best model is not the latest `k` models saved.


| Index | Hyperparameter                   | Value                                      |
| ----- | -------------------------------- | ------------------------------------------ |
| 1     | `save_strategy`, `eval_strategy` | `steps` or `epoch`; they have to be same.  |
| 2     | `eval_steps`                     | A reasonable value such as `0.1`.          |
| 3     | `save_steps`                     | Must be the multiples of the `eval_steps`. |

- It is recommended to use `wandb`. In order to do so, we need to set `report_to` and `run_name`. Note that if we need to use custom name on `wandb` portal, we should **not** rename the default output directory.

## Testing Training Scripts

| Index | Hyperparameter                                              | Value   | Notes |
| ----- | ----------------------------------------------------------- | ------- | ----- |
| 1     | `max_train_samples`, `max_eval_samples`, `max_test_samples` | 100     |       |
| 2     | `save_strategy`                                             | `no`    |       |
| 3     | `load_best_model_at_end`                                    | `False` |       |

## Checkpoints

If a model has been fine-tuned, then most likely there will be only updates in `pytorch_model.bin` file. We could reuse the original `config.json` and the tokenizer.

- A runnable model only consists of a `pytorch_model.bin` and a `config.json` file. The `config.json` documents the metadata of the model.

- A tokenizer consists of a list of files:

    ```bash
    tokenizer/
    ├── added_tokens.json
    ├── merges.txt
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── vocab.json
    ```

However, if we save checkpoints during training, then the code of saving checkpoints has already been taken care of.

# Inference

It does not seem to easily make inferences on multiple devices. However, we could use optimized attention implemented `torch>=2.0.0` and `optimum` to reduce the time and space requirement.

# Instruction Tuning

## Using the Basic transformers Library

It is possible to instruction-tune a language model using the [official example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling) `run_clm.py` working with `gpt2` or the [Phil Schimid's blog](https://www.philschmid.de/fine-tune-flan-t5-deepspeed) working with `google/flan-t5-xl`. 

## Using the trl Library

 `SFTTrainer()` provided in the `trl` provides an another layer of abstraction; this makes instruction-tuning even easier and cleaner. However, the downsides are (1) it could not work well with `deepspeed`, and (2) it does not support everything (for example, setting `save_steps` to a decimal number) defined in `transformers.TraingingArguments`; this limits its flexibility.

- Tuning a Model with the Language Modeling Objective

    This could be done in fewer than 14 lines of code. For example, tuning an LM on the `imdb` dataset. We could add more configurations to the code skeleton below (for example, PEFT, 4-bit / 8-bit) following the example script [here](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_trainer.py).

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from trl import SFTTrainer

dataset = load_dataset("imdb", split="train")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
)
trainer.train()
```

- Tuning a Model using the Completions - Self-Instruction

    

# Tuning Large Models with Constrained Hardware

## Overview

We have the following decision matrix when we are working on the single node; there may be other considerations when working with multiple nodes; a node means a machine’s GPUs are physically connected.

|  | Single GPU | Multiple GPUs |
| --- | --- | --- |
| Mode Fits into Single GPU | -  | DDP <br>ZeRO (may or may not be faster) |
| Model Does not Fit into Single GPU | ZeRO + Offload CPU + MCT (optional) + NVMe (optional) | PP (preferred) if NVLink or NVSwitch is not available <br>ZeRO <br>TP |
| Largest Layer Does not Fit into Single GPU | ZeRO + Offload CPU + MCT + NVMe (optional)   |TP <br>ZeRO + Offload CPU + MCT + NVMe (optional) |

- One single 7B LLaMA model is already almost 30 GB on HuggingFace; the 13B version will be even larger.

- When using custom training loops. The `accelerate` library improves `pytorch.distributed` and makes it possible that the same code could be run on any hardware settings without making updates to the code.

    When using `Trainer()`, all of the distributed training settings could be done without using `accelerate`.

- ZeRO is implemented using `deepspeed`.

## Using a Single GPU

- A typical model with AdamW optimizer requires 18 bytes per parameter.
- Besides the methods described below, one could try `accelerate` library to use same `torch` code for any hardware configuration (CPU, single GPU, and multiple GPUs).

- Besides the methods described below, one could try `accelerate` library to use same `torch` code for any hardware configuration (CPU, single GPU, and multiple GPUs).

| Method | Speed📈 | Memory📉 | Note |
| --- | --- | --- | --- |
| Batch Size | Yes | Yes | It should be default to 8. But choosing a batch size that makes most of GPUs is complicated. |
| Dataloader | Yes | No | Always set `pin_memory=True` and` num_workers=4` (or 8, 16, …) when possible. |
| Optimizer | Yes | Yes | Using Adafactor saves 50% compared to Adam or AdamW. But it does not converge fast. This is supported out-of-box.<br>One could alternatively use 8-bit AdamW to save more than 50% memory when bibsandbytes is installed and used. |
| Gradient Checkpointing | No | Yes | Supported by `Trainer(..., gradient_checkpointing=True, ...)`. |
| Gradient Accumulation | No | Yes | Supported by `Trainer(..., gradient_accumulation_steps=4,...)`. |
| Mixed Precision Training | Yes | No | `fp16` is supported in `TrainingArguments(.., fp16=True, ...)`. <br>With Ampere GPUs such as A100 or RTX-3090, `bf16=True` or `tf32=True` (with `torch.beakends.cuda.mamul.allow_tf32=True`) could be set. |
| DeepSpeed ZeRO | No | Yes | The model with the smallest batch size does not fit into the GPU. Using Trainer() is supported out-of-box. |

We could use the code below to measure the GPU utilization:

```python
from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
```

For example, when tuning a `bert-large-uncased` model with some dummy data, we could see on top of the following vanilla code:

- Vanilla Code to Tune a Classification Model

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

import numpy as np

from datasets import Dataset
from transformers import (
    Trainer,
    logging,
    TrainingArguments,
    AutoModelForSequenceClassification,
)

from utils.common import print_gpu_utilization

##################################################
logging.set_verbosity_error()

dataset_size, seq_len = 512, 512
train_dataset = Dataset.from_dict(
    {
        "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
        "labels": np.random.randint(0, 1, dataset_size),
    }
)
train_dataset.set_format("pt")
print_gpu_utilization()

##################################################

default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none"
}

training_args = TrainingArguments(
per_device_train_batch_size=4,
    **default_args
)

model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased").to("cuda")
print_gpu_utilization()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)
result = trainer.train()
```

After making updates to the vanilla code, we could see the changes in the memory usage.

| Basic Setup                                                  | Memory (MB) |
| ------------------------------------------------------------ | ----------- |
| Loading Dummy Data                                           | 2631        |
| Loading Model with per_device_train_batch_size=4             | 14949       |
| Loading Model with per_device_train_batch_size=4 + 8-bit Adam | 13085       |
| Loading Model with per_device_train_batch_size=4 + optim="adafactor" | 12295       |
| Loading Model with per_device_train_batch_size=4 + fp16=True | 13939       |
| Loading Model with per_device_train_batch_size=4 + fp16=True + gradient_checking=True | 7275        |
| Loading Model with per_device_train_batch_size=1 + gradient_accumulation_steps=4 | 8681        |
| Loading Model with per_device_train_batch_size=1 + gradient_accumulation_steps=4+ gradient_checkpointing=True | 6775        |
| Loading Model with per_device_train_batch_size=1 + gradient_accumulation_steps=4+ gradient_checkpointing=True + fp16=True and using accelerate. | 5363        |

## Using Multiple GPUs

There are data, tensor, and pipeline parallelism when working with multiple GPUs. Each of them has pros and cons, there does not exist an universally good solution that fits into any situation.

- Data Parallelism (DP): The same setup is replicated on all devices but we split the data and send them to different devices. One may see the acronym DDP, which refers to Distributed DP.

- Tensor Parallelism (TP): Splitting a tensor into multiple shards and process each shard on different devices; it is also called horizontal parallelism.

    ZeRO (Zero Redundancy Optimizer, ZeRO) is a preferred type of TP that does not require make changes to the model.

- Pipeline Parallelism (PP): Splitting a few layers of the model into a single GPU; it is also called vertical parallelism.

According to [Jason Phang](https://github.com/zphang/minimal-llama), the ZeRO is the more efficient method than PEFT and PP:

> There ought to be more efficient methods of tuning (DeepSpeed / ZeRO, NeoX) than the ones presented here, but folks may find this useful already.

## Reference

| Index | Name                                                         | Note                                           |
| ----- | ------------------------------------------------------------ | ---------------------------------------------- |
| 1     | https://huggingface.co/docs/transformers/perf_train_gpu_one  | Official Tutorial                              |
| 2     | https://huggingface.co/docs/transformers/perf_train_gpu_many | Official Tutorial                              |
| 3     | https://github.com/zphang/minimal-llama                      | Jason Phang                                    |
| 4     | https://huggingface.co/docs/transformers/main_classes/deepspeed | HuggingFace Documentation                      |
| 5     | https://huggingface.co/blog/4bit-transformers-bitsandbytes   | Fine-Tuning LLMs like llama, gpt-neox, and t5. |
| 6     | https://huggingface.co/blog/pytorch-ddp-accelerate-transformers | Official Tutorial                              |

# Using simpletransformers

## Minimal Working Example

`simpletransformers` could more quickly and cleanly train and evaluate PLMs compared to `transformers`, where the  `simpletransformers` library is based upon; it also comes with full support from `wandb`. Note that

- The number of steps is computed based on one GPU even though `model_args.n_gpu` is set to a different value. Therefore, we should **not** further divide `n_total_steps` by `model_args.n_gpu`.
- By default, there will be evaluations at the end of each epoch. Therefore, setting `n_eval=10` will lead to `model_args.num_epochs + n_eval` evaluations; in the example below, there will be 13 evaluations.

The following example fine-tunes `bert-base-uncased` on the `imdb` dataset:

```python
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from simpletransformers.classification import (
    ClassificationModel,
    ClassificationArgs
)

model_args = ClassificationArgs()

model_class = "roberta"
model_name = "roberta-base"

##################################################
# see full list of configurations:
# https://simpletransformers.ai/docs/usage/#configuring-a-simple-transformers-model
# critical settings
model_args.learning_rate = 1e-5
model_args.num_train_epochs = 3
model_args.train_batch_size = 32
model_args.eval_batch_size = 32
model_args.gradient_accumulation_steps = 1
model_args.fp16 = False
model_args.max_seq_length = 128
model_args.n_gpu = 4
model_args.use_multiprocessing = False
model_args.use_multiprocessing_for_evaluation = False

# saving settings
model_args.no_save = False
model_args.overwrite_output_dir = True
model_args.output_dir = "outputs/"

# the following mandates that only the best checkpoint will be saved; there will be only 1 checkpoint
model_args.best_model_dir = "{}/best_model".format(model_args.output_dir)
model_args.save_model_every_epoch = False
model_args.save_best_model = True
model_args.save_eval_checkpoints = False
model_args.save_steps = -1

# validation criterion
model_args.use_early_stopping = False
model_args.early_stopping_metric = "auroc"
model_args.early_stopping_metric_minimize = False

# evaluation settings
model_args.evaluate_during_training = True

# logging settings
model_args.silent = False
model_args.wandb_project = "simpletransformers"
model_args.wandb_kwargs = {
    "name": "sanity-check-imdb"
}

##################################################
# loading dataset

ds = load_dataset("imdb")

# data splits
# there has to be "text" and "labels" columns in the input dataframe
df = pd.DataFrame(ds["train"]).rename(columns={"label": "labels"})

train_df, eval_df = train_test_split(df.sample(frac=0.1), test_size=0.2)
test_df = pd.DataFrame(ds["test"]).sample(frac=0.1).rename(columns={"label": "labels"})

##################################################
# adaptive steps settings
# we will evaluate 10 times and log 100 times no matter how small the dataset

n_eval, n_log = 10, 100
n_total_steps = round(len(train_df) / model_args.train_batch_size) * model_args.num_train_epochs

model_args.evaluate_during_training_steps = max(1, round(n_total_steps / n_eval))
model_args.logging_steps = max(1, round(n_total_steps / n_log))
model_args.save_steps = -1

##################################################
# training
model = ClassificationModel(
    model_class,
    model_name,
    num_labels=2,
    args=model_args,
)

model.train_model(
    train_df=train_df,
    eval_df=eval_df
)

# test
result, model_outputs, wrong_predictions = model.eval_model(test_df)
```

## Validation and Early Stopping

- Validation

    Choosing which model checkpoint to save (aka. validation) depends on `early_stopping_metric` and `early_stopping_metric_minimize` even though early stopping itself is disabled.

- Early Stopping

    If we need to use early stopping, we need to also be aware of the other hyperparameters.

| Name                             | Default       | Note                                                         |
| -------------------------------- | ------------- | ------------------------------------------------------------ |
| `use_early_stopping`             | `False`       |                                                              |
| `early_stopping_metric`          | `"eval_loss"` | `eval_during_training` has to be `True`; it will use metrics computed during evaluation. |
| `early_stopping_metric_minimize` | `True`        |                                                              |
| `early_stopping_consider_epochs` | `False`       |                                                              |
| `early_stopping_patience`        | `3`           | Terminate training after `early_stopping_patience` evaluations without improvement specified by`early_stopping_delta`. |
| `early_stopping_delta`           | `0`           |                                                              |

```python
class ClassificationModel:
    
    def train_model(
        self,
        train_df,
        multi_label=False,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_df=None,
        verbose=True,
        **kwargs,
    ):
        // ...
        global_step, training_details = self.train(
            train_dataloader,
            output_dir,
            multi_label=multi_label,
            show_running_loss=show_running_loss,
            eval_df=eval_df,
            verbose=verbose,
            **kwargs,
        )
        // ...
    
    
    def train(
        self,
        train_dataloader,
        output_dir,
        multi_label=False,
        show_running_loss=True,
        eval_df=None,
        test_df=None,
        verbose=True,
        **kwargs,
    ):
    	//...
    	best_eval_metric = None
        
        // ...
        if not best_eval_metric:
            best_eval_metric = results[args.early_stopping_metric]
            self.save_model(
                args.best_model_dir,
                optimizer,
                scheduler,
                model=model,
                results=results,
            )
    	// ...
```

# Using Sentence-Transformers

## Overview

-   `sentence_transformer` is built with `torch` despite a resemblance to the `keras` API.
-   The famous METB benchmark is also largely built on top of the `sentence_transformer` library.

## Fine-Tuning Embeddings

Besides an easy interface to generate embeddings, the `sentence_transformers` library also supports fine-tuning the provided embedding models. The following data formats all have their corresponding loss functions without a need to convert data to a specific format (for example, triplets) (see [blog](https://huggingface.co/blog/how-to-train-sentence-transformers)). 

Note that these loss functions come from the `sentence_transformers` library rather than `torch` or `transformers`. These loss functions have been discussed in a [blog post](https://omoindrot.github.io/triplet-loss) that is not affiliated with the developers of `sentence_transformers`.

| Index | Description                                                  | Data                                                         | Loss                                                         | Note                                                         |
| ----- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1     | A pair of sentences and a label                              | `(premise, hypothesis, label)`                               | `ContrastiveLoss`;<br />`SoftmaxLoss`;<br />`CosineSimilarityLoss` |                                                              |
| 2     | Individual sentence and corresponding label                  | `(text, label)`                                              | `BatchHardTripletLoss` and variants                          | "batch hard" performs  best in the [blog post](https://omoindrot.github.io/triplet-loss). |
| 3     | A pair of **similar** sentences                              | `(query, response)`, `(src_lang, tgt_lang)`, `(full_text, summary)`, `(text1, text2)` (e.g., QQP), `(text, entailed_text)`  (e.g., NLI) | `MultipleNegativeRankingLoss`;<br />`MegaBatchMarginLoss`    | Frequent                                                     |
| 4     | A triplet of sentences of an positive, a positive, and a negative | `(anchor, positive, negative)`                               | `TripletLoss`                                                | Rare as it requires offline mining                           |

Here is a minimal working example of fine-tuning representation using `sst2` dataset; we could optionally evaluate the fine-tuned model on the MTEB benchmark as it is also built with `sentence_transformer` library.

Note that:

-   The `sentence_transformers` does not have a native support for `wandb` as `simpletransformers`. We could only monitor one score through the `log_with_wandb()` with an exactly same signature. The `score` it monitors depends on which specific evaluator is used (see the complete list of evaluators [here](https://www.sbert.net/docs/package_reference/evaluation.html)).

    When working with `TripletEvaluator` as in the example below. The returned metric a ratio of the number of triplets among all triplets that satisfy $d(a, p) < d(a, n)$.

-   We could easily replace the model with models available on the HuggingFace hub.

```python
import os
import wandb
import random
import logging

import pandas as pd

from datetime import datetime
from collections import defaultdict
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    SentencesDataset
)
from sentence_transformers.evaluation import (
    TripletEvaluator,
)

from sentence_transformers import LoggingHandler
from sentence_transformers.losses import (
    BatchHardTripletLoss,
)

from datasets import load_dataset
from torch.utils.data import DataLoader


logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

def triplets_from_labeled_dataset(
    records,
    text_column="sentence",
    label_column="label"
):
    # Create triplets for a [(label, sentence), (label, sentence)...] dataset
    # by using each example as an anchor and selecting randomly a
    # positive instance with the same label and a negative instance with a different label

    input_examples = [
        InputExample(guid=str(guid), texts=[record[text_column]], label=record[label_column])
        for guid, record in enumerate(records)
    ]

    triplets = []
    label2sentence = defaultdict(list)
    for inp_example in input_examples:
        label2sentence[inp_example.label].append(inp_example)

    for inp_example in input_examples:
        anchor = inp_example

        if len(label2sentence[inp_example.label]) < 2: #We need at least 2 examples per label to create a triplet
            continue

        positive = None
        while positive is None or positive.guid == anchor.guid:
            positive = random.choice(label2sentence[inp_example.label])

        negative = None
        while negative is None or negative.label == anchor.label:
            negative = random.choice(input_examples)

        triplets.append(InputExample(texts=[anchor.texts[0], positive.texts[0], negative.texts[0]]))

    return triplets

##################################################

model_name = 't5-base'
num_epochs = 10

##################################################

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
wandb.init(
    project="sentence_transformers",
    name=f"{model_name}-{current_time}"
)

##################################################
# model

output_path = (
    "output/"
    + model_name
    + "-"
    + current_time
)

model = SentenceTransformer(model_name)

##################################################

def get_dataloader(df, split, text_column, label_column, batch_size=8):
    records = df.to_dict("records")
    examples = [
        InputExample(texts=[record[text_column]], label=record[label_column])
        for record in records
    ]
    dataset = SentencesDataset(
        examples=examples,
        model=model,
    )
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    return dataloader

##################################################
# data

ds = load_dataset("sst2")

train_df = pd.DataFrame(ds["train"])
val_df = pd.DataFrame(ds["validation"])
test_df = pd.DataFrame(ds["test"])

train_dataloader = get_dataloader(train_df, "train", text_column="sentence", label_column="label")

##################################################

train_loss = BatchHardTripletLoss(model=model)
val_evaluator = TripletEvaluator.from_input_examples(
    triplets_from_labeled_dataset(val_df[["sentence", "label"]].to_dict("records")),
    name="eval"
)
val_evaluator(model)

##################################################

def log_with_wandb(score, epoch, steps):
    # https://docs.wandb.ai/ref/python/log
    wandb.log(
        data={"score": score},
        step=steps,
    )

warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data

model.fit(
    [(train_dataloader, train_loss)],
    show_progress_bar=True,
    epochs=num_epochs,
    evaluator=val_evaluator,
    evaluation_steps=50,
    warmup_steps=warmup_steps,
    output_path=output_path,
    callback=log_with_wandb
)
##################################################

test_evaluator = TripletEvaluator.from_input_examples(
    triplets_from_labeled_dataset(test_df[["sentence", "label"]].to_dict("records")),
    name="test"
)
model.evaluate(test_evaluator)
```

As our goal is not evaluating triplets but the quality of clustering, we could define our own evaluator.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score

from sentence_transformers.evaluation import (
    SentenceEvaluator,
)

class ClusteringEvaluator(SentenceEvaluator):
    def __init__(self, texts, labels, batch_size=32, show_progress_bar=False):
        self.texts = texts
        self.labels = labels
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1):
        embeddings = model.encode(
            self.texts, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True
        )
        y_pred = KMeans(n_clusters=len(set(self.labels)), n_init="auto").fit_predict(embeddings)
        score = v_measure_score(
            labels_true=self.labels,
            labels_pred=y_pred
        )

        return score
```

## Customization

### Saving Checkpoints

Similar to `simpletransformers`, `sentence_transformer` could save best checkpoints according to the evaluation metric. Model saving are controlled by `_eval_during_training()` and `_save_checkpoint()` functions. 

-   If `save_best_model=True`, the best model will be saved at the root directory of the `output_path`. Saving best checkpoint is **enabled by default**.

-   If we want to save additional checkpoints, these additional checkpoints will be saved at `checkpoint_path`; the total number of saved checkpoints is governed by `checkpoint_save_steps` and `checkpoint_save_total_limit`. Different checkpoints will be stored in the folder named `<step>`.

    Saving additional checkpoints is **disabled** **by default**.

