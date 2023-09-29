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

`simpletransformers` could more quickly and cleanly train and evaluate PLMs compared to `transformers`, where the  `simpletransformers` library is based upon; it also comes with full support from `wandb`. For example, fine-tuning `bert-base-uncased` on the `imdb` dataset:

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
model_args.wandb_project = "simpletransformers"
model_args.wandb_kwargs = {
    "name": "Test"
}

ds = load_dataset("imdb")

# data splits
# there has to be "text" and "labels" columns in the input dataframe
df = pd.DataFrame(ds["train"]).rename(columns={"label": "labels"})

train_df, eval_df = train_test_split(df.sample(frac=0.1), test_size=0.2)
test_df = pd.DataFrame(ds["test"]).sample(frac=0.1).rename(columns={"label": "labels"})

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

