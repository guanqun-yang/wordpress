---
title: Coding Notes |  LLM Practice
tags: 
- LLM
categories:
- Coding
---

Table of Contents

1. [Prompt Engineering](#prompt-engineering)
	1. [Write Clear Instructions](#write-clear-instructions)
	1. [Provide Reference Texts](#provide-reference-texts)
	1. [Split Complex Tasks into Simpler Subtasks](#split-complex-tasks-into-simpler-subtasks)
	1. [Give Model Time to Think or CoT](#give-model-time-to-think-or-cot)
	1. [Use External Tools](#use-external-tools)
	1. [Test Changes Systematically](#test-changes-systematically)
1. [Fine-Tuning](#fine-tuning)
	1. [Overview](#overview)
	1. [Recipe](#recipe)
		1. [Workflow](#workflow)
		1. [Step 1: Preparing Data](#step-1-preparing-data)
		1. [Step 2: Uploading Data](#step-2-uploading-data)
		1. [Step 3: Fine-Tuning](#step-3-fine-tuning)
		1. [Step 4: Evaluation and Iteration](#step-4-evaluation-and-iteration)
			1. [Data Quality and Quantity](#data-quality-and-quantity)
			1. [Model Hyperparameters](#model-hyperparameters)
1. [Reference](#reference)

# Prompt Engineering

The OpenAI [official documentation](https://platform.openai.com/docs/guides/prompt-engineering) summarizes 6 tricks for prompt engineering.

## Write Clear Instructions

The LM could not do that is not instructed by the user automatically.

## Provide Reference Texts

## Split Complex Tasks into Simpler Subtasks

Solving multiple problems in a cascade fashion often leads to smaller eventual error rate compared to solving the problem at the same time.

## Give Model Time to Think or CoT

## Use External Tools

It is better to use tools to solve the tasks that require algorithmic solutions; the LM is good at reasoning rather than solving problems algorithmically. 

## Test Changes Systematically

The prompts that work well on small number of samples in the playground may not work as well for a representative set of test samples. It is important to run evaluation on the large test set every time we make non-trivial changes to the prompt.

# Fine-Tuning

## Overview

-   As of 2023-11-15, OpenAI allows fine-tuning `gpt-3.5-turbo`, `davinci-002`, and `babbage-002` models. OpenAI will soon support fine-tuning `gpt-4`. Besides, it is possible to fine-tune already fine-tuned models
-   Fine-tuning is **discouraged** unless we have shown that none of the below works. This is because it is faster to iterate with prompts in the playground than fine-tuned models.
    -   Prompt Engineering: We must **closely** follow the content in [1] for prompt engineering.
    -   Prompt Chaining: Breaking complex tasks into multiple prompts.
    -   Function Calling
-   Reasons for Fine-Tuning
    -   Reducing the length of prompts or reducing latency. Fine-tuning models could save up to 90% of the tokens compared to zero-shot or few-shot prompting ([blog](https://openai.com/blog/gpt-3-5-turbo-fine-tuning-and-api-updates)). Furthermore, fine-tuning a smaller model (for example, `gpt-3.5-turbo`) could often match the performance of a larger model (for exampe, `gpt-4`), therefore reducing latency.
    -   Improving performance for tasks that are hard to articulate using prompts (i.e., tasks that "show, not tell").

## Recipe

### Workflow

-   Unlike older models, the `gpt-3.5-turbo` could be fine-tuned with as few as 10 examples. There **will be** clear improvement when fine-tuning with 50 to 100 examples.
-   It is better to start fine-tuing using 50 examples and see if there is improvement. If there is no clear improvement, we must redesign the data.

### Step 1: Preparing Data

We need to prepare data into a `.jsonl` following the format below; each line in the `.jsonl` will be an example; the token limit of each example is 4096 for `gpt-3.5-turbo`. We could estimate the token usage of a fine-tuning job using `num_tokens_from_messages()` function ([doc](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)).

-   Chat Models

    In the example below, the goal is to fine-tune a model that could generate sarcastic responses. Each sample should be formatted as follows.

```bash
{
    "messages": [
        {
            "role": "system",
            "content": "Marv is a factual chatbot that is also sarcastic."
        },
        {
            "role": "user",
            "content": "What's the capital of France?"
        },
        {
            "role": "assistant",
            "content": "Paris, as if everyone doesn't know that already."
        }
    ]
}
```

-   Other Models

```bash
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
```

### Step 2: Uploading Data

We first need to make sure the `openai` is most updated using `pip install -U openai`. Then we could upload the data.

```python
from openai import OpenAI
client = OpenAI()

message =  client.files.create(
  file=open("mydata.jsonl", "rb"),
  purpose="fine-tune"
)
# message:
# FileObject(
# 	id='file-Y0Q82yniynZAN7TeZaEXcbYg', 
# 	bytes=234106, 
#		created_at=1700031010, 
#   filename='fine_tuning_data.jsonl', 
#   object='file', 
#   purpose='fine-tune', 
#   status='processed', 
#   status_details=None
# )
```

### Step 3: Fine-Tuning

Now OpenAI supports fine-tuning models using an UI (i.e., `https://platform.openai.com/finetune`). We could also submit a fine-tuning job using Python code below. Note that

-    `filename` is returned in Step 2.
-   `model` could be `gpt-3.5-turbo` or older models.

We could optionally tune the hyperparameters of fine-tuning.

```python
from openai import OpenAI
client = OpenAI()

client.fine_tuning.jobs.create(
  training_file="filename", 
  model="gpt-3.5-turbo",
  # optional, see details below
  hyperparameters={ 
    "n_epochs":2
  }
)
```

We could monitor the status of fine-tuning on the OpenAI website. If using code is preferred, we could use one of the commands below.

```python
from openai import OpenAI
client = OpenAI()

# Retrieve the state of a fine-tune
client.fine_tuning.jobs.retrieve("ftjob-abc123")

# Return the training metrics based on the command above, such as loss, accuracy
content = client.files.retrieve_content("result-file")

# List up to 10 events from a fine-tuning job
client.fine_tuning.jobs.list_events(fine_tuning_job_id="ftjob-abc123", limit=10)

# List 10 fine-tuning jobs
client.fine_tuning.jobs.list(limit=10)
```

### Step 4: Evaluation and Iteration

After fine-tuning, we could evaluate the fine-tuned model on the held-out test set. If the performance is not satisfying, we should check the data quality from the aspects below.

#### Data Quality and Quantity

Data quality should be prioritized to data quantity: a smaller amout of high-quality data is generally better than a larger amount of low quality data.

-   Training Data Lacking Consistency

    As a rule of thumb, an inter-annotator agreement of 70% is **low**: if the humans could not agree on the labels, it is unlike the model could do better than humans.

-   Training Label Distribution Different from Testing Label Distribution

We could start from the cases that the fine-tuned model makes mistakes and starts to iterate from there. If it is indeed the case that the data quantity is the issue, we could estimate the gains by (1) fine-tuning a second model that uses half of the current data, and (2) estimating the performance difference of two models on the test set.

#### Model Hyperparameters

We could change 3 hyperparameters: number of epochs, learning rate multiplier, and batch size. The following are some typical scenarios and the corresponding action:

| Task                                        | Scenario                                | Action                                 |
| ------------------------------------------- | --------------------------------------- | -------------------------------------- |
| Task with single or a few ideal completions | Generations not following training data | Increasing `n_epochs` from 3 to 4 or 5 |
| Creative tasks                              | Generations of reduced diversity        | Desceacing `n_epochs` from 3 to 1 or 2 |
| /                                           | Not converging                          | Increaing `learning_rate_multiplier`   |

# Reference

1.   [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
2.   [OpenAI Fine-Tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
