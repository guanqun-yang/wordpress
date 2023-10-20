---
title: Research Notes | Query Generation
tags: 
- Retrieval
categories:
- Research
---

[toc]

# Problem Statement

The paper [2] notes the difficulty of optimizing the query for neural retrieval models.

> However, neural systems lack the interpretability of bag-of-words models; it is not trivial to connect a query change to a change in the latent space that ultimately determines the retrieval results. 

The query generation aims to find out "what should have been asked" for a given passage. More formally,  if we have a list of documents $D$ that are aligned with our goal (or true underlying query) $q$, is it possible to search for its approximated version $\hat{q}$ that returns $D$ as relevant documents with high probability?

# Background

- Rocchio Algorithm for Query Expansion

    Rocchio algorithm is used by search engines in the backend to improve the user's initial search queries. For example, if the initial search query is "beaches in Hawaii," if the backend finds that many users click webpages (this is one type of pseudo relevance feedback, PRF) that contain the word "sunset," then the word "sunset" will be added to the initial search query, making the real search query "beaches in Hawaii sunset."

    There could be more nuanced PRFs than binary click and not-click action. For example, how long the users stay on a webpage, whether the user eventually returns to the search result page.

# Adolphs et al., EMNLP 2022

There are two contributions of this paper:

- Query Inverter: An inverter that converts embeddings back to queries.
- Query Generation: A way to refine the initial generic query (plus a gold document) so that the gold passage will become one of the top ranked documents given a refined query.

## Query Inverter

The authors use the GTR model to generate embeddings from 3M queries from the PAQ dataset [5], thus forming a datasets ${{(q_i, \mathbf{e}_i)}} _ {i=1} ^ N$. Then the authors fine-tune a T5-base **decoder** by reversing GTR's input and output as ${{ (\mathbf{e} _ i, q_i) }} _ {i = 1} ^ N$; this process of fine-tuning T5-base requires 130 hours on 16 TPU v3

> - Note: The major limitation of this process is that the fine-tuned T5-base could only invert the embeddings of a **fixed** GTR model. When working with a GTR model fint-tuned for a new application scenario, the **expensive** fine-tuning of T5-base has to repeat.

The paper mentions the following interesting application:

> We thus evaluate the decoder quality by starting from a document paragraph, decoding a query from its embedding and then running the GTR search engine on that query to check if this query retrieves the desired paragraph as a top-ranked result.

For example, in the **sanity check** experiment in Table 3, if we use the gold passage to reconstruct the query, and then use this generated query to find relevant passages, the rank of the gold passages improves upon the original approach of querying blindly.

![image-20231019132020003](https://raw.githubusercontent.com/guanqun-yang/remote-images/master/2023/10/upgit_20231019_1697736020.png)

## Query Generation

Given an original query $\mathbf{q}$ and a gold document $\mathbf{d}$, the authors propose to generate new query embedding based on the linear combination of $\mathbf{q}$ and $\mathbf{d}$; in the experiments, the authors generate select $\kappa$ so that there are 20 new query embeddings.

The intuition of this idea that the optimal query should be between the initial generic (yet relevant) query and the gold passage. The gold passage could be the answer to multiple different questions; we therefore need to use the initial generic query $\mathbf{q}$ to guide the generation of new queries.
$$
\mathbf{q} _ \kappa \leftarrow \frac{\kappa}{k} \mathbf{d} + (1 - \frac{\kappa}{k}) \mathbf{q}
$$
![image-20231019134107901](https://raw.githubusercontent.com/guanqun-yang/remote-images/master/2023/10/upgit_20231019_1697737267.png)

Based on a dataset of succesfully reformulated queries (that is, the gold passage is top ranked by the reformulated query), the authors fine-tune a T5 model with original queries as input and reformulated queries as output; they call this query suggestion model.

The authors find that their query suggestion models (`qsT5` and `qsT5-plain`) improves the retrieval performance when compared with query expansion baselines, including a strong PRF baseline RM3.

![image-20231019134801317](https://raw.githubusercontent.com/guanqun-yang/remote-images/master/2023/10/upgit_20231019_1697737681.png)

# Vec2Text

## Method

Recall the chain rule:
$$
p(a, b, c) = p(a\vert b, c) \cdot p(b\vert c) \cdot p(c)
$$
The proposed approach is inverting an embedding $\mathbf{e}$ from an arbitrary embedding function $\phi(\cdot)$ (for example, OpenAI embedding API) back to text $x^{(t+1)}$ iteratively from an initial guess $x^{(0)}$. This correction could take multiple steps; the total number of steps should not be large (up to 40).
$$
p\left(x^{(0)}\vert \mathbf{e}\right) = p\left(x^{(0)}\vert \mathbf{e}, \emptyset, \phi(\emptyset)\right) \rightarrow \cdots \rightarrow
p\left(x^{(t+1)} \vert \mathbf{e}\right) = \sum _ {x ^ {(t)}} p\left(x ^ {(t)}\vert \mathbf{e}\right) \cdot \boxed{p\left(x^{(t+1)} \vert \mathbf{e}, x^{(t)}, \phi(x ^ {(t)})\right)}
$$
The boxed term is operationalized as a T5-base model. To make sure an arbitrary embedding fits into the dimension of T5-base, the authors further use a MLP to project arbitrary embeddings of size $d$ to the right size $s$.
$$
\mathrm{EmbToSeq}(\mathbf{e})=\mathbf{W} _ 2 \sigma(\mathbf{W} _ 1 \mathbf{e})
$$
The authors propose to feed the concatentation of 4 vectors - $\mathrm{EmbToSeq}(\mathbf{e})$, $\mathrm{EmbToSeq}(\mathbf{\hat{e}})$, $\mathrm{EmbToSeq}(\mathbf{e} - \hat{\mathbf{e}})$, and embeddings of $x ^ {(t)}$ using T5-base  (the total size input size is $3s + n$) to the model and fine-tune the T5-base with regular LM objective.

In the experiments, the authors invert the same model as the GTR model as Adolphs et al. and OpenAI text embedding API; the fine-tuning of each T5-base on each dataset took 2 days on 4 A6000 GPUs.

> - Difference from Adolphs et al.
>
>     Even though the idea to invert the GTR model and how this inverter is trained is quite similar, Adolphs et al. does not consider the multi-step correction. Further, they do not provide the code.

## Code

The authors not only open-source the code to fine-tune the model; they also provide the code to create the library `vec2text`. 

### Minimal Working Example

The provided library is very easy-to-use. The following is a minimal working example:

```python
import vec2text
from langchain.embeddings import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

query = "What is your favoriate opera?"
positives = [
    "I love Lucia di Lammermoor because Luica's powerful presence is really inspiring.",
    "Le Nozze di Figaro is my favorite because of the fun plot and timeless Mozart music.",
]
negatives = [
    "I love pepperoni pizza",
    "Cancun is my favoriate holiday destination."
]

query_embedding = embedding_model.embed_query(query)
positive_embeddings = embedding_model.embed_documents(positives)
negative_embeddings = embedding_model.embed_documents(negatives)
```

### Anatomy



# Additional Notes

- This idea could generalize to prompt search as prompt engineering could be seen as a more general form of query refinement to better elicit the knowledge from an LLM. However, the difference is that we often do not have desired output; this makes the search for the prompts difficult. Eventually, the idea could work for prompt engineering only when we have at least one ideal output.

    The [Dynasour system](https://arxiv.org/abs/2305.14327) from UCLA is one such attempt: they are trying to create instruction tuning data from regular HuggingFace datasets; these HuggingFace datasets do not come with instructions.

- The paper [2] shows a novel way of manipulating embeddings - using Seq2Seq model's decoder only. This is not previously possible for encoder-only, encoder-decoder, or decoder-only models.

- Gradients provide more information than embeddings, as is noted by [4].

    > However, such techniques do not apply to textual inversion: the gradient of the model is relatively high-resolution; we consider the more difficult problem of recovering the full input text given only a single dense embedding vector.

- In the embedding space, two embeddings could collide even though they have no token overlap [7].

# Reference

1. [[2109.00527] Boosting Search Engines with Interactive Agents](https://arxiv.org/abs/2109.00527) (Adolphs et al., TMLR 2022): This is **a feasibility study** of an emsemble of BM25 plus an interpretable reranking scheme to work on par DPR on the `natural_questions` dataset; this is consistent with DPR in its evaluation. The main advantage is intepretability rather than performance.

    ![image-20231019112747929](https://raw.githubusercontent.com/guanqun-yang/remote-images/master/2023/10/upgit_20231019_1697729268.png)

2. [Decoding a Neural Retriever’s Latent Space for Query Suggestion](https://aclanthology.org/2022.emnlp-main.601) (Adolphs et al., EMNLP 2022)

3. [What Are You Token About? Dense Retrieval as Distributions Over the Vocabulary](https://aclanthology.org/2023.acl-long.140) (Ram et al., ACL 2023): This paper proposes a method to project embeddings onto the vocabulary and obtains a distribution over the tokens. The motivation for this paper is interpretability.

4. [[2310.06816] Text Embeddings Reveal (Almost) As Much As Text](https://arxiv.org/abs/2310.06816) (Morris et al., EMNLP 2023)

5. [Large Dual Encoders Are Generalizable Retrievers](https://aclanthology.org/2022.emnlp-main.669) (Ni et al., EMNLP 2022): This paper proposes the Generalization T5 dense Retrievers (GTR) model that many papers build their solutions upon.

6. [PAQ: 65 Million Probably-Asked Questions and What You Can Do With Them](https://aclanthology.org/2021.tacl-1.65) (Lewis et al., TACL 2021)

7. [Adversarial Semantic Collisions](https://aclanthology.org/2020.emnlp-main.344) (Song et al., EMNLP 2020)