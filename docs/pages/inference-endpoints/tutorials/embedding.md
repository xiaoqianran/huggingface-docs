# Build an embedding pipeline with datasets

This tutorial will guide you through deploying an embedding endpoint and building a Python script to efficiently process datasets with embeddings. We'll use the powerful [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) model to create high-quality embeddings for your data.

This tutorial focuses on creating a production-ready script that can process any dataset and add embeddings using the **Text Embeddings Inference (TEI)** engine for optimized performance.

## Create your embedding Endpoint

First, we need to create an Inference Endpoint optimized for embeddings.

Start by navigating to the Inference Endpoints UI, and once you have logged in, click the **Catalog** button.

![new-button](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/quick_start/1-new-button.png)

From there you'll be directed to the catalog. The Model Catalog consists of popular models which have tuned configurations to work as one-click
deploys. You can search for embedding models or create a custom endpoint.

![catalog](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/quick_start/2-catalog.png)

For this tutorial, we'll use the Qwen3-Embedding-4B model available in the Inference Endpoints Model Catalog. Note if it's ever not in the catalog, you can deploy a model as a custom Endpoint from the Hugging Face Hub by entering the model repository ID `Qwen/Qwen3-Embedding-4B`.

For embedding models, we recommend:
- **GPU**: NVIDIA, T4, L4 or A10G for good performance.
- **Instance Size**: x1 (sufficient for most embedding workloads)
- **Auto-scaling**: Enable scale-to-zero to save costs by switching the endpoint to a paused state when it's not in use.
- **Timeout**: Set a timeout of 10 minutes to avoid long-running requests. You should define a timeout based on how you expect your endpoint to be used.

If you're looking for a model with less compute requirements, you can use the [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model.

The Qwen3-Embedding-4B model will automatically use the **Text Embeddings Inference (TEI)** engine, which provides optimized inference and automatic batching.

Click "Create Endpoint" to deploy your embedding service.

![config](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/quick_start/4-config.png)

This may take about 5 minutes to initialize.

## Test your Endpoint

Once your Inference Endpoint is running, you can test it directly in the playground. It accepts text input and returns high-dimensional vectors.

![playground](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/embedding-tutorial/assets/tutorials/embedding/playground.png)

Try entering some sample text like "Machine learning is transforming how we process data" and see the embedding output.

## Get your Endpoint's details

To use your endpoint programmatically, you'll need these details from the Endpoint's [Overview](https://endpoints.huggingface.co/):

- **Base URL**: `https://<endpoint-name>.endpoints.huggingface.cloud/v1/`
- **Model name**: The name of your endpoint
- **Token**: Your HF token from [settings](https://huggingface.co/settings/tokens)

![endpoint-details](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/tutorials/embedding/endpoint-page.png)

## Building the embedding script

Now let's build a script step by step to process datasets with embeddings. We'll break it down into logical blocks.

### Step 1: Set up dependencies and imports

We'll use the OpenAI client to connect to the endpoint and the datasets library to load and process the dataset. So let's install the required packages:

```bash
pip install datasets openai
```

Then, set up your imports in a new Python file:

```python
import os
from datasets import load_dataset
from openai import OpenAI
```

### Step 2: Configure the connection

Set up the configuration to connect to your Inference Endpoint based on the details you collected in the previous step.

```python
