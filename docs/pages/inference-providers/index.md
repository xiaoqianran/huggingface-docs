# Inference Providers

    
    

Hugging Face’s Inference Providers give developers access to hundreds of machine learning models, powered by world-class inference providers. They are also integrated into our client SDKs (for JS and Python), making it easy to explore serverless inference of models on your favorite providers.

## Quick Setup for Agents

Using a coding agent? Point it at Inference Providers to run [the latest open models available](https://huggingface.co/models?inference_provider=all) with a single Hugging Face token. Pick your tool below to jump straight to its setup guide.

    
        
        OpenCode
    
    
        
        Pi
    
    
        
        Codex
    
    
        
        Claude Code
    
    
        
        Hermes Agent
    

## Partners

Our platform integrates with leading AI infrastructure providers, giving you access to their specialized capabilities through a single, consistent API. Here's what each partner supports:

| Provider                                     | Chat completion (LLM) | Chat completion (VLM) | Feature Extraction | Text to Image | Text to video | Speech to text |
| -------------------------------------------- | :-------------------: | :-------------------: | :----------------: | :-----------: | :-----------: | :------------: |
| [Baseten](./providers/baseten)               |          ✅           |          ✅           |                    |               |               |                |
| [Cerebras](./providers/cerebras)             |          ✅           |                       |                    |               |               |                |
| [Cohere](./providers/cohere)                 |          ✅           |          ✅           |                    |               |               |                |
| [DeepInfra](./providers/deepinfra)           |          ✅           |          ✅           |                    |               |               |                |
| [Fal AI](./providers/fal-ai)                 |                       |                       |                    |      ✅       |      ✅       |       ✅       |
| [Featherless AI](./providers/featherless-ai) |          ✅           |          ✅           |                    |               |               |                |
| [Fireworks](./providers/fireworks-ai)        |          ✅           |          ✅           |                    |               |               |                |
| [Groq](./providers/groq)                     |          ✅           |          ✅           |                    |               |               |                |
| [HF Inference](./providers/hf-inference)     |          ✅           |          ✅           |         ✅         |      ✅       |               |       ✅       |
| [Novita](./providers/novita)                 |          ✅           |          ✅           |                    |               |      ✅       |                |
| [Nscale](./providers/nscale)                 |          ✅           |          ✅           |                    |      ✅       |               |                |
| [OVHcloud AI Endpoints](./providers/ovhcloud)|          ✅           |          ✅           |                    |               |               |                |
| [Public AI](./providers/publicai)            |          ✅           |                       |                    |               |               |                |
| [Replicate](./providers/replicate)           |                       |                       |                    |      ✅       |      ✅       |       ✅       |
| [Scaleway](./providers/scaleway)             |          ✅           |                       |         ✅         |               |               |                |
| [Together](./providers/together)             |          ✅           |          ✅           |         ✅         |      ✅       |      ✅        |       ✅       |
| [WaveSpeedAI](./providers/wavespeed)         |                       |                       |                    |      ✅       |      ✅       |                |
| [Z.ai](./providers/zai-org)                  |          ✅           |          ✅           |                    |               |               |                |

## Why Choose Inference Providers?

When you build AI applications, it's tough to manage multiple provider APIs, comparing model performance, and dealing with varying reliability. Inference Providers solves these challenges by offering:

**Instant Access to Cutting-Edge Models**: Go beyond mainstream providers to access thousands of specialized models across multiple AI tasks. Whether you need the latest language models, state-of-the-art image generators, or domain-specific embeddings, you'll find them here.

**Zero Vendor Lock-in**: Unlike being tied to a single provider's model catalog, you get access to models from Cerebras, Groq, Together AI, Replicate, and more — all through one consistent interface.

**Production-Ready Performance**: Built for enterprise workloads with the reliability your applications demand.

Here's what you can build:

- **Text Generation**: Use Large language models with tool-calling capabilities for chatbots, content generation, and code assistance
- **Image and Video Generation**: Create custom images and videos, including support for LoRAs and style customization
- **Search & Retrieval**: State-of-the-art embeddings for semantic search, RAG systems, and recommendation engines
- **Traditional ML Tasks**: Ready-to-use models for classification, NER, summarization, and speech recognition

⚡ **Get Started for Free**: Inference Providers includes a generous free tier, with additional credits for [PRO users](https://hf.co/subscribe/pro) and [Team & Enterprise organizations](https://huggingface.co/enterprise).

## Key Features

- **🎯 All-in-One API**: A single API for text generation, image generation, document embeddings, NER, summarization, image classification, and more.
- **🔀 Multi-Provider Support**: Easily run models from top-tier providers like fal, Replicate, Together AI, and others.
- **🚀 Scalable & Reliable**: Built for high availability and low-latency performance in production environments.
- **🔧 Developer-Friendly**: Simple requests, fast responses, and a consistent developer experience across Python and JavaScript clients.
- **👷 Easy to integrate**: Drop-in replacement for the OpenAI chat completions API.
- **💰 Cost-Effective**: No extra markup on provider rates.

## Getting Started

Inference Providers works with your existing development workflow. Whether you prefer Python, JavaScript, or direct HTTP calls, we provide native SDKs and OpenAI-compatible APIs to get you up and running quickly.

We'll walk through a practical example using [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b), a state-of-the-art open-weights conversational model.

### Inference Playground

Before diving into integration, explore models interactively with our [Inference Playground](https://huggingface.co/playground). Test different [chat completion models](http://huggingface.co/models?inference_provider=all&sort=trending&other=conversational) with your prompts and compare responses to find the perfect fit for your use case.

Prefer the terminal? Run `hf models ls --warm` with the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-models-list) to list every model served by at least one provider, and add `--json` for scripts and agents. See the [Hub API](./hub-api) page for the full set of filters.

### Authentication

You'll need a Hugging Face token to authenticate your requests. Create one by visiting your [token settings](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained) and generating a `fine-grained` token with `Make calls to Inference Providers` permissions.

For complete token management details, see our [security tokens guide](https://huggingface.co/docs/hub/en/security-tokens).

### Quick Start - LLM

Let's start with the most common use case: conversational AI using large language models. This section demonstrates how to perform chat completions using DeepSeek V3, showcasing the different ways you can integrate Inference Providers into your applications.

Whether you prefer our native clients, want OpenAI compatibility, or need direct HTTP access, we'll show you how to get up and running with just a few lines of code.

#### Python

Here are three ways to integrate Inference Providers into your Python applications, from high-level convenience to low-level control:

For convenience, the `huggingface_hub` library provides an [`InferenceClient`](https://huggingface.co/docs/huggingface_hub/guides/inference) that automatically handles provider selection and request routing.

In your terminal, install the Hugging Face Hub Python client and log in:

```shell
pip install huggingface_hub
hf auth login # get a read token from hf.co/settings/tokens
```

You can now use the client with a Python interpreter.

By default, our system automatically selects the fastest available provider for the specified model (equivalent to the `:fastest` policy — highest throughput in tokens per second).

You can change the provider selection policy by appending a policy suffix to the model id: `:cheapest` for the most cost-efficient provider (lowest price per output token), or `:preferred` to follow your preference order in [Inference Provider settings](https://hf.co/settings/inference-providers). For example, `openai/gpt-oss-120b:cheapest`.

You can also select the provider of your choice by appending the provider name to the model id (e.g. `"openai/gpt-oss-120b:groq"`).

```python
import os
from huggingface_hub import InferenceClient

client = InferenceClient()

completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": "How many 'G's in 'huggingface'?"
        }
    ],
)

print(completion.choices[0].message)
```

If you're already using OpenAI's Python client, then you need a **drop-in OpenAI replacement**. Just swap-out the base URL to instantly access hundreds of additional open-weights models through our provider network.

By default, our system automatically selects the fastest available provider for the specified model (equivalent to the `:fastest` policy — highest throughput in tokens per second).

You can change the provider selection policy by appending a policy suffix to the model id: `:cheapest` for the most cost-efficient provider (lowest price per output token), or `:preferred` to follow your preference order in [Inference Provider settings](https://hf.co/settings/inference-providers). For example, `openai/gpt-oss-120b:cheapest`.

You can also select the provider of your choice by appending the provider name to the model id (e.g. `"openai/gpt-oss-120b:groq"`).

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="openai/gpt-oss-120b:fastest",
    messages=[
        {
            "role": "user",
            "content": "How many 'G's in 'huggingface'?"
        }
    ],
)
```

For maximum control and interoperability with custom frameworks, use our OpenAI-compatible REST API directly.

By default, our system automatically selects the fastest available provider for the specified model (equivalent to the `:fastest` policy — highest throughput in tokens per second).

You can change the provider selection policy by appending a policy suffix to the model id: `:cheapest` for the most cost-efficient provider (lowest price per output token), or `:preferred` to follow your preference order in [Inference Provider settings](https://hf.co/settings/inference-providers). For example, `openai/gpt-oss-120b:cheapest`.

You can also select the provider of your choice by appending the provider name to the model id (e.g. `"openai/gpt-oss-120b:groq"`).

```python
import os
import requests

API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
payload = {
    "messages": [
        {
            "role": "user",
            "content": "How many 'G's in 'huggingface'?"
        }
    ],
    "model": "openai/gpt-oss-120b:fastest",
}

response = requests.post(API_URL, headers=headers, json=payload)
print(response.json()["choices"][0]["message"])
```

#### JavaScript

Integrate Inference Providers into your JavaScript applications with these flexible approaches:

Our JavaScript SDK provides a convenient interface with automatic provider selection and TypeScript support.

Install with NPM:

```shell
npm install @huggingface/inference
```

Then use the client with Javascript.

By default, our system automatically selects the fastest available provider for the specified model (equivalent to the `:fastest` policy — highest throughput in tokens per second).

You can change the provider selection policy by appending a policy suffix to the model id: `:cheapest` for the most cost-efficient provider (lowest price per output token), or `:preferred` to follow your preference order in [Inference Provider settings](https://hf.co/settings/inference-providers). For example, `openai/gpt-oss-120b:cheapest`.

You can also select the provider of your choice by appending the provider name to the model id (e.g. `"openai/gpt-oss-120b:groq"`).

```js
import { InferenceClient } from "@huggingface/inference";

const client = new InferenceClient(process.env.HF_TOKEN);

const chatCompletion = await client.chatCompletion({
  model: "openai/gpt-oss-120b:fastest",
  messages: [
    {
      role: "user",
      content: "How many 'G's in 'huggingface'?",
    },
  ],
});

console.log(chatCompletion.choices[0].message);
```

If you're already using OpenAI's Javascript client, then you need a **drop-in OpenAI replacement**. Just swap-out the base URL to instantly access hundreds of additional open-weights models through our provider network.

By default, our system automatically selects the fastest available provider for the specified model (equivalent to the `:fastest` policy — highest throughput in tokens per second).

You can change the provider selection policy by appending a policy suffix to the model id: `:cheapest` for the most cost-efficient provider (lowest price per output token), or `:preferred` to follow your preference order in [Inference Provider settings](https://hf.co/settings/inference-providers). For example, `openai/gpt-oss-120b:cheapest`.

You can also select the provider of your choice by appending the provider name to the model id (e.g. `"openai/gpt-oss-120b:groq"`).

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: REDACTED,
});

const completion = await client.chat.completions.create({
  model: "openai/gpt-oss-120b:fastest",
  messages: [
    {
      role: "user",
      content: "How many 'G's in 'huggingface'?",
    },
  ],
});

console.log(completion.choices[0].message.content);
```

For lightweight applications or custom implementations, use our REST API directly with standard fetch.

By default, our system automatically selects the fastest available provider for the specified model (equivalent to the `:fastest` policy — highest throughput in tokens per second).

You can change the provider selection policy by appending a policy suffix to the model id: `:cheapest` for the most cost-efficient provider (lowest price per output token), or `:preferred` to follow your preference order in [Inference Provider settings](https://hf.co/settings/inference-providers). For example, `openai/gpt-oss-120b:cheapest`.

You can also select the provider of your choice by appending the provider name to the model id (e.g. `"openai/gpt-oss-120b:groq"`).

```js
import fetch from "node-fetch";

const response = await fetch(
  "https://router.huggingface.co/v1/chat/completions",
  {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.HF_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "openai/gpt-oss-120b:fastest",
      messages: [
        {
          role: "user",
          content: "How many 'G's in 'huggingface'?",
        },
      ],
    }),
  }
);
console.log(await response.json());
```

#### HTTP / cURL

For testing, debugging, or integrating with any HTTP client, here's the raw REST API format.

By default, our system automatically selects the fastest available provider for the specified model (equivalent to the `:fastest` policy — highest throughput in tokens per second).

You can change the provider selection policy by appending a policy suffix to the model id: `:cheapest` for the most cost-efficient provider (lowest price per output token), or `:preferred` to follow your preference order in [Inference Provider settings](https://hf.co/settings/inference-providers). For example, `openai/gpt-oss-120b:cheapest`.

You can also select the provider of your choice by appending the provider name to the model id (e.g. `"openai/gpt-oss-120b:groq"`).

```bash
curl https://router.huggingface.co/v1/chat/completions \
    -H "Authorization: Bearer $HF_TOKEN" \
    -H 'Content-Type: application/json' \
    -d '{
        "messages": [
            {
                "role": "user",
                "content": "How many G in huggingface?"
            }
        ],
        "model": "openai/gpt-oss-120b:fastest",
        "stream": false
    }'
```

### Quick Start - Text-to-Image Generation

Let's explore how to generate images from text prompts using Inference Providers. We'll use [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev), a state-of-the-art diffusion model that produces highly detailed, photorealistic images.

#### Python

Use the `huggingface_hub` library for the simplest image generation experience with automatic provider selection:

```python
import os
from huggingface_hub import InferenceClient

client = InferenceClient(api_key=os.environ["HF_TOKEN"])

image = client.text_to_image(
    prompt="A serene lake surrounded by mountains at sunset, photorealistic style",
    model="black-forest-labs/FLUX.1-dev"
)

# Save the generated image
image.save("generated_image.png")
```

#### JavaScript

Use our JavaScript SDK for streamlined image generation with TypeScript support:

```js
import { InferenceClient } from "@huggingface/inference";
import fs from "fs";

const client = new InferenceClient(process.env.HF_TOKEN);

const imageBlob = await client.textToImage({
  model: "black-forest-labs/FLUX.1-dev",
  inputs:
    "A serene lake surrounded by mountains at sunset, photorealistic style",
});

// Save the image
const buffer = Buffer.from(await imageBlob.arrayBuffer());
fs.writeFileSync("generated_image.png", buffer);
```

## Provider Selection

The Inference Providers API acts as a unified proxy layer that sits between your application and multiple AI providers. Understanding how provider selection works is crucial for optimizing performance, cost, and reliability in your applications.

### API as a Proxy Service

When using Inference Providers, your requests go through Hugging Face's proxy infrastructure, which provides several key benefits:

- **Unified Authentication & Billing**: Use a single Hugging Face token for all providers
- **Automatic Failover**: When using automatic provider selection (`provider="auto"`), requests are automatically routed to alternative providers if the primary provider is flagged as unavailable by our validation system
- **Consistent Interface through client libraries**: When using our client libraries, the same request format works across different providers

Because the API acts as a proxy, the exact HTTP request may vary between providers as each provider has their own API requirements and response formats. **When using our official client libraries** (JavaScript or Python), these provider-specific differences are handled automatically whether you use `provider="auto"` or specify a particular provider.

### Client-Side Provider Selection (Inference Clients)

When using the Hugging Face inference clients (JavaScript or Python), you can explicitly specify a provider or let the system choose automatically. The client then formats the HTTP request to match the selected provider's API requirements.

```javascript
import { InferenceClient } from "@huggingface/inference";

const client = new InferenceClient(process.env.HF_TOKEN);

// Explicit provider selection
await client.chatCompletion({
  model: "deepseek-ai/DeepSeek-R1",
  provider: "novita", // Specific provider
  messages: [{ role: "user", content: "Hello!" }],
});

// Automatic provider selection (default: "auto")
await client.chatCompletion({
  model: "deepseek-ai/DeepSeek-R1",
  // Defaults to "auto" selection of the provider
  // provider="auto",
  messages: [{ role: "user", content: "Hello!" }],
});
```

```python
import os
from huggingface_hub import InferenceClient

client = InferenceClient(token=os.environ["HF_TOKEN"])

# Explicit provider selection
result = client.chat_completion(
    model="deepseek-ai/DeepSeek-R1",
    provider="novita",  # Specific provider
    messages=[{"role": "user", "content": "Hello!"}],
)

# Automatic provider selection (default: "auto")
result = client.chat_completion(
    model="deepseek-ai/DeepSeek-R1",
    # Defaults to "auto" selection of the provider
    # provider="auto",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

**Provider Selection Policy:**

- `provider: "auto"` (default): Selects the fastest available provider for the model (highest throughput in tokens per second), equivalent to the `:fastest` policy.
- `provider: "specific-provider"`: Forces use of a specific provider (e.g., "together", "replicate", "fal-ai", ...).

### Alternative: OpenAI-Compatible Chat Completions Endpoint (Chat Only)

If you prefer to work with familiar OpenAI APIs or want to migrate existing chat completion code with minimal changes, we offer a drop-in compatible endpoint that handles all provider selection automatically on the server side.

By default, the fastest available provider is selected for the model (highest throughput in tokens per second). This is equivalent to appending `:fastest` to the model name.
You can change that policy by adding a suffix to the model name:

- `:cheapest` selects the most cost-efficient provider for the model (lowest price per output tokens)
- `:preferred` selects the first available provider sorted by your preference order in [Inference Provider settings](https://hf.co/settings/inference-providers)

**Note**: This OpenAI-compatible endpoint is currently available for chat completion tasks only. For other tasks like text-to-image, embeddings, or speech processing, use the Hugging Face inference clients shown above.

```javascript
import { OpenAI } from "openai";

const client = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: REDACTED,
});

const completion = await client.chat.completions.create({
  model: "deepseek-ai/DeepSeek-R1:fastest",
  messages: [{ role: "user", content: "Hello!" }],
});
```

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1:fastest",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(completion.choices[0].message.content)
```

This endpoint can also be requested through direct HTTP access, making it suitable for integration with various HTTP clients and applications that need to interact with the chat completion service directly.

```bash
curl https://router.huggingface.co/v1/chat/completions \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1:fastest",
    "messages": [
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'
```

**Key Features:**

- **Server-Side Provider Selection**: The server automatically selects the fastest available provider by default (`:fastest` policy)
- **Model Listing**: GET `/v1/models` returns available models across all providers, including per-provider pricing, context length, latency, and throughput when available
- **OpenAI SDK Compatibility**: Works with existing OpenAI client libraries
- **Chat Tasks Only**: Limited to conversational workloads

### Choosing the Right Approach

**Use Inference Clients when:**

- You need support for all task types (text-to-image, speech, embeddings, etc.)
- You want explicit control over provider selection
- You're building applications that use multiple AI tasks

**Use OpenAI-Compatible Endpoint when:**

- You're only doing chat completions
- You want to migrate existing OpenAI-based code with minimal changes
- You prefer server-side provider management

**Use Direct HTTP when:**

- You're implementing custom request logic
- You need fine-grained control over the request/response cycle
- You're working in environments without available client libraries

## Next Steps

Now that you understand the basics, explore these resources to make the most of Inference Providers:

- **[Announcement Blog Post](https://huggingface.co/blog/inference-providers)**: Learn more about the launch of Inference Providers
- **[Pricing and Billing](./pricing)**: Understand costs and billing of Inference Providers
- **[Hub Integration](./hub-integration)**: Learn how Inference Providers are integrated with the Hugging Face Hub
- **[Register as a Provider](./register-as-a-provider)**: Requirements to join our partner network as a provider
- **[Hub API](./hub-api)**: Advanced API features and configuration
- **[API Reference](./tasks/index)**: Complete parameter documentation for all supported tasks

### Template
https://huggingface.co/docs/inference-providers/providers/replicate.md

### Template

If you want to update the content related to replicate's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/replicate.handlebars`.

### Logos

If you want to update replicate's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `replicate-light.png` and `replicate-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Replicate

> [!TIP]
> All supported Replicate models can be found [here](https://huggingface.co/models?inference_provider=replicate&sort=trending)

    
        
        
    

    
        
        
    

Replicate is building tools so all software engineers can use AI as if it were normal software. You should be able to import an image generator the same way you import an npm package. You should be able to customize a model as easily as you can fork something on GitHub.

## Supported tasks

### Automatic Speech Recognition

Find out more about Automatic Speech Recognition [here](../tasks/automatic-speech-recognition).

<InferenceSnippet
    pipeline=automatic-speech-recognition
    providersMapping={ {"replicate":{"modelId":"openai/whisper-large-v3","providerModelId":"openai/whisper:8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e"} } }
/>

### Image To Image

Find out more about Image To Image [here](../tasks/image-to-image).

<InferenceSnippet
    pipeline=image-to-image
    providersMapping={ {"replicate":{"modelId":"black-forest-labs/FLUX.2-dev","providerModelId":"black-forest-labs/flux-2-dev"} } }
/>

### Text To Image

Find out more about Text To Image [here](../tasks/text-to-image).

<InferenceSnippet
    pipeline=text-to-image
    providersMapping={ {"replicate":{"modelId":"black-forest-labs/FLUX.1-dev","providerModelId":"black-forest-labs/flux-dev"} } }
/>

### Text To Video

Find out more about Text To Video [here](../tasks/text-to-video).

<InferenceSnippet
    pipeline=text-to-video
    providersMapping={ {"replicate":{"modelId":"Wan-AI/Wan2.2-TI2V-5B","providerModelId":"wan-video/wan-2.2-5b-fast"} } }
/>

### Template
https://huggingface.co/docs/inference-providers/providers/cohere.md

### Template

If you want to update the content related to cohere's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/cohere.handlebars`.

### Logos

If you want to update cohere's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `cohere-light.png` and `cohere-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Cohere

> [!TIP]
> All supported Cohere models can be found [here](https://huggingface.co/models?inference_provider=cohere&sort=trending)

    
        
        
    

    
        
        
    

Cohere brings you cutting-edge multilingual models, advanced retrieval, and an AI workspace tailored for the modern enterprise — all within a single, secure platform.

Cohere is the first model creator to share and serve their models directly on the Hub as an Inference Provider.

## Supported tasks

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"cohere":{"modelId":"CohereLabs/aya-expanse-32b","providerModelId":"c4ai-aya-expanse-32b"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"cohere":{"modelId":"CohereLabs/aya-vision-32b","providerModelId":"c4ai-aya-vision-32b"} } }
conversational />

### Template
https://huggingface.co/docs/inference-providers/providers/nscale.md

### Template

If you want to update the content related to nscale's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/nscale.handlebars`.

### Logos

If you want to update nscale's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `nscale-light.png` and `nscale-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Nscale

> [!TIP]
> All supported Nscale models can be found [here](https://huggingface.co/models?inference_provider=nscale&sort=trending)

    
        
        
    

    
        
        
    

Nscale is a vertically integrated AI cloud that delivers bespoke, sovereign AI infrastructure at scale.

Built on this foundation, Nscale's inference service empowers developers with a wide range of models and ready-to-use inference services that integrate into workflows without the need to manage the underlying infrastructure.

## Supported tasks

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"nscale":{"modelId":"meta-llama/Llama-3.1-8B-Instruct","providerModelId":"meta-llama/Llama-3.1-8B-Instruct"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"nscale":{"modelId":"meta-llama/Llama-4-Scout-17B-16E-Instruct","providerModelId":"meta-llama/Llama-4-Scout-17B-16E-Instruct"} } }
conversational />

### Text To Image

Find out more about Text To Image [here](../tasks/text-to-image).

<InferenceSnippet
    pipeline=text-to-image
    providersMapping={ {"nscale":{"modelId":"black-forest-labs/FLUX.1-schnell","providerModelId":"black-forest-labs/FLUX.1-schnell"} } }
/>

### Template
https://huggingface.co/docs/inference-providers/providers/cerebras.md

### Template

If you want to update the content related to cerebras's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/cerebras.handlebars`.

### Logos

If you want to update cerebras's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `cerebras-light.png` and `cerebras-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Cerebras

> [!TIP]
> All supported Cerebras models can be found [here](https://huggingface.co/models?inference_provider=cerebras&sort=trending)

    
        
        
    

    
        
        
    

Cerebras stands alone as the world’s fastest AI inference and training platform. Organizations across fields like medical research, cryptography, energy, and agentic AI use our CS-2 and CS-3 systems to build on-premise supercomputers, while developers and enterprises everywhere can access the power of Cerebras through our pay-as-you-go cloud offerings.

## Supported tasks

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"cerebras":{"modelId":"openai/gpt-oss-120b","providerModelId":"gpt-oss-120b"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"cerebras":{"modelId":"google/gemma-4-31B-it","providerModelId":"gemma-4-31b"} } }
conversational />

### Template
https://huggingface.co/docs/inference-providers/providers/fal-ai.md

### Template

If you want to update the content related to fal-ai's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/fal-ai.handlebars`.

### Logos

If you want to update fal-ai's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `fal-ai-light.png` and `fal-ai-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Fal

> [!TIP]
> All supported Fal models can be found [here](https://huggingface.co/models?inference_provider=fal-ai&sort=trending)

    
        
        
    

    
        
        
    

Founded in 2021 by [Burkay Gur](https://huggingface.co/burkaygur) and [Gorkem Yurtseven](https://huggingface.co/gorkemyurt), fal.ai was born out of a shared passion for AI and a desire to address the challenges in AI infrastructure observed during their tenures at Coinbase and Amazon.

## Supported tasks

### Automatic Speech Recognition

Find out more about Automatic Speech Recognition [here](../tasks/automatic-speech-recognition).

<InferenceSnippet
    pipeline=automatic-speech-recognition
    providersMapping={ {"fal-ai":{"modelId":"openai/whisper-large-v3","providerModelId":"fal-ai/whisper"} } }
/>

### Image Segmentation

Find out more about Image Segmentation [here](../tasks/image-segmentation).

<InferenceSnippet
    pipeline=image-segmentation
    providersMapping={ {"fal-ai":{"modelId":"briaai/RMBG-2.0","providerModelId":"fal-ai/bria/background/remove"} } }
/>

### Image To Image

Find out more about Image To Image [here](../tasks/image-to-image).

<InferenceSnippet
    pipeline=image-to-image
    providersMapping={ {"fal-ai":{"modelId":"black-forest-labs/FLUX.2-dev","providerModelId":"fal-ai/flux-2/edit"} } }
/>

### Text To Image

Find out more about Text To Image [here](../tasks/text-to-image).

<InferenceSnippet
    pipeline=text-to-image
    providersMapping={ {"fal-ai":{"modelId":"black-forest-labs/FLUX.1-dev","providerModelId":"fal-ai/flux/dev"} } }
/>

### Text To Video

Find out more about Text To Video [here](../tasks/text-to-video).

<InferenceSnippet
    pipeline=text-to-video
    providersMapping={ {"fal-ai":{"modelId":"Wan-AI/Wan2.2-TI2V-5B","providerModelId":"fal-ai/wan/v2.2-5b/text-to-video"} } }
/>

### Template
https://huggingface.co/docs/inference-providers/providers/wavespeed.md

### Template

If you want to update the content related to wavespeed's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/wavespeed.handlebars`.

### Logos

If you want to update wavespeed's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `wavespeed-light.png` and `wavespeed-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# WaveSpeed

> [!TIP]
> All supported WaveSpeed models can be found [here](https://huggingface.co/models?inference_provider=wavespeed&sort=trending)

    
        
        
    

    
        
        
    

[WaveSpeedAI](https://wavespeed.ai/) is a high-performance AI inference platform specializing in image and video generation. Built with cutting-edge infrastructure and optimization techniques, [WaveSpeedAI](https://wavespeed.ai/) provides fast, scalable, and cost-effective model serving for creative AI applications.

## Supported tasks

### Image To Image

Find out more about Image To Image [here](../tasks/image-to-image).

<InferenceSnippet
    pipeline=image-to-image
    providersMapping={ {"wavespeed":{"modelId":"black-forest-labs/FLUX.2-dev","providerModelId":"wavespeed-ai/flux-2-dev/edit"} } }
/>

### Text To Image

Find out more about Text To Image [here](../tasks/text-to-image).

<InferenceSnippet
    pipeline=text-to-image
    providersMapping={ {"wavespeed":{"modelId":"black-forest-labs/FLUX.1-dev","providerModelId":"wavespeed-ai/flux-dev"} } }
/>

### Text To Video

Find out more about Text To Video [here](../tasks/text-to-video).

<InferenceSnippet
    pipeline=text-to-video
    providersMapping={ {"wavespeed":{"modelId":"larryvrh/MiniMax-H3-Turbo-Lora","providerModelId":"wavespeed-ai/minimax-h3/text-to-video-lora"} } }
/>

### Template
https://huggingface.co/docs/inference-providers/providers/zai-org.md

### Template

If you want to update the content related to zai-org's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/zai-org.handlebars`.

### Logos

If you want to update zai-org's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `zai-org-light.png` and `zai-org-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Z.ai

> [!TIP]
> All supported Z.ai models can be found [here](https://huggingface.co/models?inference_provider=zai-org&sort=trending)

    
        
        
    

    
        
        
    

Z.ai is an AI platform that provides cutting-edge large language models powered by GLM series. Their flagship models feature Mixture-of-Experts (MoE) architecture with advanced reasoning, coding, and agentic capabilities.

For latest pricing, visit the [pricing page](https://docs.z.ai/guides/overview/pricing).

## Resources
 - **Website**: https://z.ai/
 - **Documentation**: https://docs.z.ai/
 - **API Documentation**: https://docs.z.ai/api-reference/introduction
 - **GitHub**: https://github.com/zai-org
 - **Hugging Face**: https://huggingface.co/zai-org

## Supported tasks

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"zai-org":{"modelId":"zai-org/GLM-5.2","providerModelId":"glm-5.2"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"zai-org":{"modelId":"zai-org/GLM-4.6V-Flash","providerModelId":"glm-4.6v-flash"} } }
conversational />

### Template
https://huggingface.co/docs/inference-providers/providers/novita.md

### Template

If you want to update the content related to novita's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/novita.handlebars`.

### Logos

If you want to update novita's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `novita-light.png` and `novita-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Novita

> [!TIP]
> All supported Novita models can be found [here](https://huggingface.co/models?inference_provider=novita&sort=trending)

    
        
        
    

    
        
        
    

[Novita](https://novita.ai) is the go-to inference platform for AI developers seeking a low-cost, reliable, and simple solution for shipping AI models.

Offering 200+ APIs (LLMs, image, video, audio) with fully managed deployment — enterprise-grade, scalable, and maintenance-free.

## Supported tasks

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"novita":{"modelId":"deepseek-ai/DeepSeek-V4-Flash-0731","providerModelId":"deepseek/deepseek-v4-flash-0731"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"novita":{"modelId":"google/gemma-4-31B-it","providerModelId":"google/gemma-4-31b-it"} } }
conversational />

### Template
https://huggingface.co/docs/inference-providers/providers/groq.md

### Template

If you want to update the content related to groq's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/groq.handlebars`.

### Logos

If you want to update groq's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `groq-light.png` and `groq-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Groq

> [!TIP]
> All supported Groq models can be found [here](https://huggingface.co/models?inference_provider=groq&sort=trending)

    
        
        
    

    
        
        
    

Groq is fast AI inference. Their groundbreaking LPU technology delivers record-setting performance and efficiency for GenAI models. With custom chips specifically designed for AI inference workloads and a deterministic, software-first approach, Groq eliminates the bottlenecks of conventional hardware to enable real-time AI applications with predictable latency and exceptional throughput so developers can build fast.

For latest pricing, visit our [pricing page](https://groq.com/pricing/).

## Resources
 - **Website**: https://groq.com/
 - **Documentation**: https://console.groq.com/docs
 - **Community Forum**: https://community.groq.com/
 - **X**: [@GroqInc](https://x.com/GroqInc)
 - **LinkedIn**: [Groq](https://www.linkedin.com/company/groq/)
 - **YouTube**: [Groq](https://www.youtube.com/@GroqInc)

## Supported tasks

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"groq":{"modelId":"openai/gpt-oss-20b","providerModelId":"openai/gpt-oss-20b"} } }
conversational />

### Template
https://huggingface.co/docs/inference-providers/providers/together.md

### Template

If you want to update the content related to together's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/together.handlebars`.

### Logos

If you want to update together's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `together-light.png` and `together-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Together

> [!TIP]
> All supported Together models can be found [here](https://huggingface.co/models?inference_provider=together&sort=trending)

    
        
        
    

    
        
        
    

Together decentralized cloud services empower developers and researchers at organizations of all sizes to train, fine-tune, and deploy generative AI models.

## Supported tasks

### Automatic Speech Recognition

Find out more about Automatic Speech Recognition [here](../tasks/automatic-speech-recognition).

<InferenceSnippet
    pipeline=automatic-speech-recognition
    providersMapping={ {"together":{"modelId":"openai/whisper-large-v3","providerModelId":"openai/whisper-large-v3"} } }
/>

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"together":{"modelId":"deepseek-ai/DeepSeek-V4-Flash-0731","providerModelId":"deepseek-ai/DeepSeek-V4-Flash-0731"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"together":{"modelId":"meta-models/Muse-Glimmer-30B","providerModelId":"meta-models/Muse-Glimmer-30B"} } }
conversational />

### Feature Extraction

Find out more about Feature Extraction [here](../tasks/feature-extraction).

<InferenceSnippet
    pipeline=feature-extraction
    providersMapping={ {"together":{"modelId":"intfloat/multilingual-e5-large-instruct","providerModelId":"intfloat/multilingual-e5-large-instruct"} } }
/>

### Template
https://huggingface.co/docs/inference-providers/providers/hf-inference.md

### Template

If you want to update the content related to hf-inference's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/hf-inference.handlebars`.

### Logos

If you want to update hf-inference's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `hf-inference-light.png` and `hf-inference-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# HF Inference

> [!TIP]
> All supported HF Inference models can be found [here](https://huggingface.co/models?inference_provider=hf-inference&sort=trending)

    
        
        
    

    
        
        
    

HF Inference is the serverless Inference API powered by Hugging Face. This service used to be called "Inference API (serverless)" prior to Inference Providers.
If you are interested in deploying models to a dedicated and autoscaling infrastructure managed by Hugging Face, check out [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) instead.

As of July 2025, hf-inference focuses mostly on CPU inference (e.g. embedding, text-ranking, text-classification, or smaller LLMs that have historical importance like BERT or GPT-2).

## Supported tasks

### Automatic Speech Recognition

Find out more about Automatic Speech Recognition [here](../tasks/automatic-speech-recognition).

<InferenceSnippet
    pipeline=automatic-speech-recognition
    providersMapping={ {"hf-inference":{"modelId":"openai/whisper-large-v3","providerModelId":"openai/whisper-large-v3"} } }
/>

### Feature Extraction

Find out more about Feature Extraction [here](../tasks/feature-extraction).

<InferenceSnippet
    pipeline=feature-extraction
    providersMapping={ {"hf-inference":{"modelId":"microsoft/harrier-oss-v1-0.6b","providerModelId":"microsoft/harrier-oss-v1-0.6b"} } }
/>

### Fill Mask

Find out more about Fill Mask [here](../tasks/fill-mask).

<InferenceSnippet
    pipeline=fill-mask
    providersMapping={ {"hf-inference":{"modelId":"google-bert/bert-base-uncased","providerModelId":"google-bert/bert-base-uncased"} } }
/>

### Image Classification

Find out more about Image Classification [here](../tasks/image-classification).

<InferenceSnippet
    pipeline=image-classification
    providersMapping={ {"hf-inference":{"modelId":"Falconsai/nsfw_image_detection","providerModelId":"Falconsai/nsfw_image_detection"} } }
/>

### Image Segmentation

Find out more about Image Segmentation [here](../tasks/image-segmentation).

<InferenceSnippet
    pipeline=image-segmentation
    providersMapping={ {"hf-inference":{"modelId":"nvidia/segformer-b0-finetuned-ade-512-512","providerModelId":"nvidia/segformer-b0-finetuned-ade-512-512"} } }
/>

### Object Detection

Find out more about Object Detection [here](../tasks/object-detection).

<InferenceSnippet
    pipeline=object-detection
    providersMapping={ {"hf-inference":{"modelId":"facebook/detr-resnet-50","providerModelId":"facebook/detr-resnet-50"} } }
/>

### Question Answering

Find out more about Question Answering [here](../tasks/question-answering).

<InferenceSnippet
    pipeline=question-answering
    providersMapping={ {"hf-inference":{"modelId":"deepset/roberta-base-squad2","providerModelId":"deepset/roberta-base-squad2"} } }
/>

### Summarization

Find out more about Summarization [here](../tasks/summarization).

<InferenceSnippet
    pipeline=summarization
    providersMapping={ {"hf-inference":{"modelId":"human-centered-summarization/financial-summarization-pegasus","providerModelId":"human-centered-summarization/financial-summarization-pegasus"} } }
/>

### Table Question Answering

Find out more about Table Question Answering [here](../tasks/table-question-answering).

<InferenceSnippet
    pipeline=table-question-answering
    providersMapping={ {"hf-inference":{"modelId":"google/tapas-base-finetuned-wtq","providerModelId":"google/tapas-base-finetuned-wtq"} } }
/>

### Text Classification

Find out more about Text Classification [here](../tasks/text-classification).

<InferenceSnippet
    pipeline=text-classification
    providersMapping={ {"hf-inference":{"modelId":"BAAI/bge-reranker-v2-m3","providerModelId":"BAAI/bge-reranker-v2-m3"} } }
/>

### Text To Image

Find out more about Text To Image [here](../tasks/text-to-image).

<InferenceSnippet
    pipeline=text-to-image
    providersMapping={ {"hf-inference":{"modelId":"stabilityai/stable-diffusion-3-medium-diffusers","providerModelId":"stabilityai/stable-diffusion-3-medium-diffusers"} } }
/>

### Token Classification

Find out more about Token Classification [here](../tasks/token-classification).

<InferenceSnippet
    pipeline=token-classification
    providersMapping={ {"hf-inference":{"modelId":"OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1","providerModelId":"OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1"} } }
/>

### Translation

Find out more about Translation [here](../tasks/translation).

<InferenceSnippet
    pipeline=translation
    providersMapping={ {"hf-inference":{"modelId":"google-t5/t5-small","providerModelId":"google-t5/t5-small"} } }
/>

### Zero Shot Classification

Find out more about Zero Shot Classification [here](../tasks/zero-shot-classification).

<InferenceSnippet
    pipeline=zero-shot-classification
    providersMapping={ {"hf-inference":{"modelId":"facebook/bart-large-mnli","providerModelId":"facebook/bart-large-mnli"} } }
/>

### Template
https://huggingface.co/docs/inference-providers/providers/scaleway.md

### Template

If you want to update the content related to scaleway's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/scaleway.handlebars`.

### Logos

If you want to update scaleway's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `scaleway-light.png` and `scaleway-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Scaleway

> [!TIP]
> All supported Scaleway models can be found [here](https://huggingface.co/models?inference_provider=scaleway&sort=trending)

    
        
        
    

    
        
        
    

Scaleway is a European cloud provider, serving latest LLM models through its [Generative APIs](https://www.scaleway.com/en/generative-apis/) alongside a complete cloud ecosystem.

## Supported tasks

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"scaleway":{"modelId":"zai-org/GLM-5.2","providerModelId":"glm-5.2"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"scaleway":{"modelId":"Qwen/Qwen3.6-35B-A3B","providerModelId":"qwen3.6-35b-a3b"} } }
conversational />

### Feature Extraction

Find out more about Feature Extraction [here](../tasks/feature-extraction).

<InferenceSnippet
    pipeline=feature-extraction
    providersMapping={ {"scaleway":{"modelId":"Qwen/Qwen3-Embedding-8B","providerModelId":"qwen3-embedding-8b"} } }
/>

### Template
https://huggingface.co/docs/inference-providers/providers/fireworks-ai.md

### Template

If you want to update the content related to fireworks-ai's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/fireworks-ai.handlebars`.

### Logos

If you want to update fireworks-ai's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `fireworks-ai-light.png` and `fireworks-ai-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Fireworks AI

> [!TIP]
> All supported Fireworks AI models can be found [here](https://huggingface.co/models?inference_provider=fireworks-ai&sort=trending)

    
        
        
    

    
        
        
    

Fireworks AI is a developer-centric platform that delivers high-performance generative AI solutions, enabling efficient deployment and fine-tuning of large language models (LLMs) and image models.
## Supported tasks

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"fireworks-ai":{"modelId":"deepseek-ai/DeepSeek-V4-Flash-0731","providerModelId":"accounts/fireworks/models/deepseek-v4-flash-0731"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"fireworks-ai":{"modelId":"moonshotai/Kimi-K3","providerModelId":"accounts/fireworks/models/kimi-k3"} } }
conversational />

### Template
https://huggingface.co/docs/inference-providers/providers/publicai.md

### Template

If you want to update the content related to publicai's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/publicai.handlebars`.

### Logos

If you want to update publicai's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `publicai-light.png` and `publicai-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# PublicAI

> [!TIP]
> All models supported by the PublicAI Inference Utility can be found [here](https://huggingface.co/models?inference_provider=publicai&sort=trending)

    
        
        
    

    
        
        
    

The Public AI Inference Utility is a nonprofit, open-source project. Their team builds products and organizes advocacy to support the work of public AI model builders like the Swiss AI Initiative, AI Singapore, AI Sweden, and the Barcelona Supercomputing Center.

They believe in public AI — AI as public infrastructure like highways, water, or electricity. Think of a BBC for AI, a public utility for AI, or public libraries for AI.

## Supported tasks

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"publicai":{"modelId":"speakleash/Bielik-11B-v3.0-Instruct","providerModelId":"speakleash/Bielik-11B-v3.0-Instruct"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"publicai":{"modelId":"swiss-ai/Apertus-v1.5-70B","providerModelId":"swiss-ai/apertus-v1.5-70b"} } }
conversational />

### Template
https://huggingface.co/docs/inference-providers/providers/deepinfra.md

### Template

If you want to update the content related to deepinfra's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/deepinfra.handlebars`.

### Logos

If you want to update deepinfra's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `deepinfra-light.png` and `deepinfra-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# DeepInfra

> [!TIP]
> All supported DeepInfra models can be found [here](https://huggingface.co/models?inference_provider=deepinfra&sort=trending)

    
        
        
    

    
        
        
    

DeepInfra is a serverless AI inference platform offering one of the most cost-effective pricing per token in the industry. With a catalog of over 100 models spanning LLMs, text-to-image, text-to-speech, speech-to-text, video generation, OCR, and more, DeepInfra makes it easy for developers to integrate a wide range of AI capabilities into their applications with minimal setup.

## Resources
- **Website**: https://deepinfra.com/
- **Documentation**: https://docs.deepinfra.com

## Supported tasks

### Automatic Speech Recognition

Find out more about Automatic Speech Recognition [here](../tasks/automatic-speech-recognition).

<InferenceSnippet
    pipeline=automatic-speech-recognition
    providersMapping={ {"deepinfra":{"modelId":"openai/whisper-large-v3","providerModelId":"openai/whisper-large-v3"} } }
/>

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"deepinfra":{"modelId":"deepseek-ai/DeepSeek-V4-Flash-0731","providerModelId":"deepseek-ai/DeepSeek-V4-Flash-0731"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"deepinfra":{"modelId":"moonshotai/Kimi-K3","providerModelId":"moonshotai/Kimi-K3"} } }
conversational />

### Feature Extraction

Find out more about Feature Extraction [here](../tasks/feature-extraction).

<InferenceSnippet
    pipeline=feature-extraction
    providersMapping={ {"deepinfra":{"modelId":"Qwen/Qwen3-Embedding-0.6B","providerModelId":"Qwen/Qwen3-Embedding-0.6B"} } }
/>

### Template
https://huggingface.co/docs/inference-providers/providers/ovhcloud.md

### Template

If you want to update the content related to ovhcloud's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/ovhcloud.handlebars`.

### Logos

If you want to update ovhcloud's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `ovhcloud-light.png` and `ovhcloud-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# OVHcloud AI Endpoints

> [!TIP]
> All supported OVHcloud AI Endpoints models can be found [here](https://huggingface.co/models?inference_provider=ovhcloud&sort=trending)

    
        
        
    

    
        
        
    

[OVHcloud AI Endpoints](https://www.ovhcloud.com/en/public-cloud/ai-endpoints/) is a managed AI inference service that provides access to a wide range of state-of-the-art machine learning models. As part of OVHcloud's Public Cloud offering, AI Endpoints enables developers to easily integrate AI capabilities into their applications with data sovereignty, privacy, and GDPR compliance.

## Supported tasks

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"ovhcloud":{"modelId":"openai/gpt-oss-20b","providerModelId":"gpt-oss-20b"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"ovhcloud":{"modelId":"Qwen/Qwen3.6-27B","providerModelId":"Qwen3.6-27B"} } }
conversational />

### Template
https://huggingface.co/docs/inference-providers/providers/baseten.md

### Template

If you want to update the content related to baseten's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/baseten.handlebars`.

### Logos

If you want to update baseten's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `baseten-light.png` and `baseten-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Baseten

> [!TIP]
> All supported Baseten models can be found [here](https://huggingface.co/models?inference_provider=baseten&sort=trending)

    
        
        
    

    
        
        
    

Baseten provides on-demand frontier model APIs designed for production applications, not just experimentation. Built on the Baseten Inference Stack, these APIs deliver enterprise-grade performance and reliability with optimized inference for leading open-source models.

## Supported tasks

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"baseten":{"modelId":"deepseek-ai/DeepSeek-V4-Flash-0731","providerModelId":"deepseek-ai/DeepSeek-V4-Flash-0731"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"baseten":{"modelId":"moonshotai/Kimi-K3","providerModelId":"moonshotai/Kimi-K3"} } }
conversational />

### Template
https://huggingface.co/docs/inference-providers/providers/featherless-ai.md

### Template

If you want to update the content related to featherless-ai's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/featherless-ai.handlebars`.

### Logos

If you want to update featherless-ai's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `featherless-ai-light.png` and `featherless-ai-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Featherless AI

> [!TIP]
> All supported Featherless AI models can be found [here](https://huggingface.co/models?inference_provider=featherless-ai&sort=trending)

    
        
        
    

    
        
        
    

[Featherless AI](https://featherless.ai) is a serverless AI inference platform that offers access to thousands of open-source models. 

Our goal is to make all AI models available for serverless inference. We provide inference via API to a continually expanding library of open-weight models.

## Supported tasks

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"featherless-ai":{"modelId":"zai-org/GLM-5.2","providerModelId":"zai-org/GLM-5.2"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"featherless-ai":{"modelId":"Qwen/Qwen3.8-27B","providerModelId":"Qwen/Qwen3.8-27B"} } }
conversational />

### Text Generation

Find out more about Text Generation [here](../tasks/text-generation).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"featherless-ai":{"modelId":"zai-org/GLM-5.2","providerModelId":"zai-org/GLM-5.2"} } }
/>

### Function Calling with Inference Providers
https://huggingface.co/docs/inference-providers/guides/function-calling.md
