# Automating Code Review with GitHub Actions

[OpenCode](https://opencode.ai) is an AI coding agent that runs in your terminal and can work with open models via Hugging Face Inference Providers.

You can also install OpenCode as a GitHub App to help automate GitHub workflows!

In less than 5 minutes, you can set it up to respond to issues and pull requests using powerful open source language models.

This guide shows you how to use Hugging Face Inference Providers with OpenCode to power GitHub automation. You'll be able to triage issues, implement features, and review code using models like DeepSeek, GLM-4.5, and Kimi K2.

> [!TIP]
> This guide assumes you have a Hugging Face account and a GitHub repository. You can create a free Hugging Face account at [huggingface.co](https://huggingface.co).

## What the OpenCode GitHub App Does

Once set up, OpenCode will respond to `/oc` or `/opencode` commands in GitHub issues and pull requests:

- **Triage and explain issues** - Ask OpenCode to analyze an issue and explain it
- **Implement features** - Request a fix or feature, and OpenCode creates a PR with the changes
- **Review and modify PRs** - Request changes on pull requests and OpenCode commits them

The entire workflow runs securely in your GitHub Actions runners, and you maintain full control over which repositories you install the OpenCode GitHub app on.

## Step 1: Install OpenCode

First, install OpenCode on your local machine following the [installation instructions](https://opencode.ai/docs/#install).

## Step 2: Configure OpenCode for GitHub

Navigate to your local clone of your GitHub repository and run the setup command:

```bash
cd your-repository
opencode github install
```

This triggers an interactive setup process that will guide you through:

1. **Installing the GitHub app** - Your browser opens to authorize OpenCode for your repositories. You can choose specific repositories or all repositories.
2. **Selecting a provider** - Choose **Hugging Face** from the list
3. **Choosing a model** - Select a model like **GLM-4.5-Air** or **Kimi-K2-Instruct**
4. **Creating the workflow** - Generates `.github/workflows/opencode.yml` automatically

Once the `.github/workflows/opencode.yml` file is created, you'll need to commit and push it to your repository:

```bash
git add .github/workflows/opencode.yml
git commit -m "Add OpenCode workflow"
git push
```

After the setup completes, you'll need to add your Hugging Face token as a repository secret:

1. Get a token from your [Hugging Face settings](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained). The token needs permissions to `Make calls to Inference Providers`.
2. In your GitHub repository, go to **Settings → Secrets and variables → Actions**
3. Click **New repository secret**
4. Name it `HF_TOKEN` and paste your token

> [!TIP]
> Make sure your Hugging Face token has Inference Providers permissions enabled. We recommend creating a separate token specifically for OpenCode usage.

## Step 3: Try It Out

Once your workflow is set up and your token is configured, test it by commenting on any issue:

```
/oc summarize
```

OpenCode will analyze the issue and provide a summary. Here are other commands you can try:

**Explain an issue:**

```
/opencode explain this issue
```

**Implement a fix:**

OpenCode creates a new branch, implements the changes, and opens a pull request.

**Request changes on a PR:**

```
/oc please add error handling
```

> [!TIP]
> Use `/oc` (short) or `/opencode` (long) to trigger commands. OpenCode understands natural language, so feel free to be specific about what you need.

## Security Considerations

> [!WARNING]
> On public repositories, anyone can trigger the bot by commenting `/oc` or `/opencode`. This could lead to unexpected inference costs or unwanted PRs. Consider:
>
> - **Using a separate Hugging Face token** specifically for OpenCode (not your main token), so you can revoke it if needed without affecting other services
> - Monitoring your Hugging Face usage to track costs
> - Adding workflow conditions to restrict who can trigger the bot (e.g., only repository collaborators)
> - Starting with a private repository or test repo to control access

## Example Workflow in Action

Here's a real example from a [fork of the Hugging Face datasets repository](https://github.com/davanstrien/datasets). The issue requests adding `uv` installation support:

When someone comments `/oc fix this`, OpenCode analyzes the issue, creates a new branch, implements the changes, and opens a pull request:

The PR includes all the necessary changes:

## Switching Models

Different models offer different trade-offs between speed, cost, and capability. You can experiment with various models by editing the workflow file.

Edit `.github/workflows/opencode.yml` and update the `model` parameter:

```yaml
with:
  model: huggingface/deepseek-ai/DeepSeek-V3         # Powerful reasoning
  # or
  model: huggingface/zai-org/GLM-4.5-Air             # Balanced performance
```

Commit and push the changes:

```bash
git add .github/workflows/opencode.yml
git commit -m "Switch to the DeepSeek model"
git push
```

## Understanding the Workflow

The setup wizard generates `.github/workflows/opencode.yml`, which defines when and how OpenCode runs:

```yaml
name: opencode

on:
  issue_comment:
    types: [created] # Trigger on new comments

jobs:
  opencode:
    # Only run if comment contains /oc or /opencode
    if: |
      contains(github.event.comment.body, ' /oc') ||
      startsWith(github.event.comment.body, '/oc') ||
      contains(github.event.comment.body, ' /opencode') ||
      startsWith(github.event.comment.body, '/opencode')

    runs-on: ubuntu-latest

    permissions:
      id-token: write # Required for OpenCode authentication
      contents: read # Read repository contents
      pull-requests: read # Access PR information
      issues: read # Access issue information

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Run opencode
        uses: sst/opencode/github@latest
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }} # Your Hugging Face token
        with:
          model: huggingface/zai-org/GLM-4.5-Air # The model to use
```

The workflow triggers whenever someone comments on an issue or PR with `/oc` or `/opencode`, checks out your repository code, and runs OpenCode with your chosen model from Hugging Face Inference Providers.

## Next Steps

- Explore the [OpenCode GitHub documentation](https://opencode.ai/docs/github/) for advanced configuration
- Browse [models available through Inference Providers](https://huggingface.co/models?pipeline_tag=text-generation&inference_provider=all&sort=trending) to find the best model for your needs
- Try OpenCode in a private repository first to test with controlled access

### Image to Image
https://huggingface.co/docs/inference-providers/tasks/image-to-image.md

## Image to Image

Image-to-image is the task of transforming a source image to match the characteristics of a target image or a target image domain.

Example applications:
* Transferring the style of an image to another image
* Colorizing a black and white image
* Increasing the resolution of an image

> [!TIP]
> For more details about the `image-to-image` task, check out its [dedicated page](https://huggingface.co/tasks/image-to-image)! You will find examples and related materials.

### Recommended models

- [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev): Powerful image editing model.
- [kontext-community/relighting-kontext-dev-lora-v3](https://huggingface.co/kontext-community/relighting-kontext-dev-lora-v3): Image re-lighting model.

Explore all available models and find the one that suits you best [here](https://huggingface.co/models?inference=warm&pipeline_tag=image-to-image&sort=trending), or from the terminal with the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-models-list):

```bash
hf models ls --warm --pipeline-tag image-to-image --sort trending_score
```

### Using the API

<InferenceSnippet
    pipeline=image-to-image
    providersMapping={ {"fal-ai":{"modelId":"black-forest-labs/FLUX.2-klein-9B","providerModelId":"fal-ai/flux-2/klein/9b/edit"},"replicate":{"modelId":"black-forest-labs/FLUX.2-klein-9B","providerModelId":"black-forest-labs/flux-2-klein-9b"},"wavespeed":{"modelId":"black-forest-labs/FLUX.2-dev","providerModelId":"wavespeed-ai/flux-2-dev/edit"}} }
/>

### API specification

#### Request

| Headers |   |    |
| :--- | :--- | :--- |
| **authorization** | _string_ | Authentication header in the form `'Bearer: hf_****'` when `hf_****` is a personal user access token with "Inference Providers" permission. You can generate one from [your settings page](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained). |

| Payload |  |  |
| :--- | :--- | :--- |
| **inputs*** | _string_ | The input image data as a base64-encoded string. If no `parameters` are provided, you can also provide the image data as a raw bytes payload. |
| **parameters** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;prompt** | _string_ | The text prompt to guide the image generation. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;guidance_scale** | _number_ | For diffusion models. A higher guidance scale value encourages the model to generate images closely linked to the text prompt at the expense of lower image quality. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;negative_prompt** | _string_ | One prompt to guide what NOT to include in image generation. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;num_inference_steps** | _integer_ | For diffusion models. The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_size** | _object_ | The size in pixels of the output image. This parameter is only supported by some providers and for specific models. It will be ignored when unsupported. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;width*** | _integer_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;height*** | _integer_ |  |

#### Response

| Body |  |
| :--- | :--- | :--- |
| **image** | _unknown_ | The output image returned as raw bytes in the payload. |

### Token Classification
https://huggingface.co/docs/inference-providers/tasks/token-classification.md

## Token Classification

Token classification is a task in which a label is assigned to some tokens in a text. Some popular token classification subtasks are Named Entity Recognition (NER) and Part-of-Speech (PoS) tagging.

> [!TIP]
> For more details about the `token-classification` task, check out its [dedicated page](https://huggingface.co/tasks/token-classification)! You will find examples and related materials.

### Recommended models

- [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER): A robust performance model to identify people, locations, organizations and names of miscellaneous entities.
- [FacebookAI/xlm-roberta-large-finetuned-conll03-english](https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll03-english): A strong model to identify people, locations, organizations and names in multiple languages.
- [blaze999/Medical-NER](https://huggingface.co/blaze999/Medical-NER): A token classification model specialized on medical entity recognition.

Explore all available models and find the one that suits you best [here](https://huggingface.co/models?inference=warm&pipeline_tag=token-classification&sort=trending), or from the terminal with the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-models-list):

```bash
hf models ls --warm --pipeline-tag token-classification --sort trending_score
```

### Using the API

<InferenceSnippet
    pipeline=token-classification
    providersMapping={ {"hf-inference":{"modelId":"rizzoaiacademy/rizzo-pii-0.3B","providerModelId":"rizzoaiacademy/rizzo-pii-0.3B"}} }
/>

### API specification

#### Request

| Headers |   |    |
| :--- | :--- | :--- |
| **authorization** | _string_ | Authentication header in the form `'Bearer: hf_****'` when `hf_****` is a personal user access token with "Inference Providers" permission. You can generate one from [your settings page](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained). |

| Payload |  |  |
| :--- | :--- | :--- |
| **inputs*** | _string_ | The input text data |
| **parameters** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ignore_labels** | _string[]_ | A list of labels to ignore |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;stride** | _integer_ | The number of overlapping tokens between chunks when splitting the input text. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;aggregation_strategy** | _string_ | One of the following: |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#1)** | _&#x27;none&#x27;_ | Do not aggregate tokens |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#2)** | _&#x27;simple&#x27;_ | Group consecutive tokens with the same label in a single entity. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#3)** | _&#x27;first&#x27;_ | Similar to "simple", also preserves word integrity (use the label predicted for the first token in a word). |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#4)** | _&#x27;average&#x27;_ | Similar to "simple", also preserves word integrity (uses the label with the highest score, averaged across the word's tokens). |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#5)** | _&#x27;max&#x27;_ | Similar to "simple", also preserves word integrity (uses the label with the highest score across the word's tokens). |

#### Response

| Body |  |
| :--- | :--- | :--- |
| **(array)** | _object[]_ | Output is an array of objects. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;entity_group** | _string_ | The predicted label for a group of one or more tokens |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;entity** | _string_ | The predicted label for a single token |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;score** | _number_ | The associated score / probability |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;word** | _string_ | The corresponding text |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;start** | _integer_ | The character position in the input where this group begins. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;end** | _integer_ | The character position in the input where this group ends. |

### Image Segmentation
https://huggingface.co/docs/inference-providers/tasks/image-segmentation.md

## Image Segmentation

Image Segmentation divides an image into segments where each pixel in the image is mapped to an object.

> [!TIP]
> For more details about the `image-segmentation` task, check out its [dedicated page](https://huggingface.co/tasks/image-segmentation)! You will find examples and related materials.

### Recommended models

- [facebook/mask2former-swin-large-coco-panoptic](https://huggingface.co/facebook/mask2former-swin-large-coco-panoptic): Panoptic segmentation model trained on the COCO (common objects) dataset.

Explore all available models and find the one that suits you best [here](https://huggingface.co/models?inference=warm&pipeline_tag=image-segmentation&sort=trending), or from the terminal with the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-models-list):

```bash
hf models ls --warm --pipeline-tag image-segmentation --sort trending_score
```

### Using the API

<InferenceSnippet
    pipeline=image-segmentation
    providersMapping={ {"fal-ai":{"modelId":"briaai/RMBG-2.0","providerModelId":"fal-ai/bria/background/remove"},"hf-inference":{"modelId":"mattmdjaga/segformer_b2_clothes","providerModelId":"mattmdjaga/segformer_b2_clothes"}} }
/>

### API specification

#### Request

| Headers |   |    |
| :--- | :--- | :--- |
| **authorization** | _string_ | Authentication header in the form `'Bearer: hf_****'` when `hf_****` is a personal user access token with "Inference Providers" permission. You can generate one from [your settings page](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained). |

| Payload |  |  |
| :--- | :--- | :--- |
| **inputs*** | _string_ | The input image data as a base64-encoded string. If no `parameters` are provided, you can also provide the image data as a raw bytes payload. |
| **parameters** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;mask_threshold** | _number_ | Threshold to use when turning the predicted masks into binary values. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;overlap_mask_area_threshold** | _number_ | Mask overlap threshold to eliminate small, disconnected segments. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;subtask** | _enum_ | Possible values: instance, panoptic, semantic. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;threshold** | _number_ | Probability threshold to filter out predicted masks. |

#### Response

| Body |  |
| :--- | :--- | :--- |
| **(array)** | _object[]_ | A predicted mask / segment |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;label** | _string_ | The label of the predicted segment. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;mask** | _string_ | The corresponding mask as a black-and-white image (base64-encoded). |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;score** | _number_ | The score or confidence degree the model has. |

### Text Classification
https://huggingface.co/docs/inference-providers/tasks/text-classification.md

## Text Classification

Text Classification is the task of assigning a label or class to a given text. Some use cases are sentiment analysis, natural language inference, and assessing grammatical correctness.

> [!TIP]
> For more details about the `text-classification` task, check out its [dedicated page](https://huggingface.co/tasks/text-classification)! You will find examples and related materials.

### Recommended models

- [distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english): A robust model trained for sentiment analysis.
- [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert): A sentiment analysis model specialized in financial sentiment.
- [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest): A sentiment analysis model specialized in analyzing tweets.
- [papluca/xlm-roberta-base-language-detection](https://huggingface.co/papluca/xlm-roberta-base-language-detection): A model that can classify languages.
- [meta-llama/Prompt-Guard-86M](https://huggingface.co/meta-llama/Prompt-Guard-86M): A model that can classify text generation attacks.

Explore all available models and find the one that suits you best [here](https://huggingface.co/models?inference=warm&pipeline_tag=text-classification&sort=trending), or from the terminal with the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-models-list):

```bash
hf models ls --warm --pipeline-tag text-classification --sort trending_score
```

### Using the API

<InferenceSnippet
    pipeline=text-classification
    providersMapping={ {"hf-inference":{"modelId":"BAAI/bge-reranker-v2-m3","providerModelId":"BAAI/bge-reranker-v2-m3"}} }
/>

### API specification

#### Request

| Headers |   |    |
| :--- | :--- | :--- |
| **authorization** | _string_ | Authentication header in the form `'Bearer: hf_****'` when `hf_****` is a personal user access token with "Inference Providers" permission. You can generate one from [your settings page](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained). |

| Payload |  |  |
| :--- | :--- | :--- |
| **inputs*** | _string_ | The text to classify |
| **parameters** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;function_to_apply** | _enum_ | Possible values: sigmoid, softmax, none. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;top_k** | _integer_ | When specified, limits the output to the top K most probable classes. |

#### Response

| Body |  |
| :--- | :--- | :--- |
| **(array)** | _object[]_ | Output is an array of objects. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;label** | _string_ | The predicted class label. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;score** | _number_ | The corresponding probability. |

### Image Classification
https://huggingface.co/docs/inference-providers/tasks/image-classification.md

## Image Classification

Image classification is the task of assigning a label or class to an entire image. Images are expected to have only one class for each image.

> [!TIP]
> For more details about the `image-classification` task, check out its [dedicated page](https://huggingface.co/tasks/image-classification)! You will find examples and related materials.

### Recommended models

- [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224): A strong image classification model.
- [facebook/deit-base-distilled-patch16-224](https://huggingface.co/facebook/deit-base-distilled-patch16-224): A robust image classification model.
- [facebook/convnext-large-224](https://huggingface.co/facebook/convnext-large-224): A strong image classification model.

Explore all available models and find the one that suits you best [here](https://huggingface.co/models?inference=warm&pipeline_tag=image-classification&sort=trending), or from the terminal with the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-models-list):

```bash
hf models ls --warm --pipeline-tag image-classification --sort trending_score
```

### Using the API

<InferenceSnippet
    pipeline=image-classification
    providersMapping={ {"hf-inference":{"modelId":"Falconsai/nsfw_image_detection","providerModelId":"Falconsai/nsfw_image_detection"}} }
/>

### API specification

#### Request

| Headers |   |    |
| :--- | :--- | :--- |
| **authorization** | _string_ | Authentication header in the form `'Bearer: hf_****'` when `hf_****` is a personal user access token with "Inference Providers" permission. You can generate one from [your settings page](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained). |

| Payload |  |  |
| :--- | :--- | :--- |
| **inputs*** | _string_ | The input image data as a base64-encoded string. If no `parameters` are provided, you can also provide the image data as a raw bytes payload. |
| **parameters** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;function_to_apply** | _enum_ | Possible values: sigmoid, softmax, none. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;top_k** | _integer_ | When specified, limits the output to the top K most probable classes. |

#### Response

| Body |  |
| :--- | :--- | :--- |
| **(array)** | _object[]_ | Output is an array of objects. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;label** | _string_ | The predicted class label. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;score** | _number_ | The corresponding probability. |

### Chat Completion
https://huggingface.co/docs/inference-providers/tasks/chat-completion.md

## Chat Completion

Generate a response given a list of messages in a conversational context, supporting both conversational Language Models (LLMs) and conversational Vision-Language Models (VLMs).
This is a subtask of [`text-generation`](https://huggingface.co/docs/inference-providers/tasks/text-generation) and [`image-text-to-text`](https://huggingface.co/docs/inference-providers/tasks/image-text-to-text).

### Recommended models

#### Conversational Large Language Models (LLMs)

- [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it): A text-generation model trained to follow instructions.
- [Qwen/Qwen3-Coder-480B-A35B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct): Powerful text generation model for coding.
- [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b): Great text generation model with top-notch tool calling capabilities.
- [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507): A powerful small model with reasoning capabilities.
- [Qwen/Qwen2.5-7B-Instruct-1M](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M): Strong conversational model that supports very long instructions.
- [Qwen/Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct): Text generation model used to write code.
- [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1): Powerful reasoning based open large language model.

#### Conversational Vision-Language Models (VLMs)

- [zai-org/GLM-4.5V](https://huggingface.co/zai-org/GLM-4.5V): Cutting-edge reasoning vision language model.
- [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct): Small yet powerful model.

Explore all available models and find the one that suits you best [here](https://huggingface.co/models?inference=warm&pipeline_tag=image-text-to-text&sort=trending), or from the terminal with the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-models-list):

```bash
hf models ls --warm --pipeline-tag image-text-to-text --sort trending_score
```

### API Playground

For Chat Completion models, we provide an interactive UI Playground for easier testing:

- Quickly iterate on your prompts from the UI.
- Set and override system, assistant and user messages.
- Browse and select models currently available on the Inference API.
- Compare the output of two models side-by-side.
- Adjust requests parameters from the UI.
- Easily switch between UI view and code snippets.

Access the Inference UI Playground and start exploring: [https://huggingface.co/playground](https://huggingface.co/playground)

### Using the API

The API supports:

* Using the chat completion API compatible with the OpenAI SDK.
* Using grammars, constraints, and tools.
* Streaming the output

#### Code snippet example for conversational LLMs

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"baseten":{"modelId":"zai-org/GLM-5.2","providerModelId":"zai-org/GLM-5.2"},"cerebras":{"modelId":"openai/gpt-oss-120b","providerModelId":"gpt-oss-120b"},"cohere":{"modelId":"CohereLabs/aya-expanse-32b","providerModelId":"c4ai-aya-expanse-32b"},"deepinfra":{"modelId":"deepseek-ai/DeepSeek-V4-Flash-0731","providerModelId":"deepseek-ai/DeepSeek-V4-Flash-0731"},"featherless-ai":{"modelId":"poolside/Laguna-S-2.1","providerModelId":"poolside/Laguna-S-2.1"},"fireworks-ai":{"modelId":"deepseek-ai/DeepSeek-V4-Flash-0731","providerModelId":"accounts/fireworks/models/deepseek-v4-flash-0731"},"groq":{"modelId":"openai/gpt-oss-120b","providerModelId":"openai/gpt-oss-120b"},"novita":{"modelId":"deepseek-ai/DeepSeek-V4-Flash-0731","providerModelId":"deepseek/deepseek-v4-flash-0731"},"nscale":{"modelId":"meta-llama/Llama-3.1-8B-Instruct","providerModelId":"meta-llama/Llama-3.1-8B-Instruct"},"ovhcloud":{"modelId":"openai/gpt-oss-120b","providerModelId":"gpt-oss-120b"},"publicai":{"modelId":"speakleash/Bielik-11B-v3.0-Instruct","providerModelId":"speakleash/Bielik-11B-v3.0-Instruct"},"scaleway":{"modelId":"zai-org/GLM-5.2","providerModelId":"glm-5.2"},"together":{"modelId":"deepseek-ai/DeepSeek-V4-Flash-0731","providerModelId":"deepseek-ai/DeepSeek-V4-Flash-0731"},"zai-org":{"modelId":"zai-org/GLM-5.2","providerModelId":"glm-5.2"}} }
conversational />

#### Code snippet example for conversational VLMs

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"baseten":{"modelId":"moonshotai/Kimi-K3","providerModelId":"moonshotai/Kimi-K3"},"cerebras":{"modelId":"google/gemma-4-31B-it","providerModelId":"gemma-4-31b"},"cohere":{"modelId":"CohereLabs/command-a-plus-05-2026-w4a4","providerModelId":"command-a-plus-05-2026"},"deepinfra":{"modelId":"moonshotai/Kimi-K3","providerModelId":"moonshotai/Kimi-K3"},"featherless-ai":{"modelId":"moonshotai/Kimi-K3","providerModelId":"moonshotai/Kimi-K3"},"fireworks-ai":{"modelId":"moonshotai/Kimi-K3","providerModelId":"accounts/fireworks/models/kimi-k3"},"novita":{"modelId":"google/gemma-4-31B-it","providerModelId":"google/gemma-4-31b-it"},"nscale":{"modelId":"meta-llama/Llama-4-Scout-17B-16E-Instruct","providerModelId":"meta-llama/Llama-4-Scout-17B-16E-Instruct"},"ovhcloud":{"modelId":"Qwen/Qwen3.6-27B","providerModelId":"Qwen3.6-27B"},"publicai":{"modelId":"swiss-ai/Apertus-v1.5-70B","providerModelId":"swiss-ai/apertus-v1.5-70b"},"scaleway":{"modelId":"Qwen/Qwen3.6-35B-A3B","providerModelId":"qwen3.6-35b-a3b"},"together":{"modelId":"moonshotai/Kimi-K3","providerModelId":"moonshotai/Kimi-K3"},"zai-org":{"modelId":"zai-org/GLM-4.6V-Flash","providerModelId":"glm-4.6v-flash"}} }
conversational />

### API specification

#### Request

| Headers |   |    |
| :--- | :--- | :--- |
| **authorization** | _string_ | Authentication header in the form `'Bearer: hf_****'` when `hf_****` is a personal user access token with "Inference Providers" permission. You can generate one from [your settings page](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained). |

| Payload |  |  |
| :--- | :--- | :--- |
| **frequency_penalty** | _number_ | Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. |
| **logprobs** | _boolean_ | Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message. |
| **max_tokens** | _integer_ | The maximum number of tokens that can be generated in the chat completion. |
| **messages*** | _object[]_ | A list of messages comprising the conversation so far. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#1)** | _unknown_ | One of the following: |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#1)** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;content*** | _unknown_ | One of the following: |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#1)** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#2)** | _object[]_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#1)** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;text*** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;type*** | _enum_ | Possible values: text. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#2)** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;image_url*** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;url*** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;type*** | _enum_ | Possible values: image_url. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#2)** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tool_calls*** | _object[]_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;function*** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;parameters** | _unknown_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;description** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;name*** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;id*** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;type*** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#2)** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;name** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;role*** | _string_ |  |
| **presence_penalty** | _number_ | Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics |
| **reasoning_effort** | _string_ | Optional. Constrains effort on reasoning for models that support reasoning. Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning. Common values: none, minimal, low, medium, high, xhigh. Support and defaults are provider and model-dependent. |
| **response_format** | _unknown_ | One of the following: |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#1)** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;type*** | _enum_ | Possible values: text. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#2)** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;type*** | _enum_ | Possible values: json_schema. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;json_schema*** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;name*** | _string_ | The name of the response format. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;description** | _string_ | A description of what the response format is for, used by the model to determine how to respond in the format. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;schema** | _object_ | The schema for the response format, described as a JSON Schema object. Learn how to build JSON schemas [here](https://json-schema.org/). |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;strict** | _boolean_ | Whether to enable strict schema adherence when generating the output. If set to true, the model will always follow the exact schema defined in the `schema` field. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#3)** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;type*** | _enum_ | Possible values: json_object. |
| **seed** | _integer_ |  |
| **stop** | _string[]_ | Up to 4 sequences where the API will stop generating further tokens. |
| **stream** | _boolean_ |  |
| **stream_options** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;include_usage** | _boolean_ | If set, an additional chunk will be streamed before the data: [DONE] message. The usage field on this chunk shows the token usage statistics for the entire request, and the choices field will always be an empty array. All other chunks will also include a usage field, but with a null value. |
| **temperature** | _number_ | What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.  We generally recommend altering this or `top_p` but not both. |
| **tool_choice** | _unknown_ | One of the following: |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#1)** | _enum_ | Possible values: auto. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#2)** | _enum_ | Possible values: none. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#3)** | _enum_ | Possible values: required. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#4)** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;function*** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;name*** | _string_ |  |
| **tool_prompt** | _string_ | A prompt to be appended before the tools |
| **tools** | _object[]_ | A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;function*** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;parameters** | _unknown_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;description** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;name*** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;type*** | _string_ |  |
| **top_logprobs** | _integer_ | An integer between 0 and 5 specifying the number of most likely tokens to return at each token position, each with an associated log probability. logprobs must be set to true if this parameter is used. |
| **top_p** | _number_ | An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. |

#### Response

Output type depends on the `stream` input parameter.
If `stream` is `false` (default), the response will be a JSON object with the following fields:

| Body |  |
| :--- | :--- | :--- |
| **choices** | _object[]_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;finish_reason** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;index** | _integer_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logprobs** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;content** | _object[]_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logprob** | _number_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;token** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;top_logprobs** | _object[]_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logprob** | _number_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;token** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;message** | _unknown_ | One of the following: |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#1)** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;content** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;role** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tool_call_id** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#2)** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;role** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tool_calls** | _object[]_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;function** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;arguments** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;description** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;name** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;id** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;type** | _string_ |  |
| **created** | _integer_ |  |
| **id** | _string_ |  |
| **model** | _string_ |  |
| **system_fingerprint** | _string_ |  |
| **usage** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;completion_tokens** | _integer_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;prompt_tokens** | _integer_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;total_tokens** | _integer_ |  |

If `stream` is `true`, generated tokens are returned as a stream, using Server-Sent Events (SSE).
For more information about streaming, check out [this guide](https://huggingface.co/docs/text-generation-inference/conceptual/streaming).

| Body |  |
| :--- | :--- | :--- |
| **choices** | _object[]_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;delta** | _unknown_ | One of the following: |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#1)** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;content** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;role** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tool_call_id** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#2)** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;role** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tool_calls** | _object[]_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;function** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;arguments** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;name** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;id** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;index** | _integer_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;type** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;finish_reason** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;index** | _integer_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logprobs** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;content** | _object[]_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logprob** | _number_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;token** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;top_logprobs** | _object[]_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logprob** | _number_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;token** | _string_ |  |
| **created** | _integer_ |  |
| **id** | _string_ |  |
| **model** | _string_ |  |
| **system_fingerprint** | _string_ |  |
| **usage** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;completion_tokens** | _integer_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;prompt_tokens** | _integer_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;total_tokens** | _integer_ |  |

### Feature Extraction
https://huggingface.co/docs/inference-providers/tasks/feature-extraction.md

## Feature Extraction

Feature extraction is the task of converting a text into a vector (often called "embedding").

Example applications:
* Retrieving the most relevant documents for a query (for RAG applications).
* Reranking a list of documents based on their similarity to a query.
* Calculating the similarity between two sentences.

> [!TIP]
> For more details about the `feature-extraction` task, check out its [dedicated page](https://huggingface.co/tasks/feature-extraction)! You will find examples and related materials.

### Recommended models

- [thenlper/gte-large](https://huggingface.co/thenlper/gte-large): A powerful feature extraction model for natural language processing tasks.

Explore all available models and find the one that suits you best [here](https://huggingface.co/models?inference=warm&pipeline_tag=feature-extraction&sort=trending), or from the terminal with the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-models-list):

```bash
hf models ls --warm --pipeline-tag feature-extraction --sort trending_score
```

### Using the API

<InferenceSnippet
    pipeline=feature-extraction
    providersMapping={ {"hf-inference":{"modelId":"microsoft/harrier-oss-v1-0.6b","providerModelId":"microsoft/harrier-oss-v1-0.6b"},"scaleway":{"modelId":"Qwen/Qwen3-Embedding-8B","providerModelId":"qwen3-embedding-8b"},"together":{"modelId":"intfloat/multilingual-e5-large-instruct","providerModelId":"intfloat/multilingual-e5-large-instruct"}} }
/>

### API specification

#### Request

| Headers |   |    |
| :--- | :--- | :--- |
| **authorization** | _string_ | Authentication header in the form `'Bearer: hf_****'` when `hf_****` is a personal user access token with "Inference Providers" permission. You can generate one from [your settings page](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained). |

| Payload |  |  |
| :--- | :--- | :--- |
| **inputs*** | _unknown_ | One of the following: |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#1)** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#2)** | _string[]_ |  |
| **normalize** | _boolean_ |  |
| **prompt_name** | _string_ | The name of the prompt that should be used by for encoding. If not set, no prompt will be applied.  Must be a key in the `sentence-transformers` configuration `prompts` dictionary.  For example if ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?" because the prompt text will be prepended before any text to encode. |
| **truncate** | _boolean_ |  |
| **truncation_direction** | _enum_ | Possible values: left, right. |

#### Response

| Body |  |
| :--- | :--- | :--- |
| **(array)** | _array[]_ | Output is an array of arrays. |

### Text Generation
https://huggingface.co/docs/inference-providers/tasks/text-generation.md

## Text Generation

Generate text based on a prompt.

If you are interested in a Chat Completion task, which generates a response based on a list of messages, check out the [`chat-completion`](./chat-completion) task.

> [!TIP]
> For more details about the `text-generation` task, check out its [dedicated page](https://huggingface.co/tasks/text-generation)! You will find examples and related materials.

### Recommended models

- [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it): A text-generation model trained to follow instructions.
- [Qwen/Qwen3-Coder-480B-A35B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct): Powerful text generation model for coding.
- [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b): Great text generation model with top-notch tool calling capabilities.
- [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507): A powerful small model with reasoning capabilities.
- [Qwen/Qwen2.5-7B-Instruct-1M](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M): Strong conversational model that supports very long instructions.
- [Qwen/Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct): Text generation model used to write code.
- [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1): Powerful reasoning based open large language model.

Explore all available models and find the one that suits you best [here](https://huggingface.co/models?inference=warm&pipeline_tag=text-generation&sort=trending), or from the terminal with the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-models-list):

```bash
hf models ls --warm --pipeline-tag text-generation --sort trending_score
```

### Using the API

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"featherless-ai":{"modelId":"poolside/Laguna-S-2.1","providerModelId":"poolside/Laguna-S-2.1"}} }
/>

### API specification

#### Request

| Headers |   |    |
| :--- | :--- | :--- |
| **authorization** | _string_ | Authentication header in the form `'Bearer: hf_****'` when `hf_****` is a personal user access token with "Inference Providers" permission. You can generate one from [your settings page](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained). |

| Payload |  |  |
| :--- | :--- | :--- |
| **inputs*** | _string_ |  |
| **parameters** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;adapter_id** | _string_ | Lora adapter id |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;best_of** | _integer_ | Generate best_of sequences and return the one if the highest token logprobs. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;decoder_input_details** | _boolean_ | Whether to return decoder input token logprobs and ids. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;details** | _boolean_ | Whether to return generation details. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;do_sample** | _boolean_ | Activate logits sampling. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;frequency_penalty** | _number_ | The parameter for frequency penalty. 1.0 means no penalty Penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;grammar** | _unknown_ | One of the following: |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#1)** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;type*** | _enum_ | Possible values: json. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;value*** | _unknown_ | A string that represents a [JSON Schema](https://json-schema.org/).  JSON Schema is a declarative language that allows to annotate JSON documents with types and descriptions. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#2)** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;type*** | _enum_ | Possible values: regex. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;value*** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(#3)** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;type*** | _enum_ | Possible values: json_schema. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;value*** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;name** | _string_ | Optional name identifier for the schema |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;schema*** | _unknown_ | The actual JSON schema definition |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;max_new_tokens** | _integer_ | Maximum number of tokens to generate. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;repetition_penalty** | _number_ | The parameter for repetition penalty. 1.0 means no penalty. See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return_full_text** | _boolean_ | Whether to prepend the prompt to the generated text |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;seed** | _integer_ | Random sampling seed. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;stop** | _string[]_ | Stop generating tokens if a member of `stop` is generated. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;temperature** | _number_ | The value used to module the logits distribution. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;top_k** | _integer_ | The number of highest probability vocabulary tokens to keep for top-k-filtering. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;top_n_tokens** | _integer_ | The number of highest probability vocabulary tokens to keep for top-n-filtering. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;top_p** | _number_ | Top-p value for nucleus sampling. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;truncate** | _integer_ | Truncate inputs tokens to the given size. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typical_p** | _number_ | Typical Decoding mass See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;watermark** | _boolean_ | Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226). |
| **stream** | _boolean_ |  |

#### Response

Output type depends on the `stream` input parameter.
If `stream` is `false` (default), the response will be a JSON object with the following fields:

| Body |  |
| :--- | :--- | :--- |
| **details** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;best_of_sequences** | _object[]_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;finish_reason** | _enum_ | Possible values: length, eos_token, stop_sequence. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;generated_text** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;generated_tokens** | _integer_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;prefill** | _object[]_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;id** | _integer_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logprob** | _number_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;text** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;seed** | _integer_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tokens** | _object[]_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;id** | _integer_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logprob** | _number_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;special** | _boolean_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;text** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;top_tokens** | _array[]_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;id** | _integer_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logprob** | _number_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;special** | _boolean_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;text** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;finish_reason** | _enum_ | Possible values: length, eos_token, stop_sequence. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;generated_tokens** | _integer_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;prefill** | _object[]_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;id** | _integer_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logprob** | _number_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;text** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;seed** | _integer_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tokens** | _object[]_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;id** | _integer_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logprob** | _number_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;special** | _boolean_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;text** | _string_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;top_tokens** | _array[]_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;id** | _integer_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logprob** | _number_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;special** | _boolean_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;text** | _string_ |  |
| **generated_text** | _string_ |  |

If `stream` is `true`, generated tokens are returned as a stream, using Server-Sent Events (SSE).
For more information about streaming, check out [this guide](https://huggingface.co/docs/text-generation-inference/conceptual/streaming).

| Body |  |
| :--- | :--- | :--- |
| **details** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;finish_reason** | _enum_ | Possible values: length, eos_token, stop_sequence. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;generated_tokens** | _integer_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;input_length** | _integer_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;seed** | _integer_ |  |
| **generated_text** | _string_ |  |
| **index** | _integer_ |  |
| **token** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;id** | _integer_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logprob** | _number_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;special** | _boolean_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;text** | _string_ |  |
| **top_tokens** | _object[]_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;id** | _integer_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logprob** | _number_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;special** | _boolean_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;text** | _string_ |  |

### Table Question Answering
https://huggingface.co/docs/inference-providers/tasks/table-question-answering.md

## Table Question Answering

Table Question Answering (Table QA) is the answering a question about an information on a given table.

> [!TIP]
> For more details about the `table-question-answering` task, check out its [dedicated page](https://huggingface.co/tasks/table-question-answering)! You will find examples and related materials.

### Recommended models

- [google/tapas-base-finetuned-wtq](https://huggingface.co/google/tapas-base-finetuned-wtq): A robust table question answering model.

Explore all available models and find the one that suits you best [here](https://huggingface.co/models?inference=warm&pipeline_tag=table-question-answering&sort=trending), or from the terminal with the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-models-list):

```bash
hf models ls --warm --pipeline-tag table-question-answering --sort trending_score
```

### Using the API

<InferenceSnippet
    pipeline=table-question-answering
    providersMapping={ {"hf-inference":{"modelId":"google/tapas-base-finetuned-wtq","providerModelId":"google/tapas-base-finetuned-wtq"}} }
/>

### API specification

#### Request

| Headers |   |    |
| :--- | :--- | :--- |
| **authorization** | _string_ | Authentication header in the form `'Bearer: hf_****'` when `hf_****` is a personal user access token with "Inference Providers" permission. You can generate one from [your settings page](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained). |

| Payload |  |  |
| :--- | :--- | :--- |
| **inputs*** | _object_ | One (table, question) pair to answer |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;table*** | _object_ | The table to serve as context for the questions |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;question*** | _string_ | The question to be answered about the table |
| **parameters** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;padding** | _enum_ | Possible values: do_not_pad, longest, max_length. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;sequential** | _boolean_ | Whether to do inference sequentially or as a batch. Batching is faster, but models like SQA require the inference to be done sequentially to extract relations within sequences, given their conversational nature. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;truncation** | _boolean_ | Activates and controls truncation. |

#### Response

| Body |  |
| :--- | :--- | :--- |
| **(array)** | _object[]_ | Output is an array of objects. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;answer** | _string_ | The answer of the question given the table. If there is an aggregator, the answer will be preceded by `AGGREGATOR >`. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;coordinates** | _array[]_ | Coordinates of the cells of the answers. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;cells** | _string[]_ | List of strings made up of the answer cell values. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;aggregator** | _string_ | If the model has an aggregator, this returns the aggregator. |

### Zero-Shot Classification
https://huggingface.co/docs/inference-providers/tasks/zero-shot-classification.md

## Zero-Shot Classification

Zero-shot text classification is super useful to try out classification with zero code, you simply pass a sentence/paragraph and the possible labels for that sentence, and you get a result. The model has not been necessarily trained on the labels you provide, but it can still predict the correct label.

> [!TIP]
> For more details about the `zero-shot-classification` task, check out its [dedicated page](https://huggingface.co/tasks/zero-shot-classification)! You will find examples and related materials.

### Recommended models

- [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli): Powerful zero-shot text classification model.
- [MoritzLaurer/ModernBERT-large-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0): Cutting-edge zero-shot multilingual text classification model.

Explore all available models and find the one that suits you best [here](https://huggingface.co/models?inference=warm&pipeline_tag=zero-shot-classification&sort=trending), or from the terminal with the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-models-list):

```bash
hf models ls --warm --pipeline-tag zero-shot-classification --sort trending_score
```

### Using the API

<InferenceSnippet
    pipeline=zero-shot-classification
    providersMapping={ {"hf-inference":{"modelId":"facebook/bart-large-mnli","providerModelId":"facebook/bart-large-mnli"}} }
/>

### API specification

#### Request

| Headers |   |    |
| :--- | :--- | :--- |
| **authorization** | _string_ | Authentication header in the form `'Bearer: hf_****'` when `hf_****` is a personal user access token with "Inference Providers" permission. You can generate one from [your settings page](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained). |

| Payload |  |  |
| :--- | :--- | :--- |
| **inputs*** | _string_ | The text to classify |
| **parameters*** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;candidate_labels*** | _string[]_ | The set of possible class labels to classify the text into. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;hypothesis_template** | _string_ | The sentence used in conjunction with `candidate_labels` to attempt the text classification by replacing the placeholder with the candidate labels. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;multi_label** | _boolean_ | Whether multiple candidate labels can be true. If false, the scores are normalized such that the sum of the label likelihoods for each sequence is 1. If true, the labels are considered independent and probabilities are normalized for each candidate. |

#### Response

| Body |  |
| :--- | :--- | :--- |
| **(array)** | _object[]_ | Output is an array of objects. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;label** | _string_ | The predicted class label. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;score** | _number_ | The corresponding probability. |

### Object detection
https://huggingface.co/docs/inference-providers/tasks/object-detection.md

## Object detection

Object Detection models allow users to identify objects of certain defined classes. These models receive an image as input and output the images with bounding boxes and labels on detected objects.

> [!TIP]
> For more details about the `object-detection` task, check out its [dedicated page](https://huggingface.co/tasks/object-detection)! You will find examples and related materials.

### Recommended models

- [facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50): Solid object detection model pre-trained on the COCO 2017 dataset.

Explore all available models and find the one that suits you best [here](https://huggingface.co/models?inference=warm&pipeline_tag=object-detection&sort=trending), or from the terminal with the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-models-list):

```bash
hf models ls --warm --pipeline-tag object-detection --sort trending_score
```

### Using the API

<InferenceSnippet
    pipeline=object-detection
    providersMapping={ {"hf-inference":{"modelId":"facebook/detr-resnet-50","providerModelId":"facebook/detr-resnet-50"}} }
/>

### API specification

#### Request

| Headers |   |    |
| :--- | :--- | :--- |
| **authorization** | _string_ | Authentication header in the form `'Bearer: hf_****'` when `hf_****` is a personal user access token with "Inference Providers" permission. You can generate one from [your settings page](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained). |

| Payload |  |  |
| :--- | :--- | :--- |
| **inputs*** | _string_ | The input image data as a base64-encoded string. If no `parameters` are provided, you can also provide the image data as a raw bytes payload. |
| **parameters** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;threshold** | _number_ | The probability necessary to make a prediction. |

#### Response

| Body |  |
| :--- | :--- | :--- |
| **(array)** | _object[]_ | Output is an array of objects. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;label** | _string_ | The predicted label for the bounding box. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;score** | _number_ | The associated score / probability. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;box** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;xmin** | _integer_ | The x-coordinate of the top-left corner of the bounding box. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;xmax** | _integer_ | The x-coordinate of the bottom-right corner of the bounding box. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ymin** | _integer_ | The y-coordinate of the top-left corner of the bounding box. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ymax** | _integer_ | The y-coordinate of the bottom-right corner of the bounding box. |

### Text to Video
https://huggingface.co/docs/inference-providers/tasks/text-to-video.md

## Text to Video

Generate an video based on a given text prompt.

> [!TIP]
> For more details about the `text-to-video` task, check out its [dedicated page](https://huggingface.co/tasks/text-to-video)! You will find examples and related materials.

### Recommended models

- [tencent/HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo): A strong model for consistent video generation.
- [Lightricks/LTX-Video-0.9.8-13B-distilled](https://huggingface.co/Lightricks/LTX-Video-0.9.8-13B-distilled): Very fast model for video generation.

Explore all available models and find the one that suits you best [here](https://huggingface.co/models?inference=warm&pipeline_tag=text-to-video&sort=trending), or from the terminal with the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-models-list):

```bash
hf models ls --warm --pipeline-tag text-to-video --sort trending_score
```

### Using the API

<InferenceSnippet
    pipeline=text-to-video
    providersMapping={ {"fal-ai":{"modelId":"Wan-AI/Wan2.2-TI2V-5B","providerModelId":"fal-ai/wan/v2.2-5b/text-to-video"},"replicate":{"modelId":"Wan-AI/Wan2.2-TI2V-5B","providerModelId":"wan-video/wan-2.2-5b-fast"},"wavespeed":{"modelId":"Wan-AI/Wan2.2-TI2V-5B","providerModelId":"wavespeed-ai/wan-2.2/t2v-5b-720p"}} }
/>

### API specification

#### Request

| Payload |  |  |
| :--- | :--- | :--- |
| **inputs*** | _string_ | The input text data (sometimes called "prompt") |
| **parameters** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;num_frames** | _number_ | The num_frames parameter determines how many video frames are generated. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;guidance_scale** | _number_ | A higher guidance scale value encourages the model to generate videos closely linked to the text prompt, but values too high may cause saturation and other artifacts. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;negative_prompt** | _string[]_ | One or several prompt to guide what NOT to include in video generation. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;num_inference_steps** | _integer_ | The number of denoising steps. More denoising steps usually lead to a higher quality video at the expense of slower inference. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;seed** | _integer_ | Seed for the random number generator. |

| Headers |   |    |
| :--- | :--- | :--- |
| **authorization** | _string_ | Authentication header in the form `'Bearer: hf_****'` when `hf_****` is a personal user access token with "Inference Providers" permission. You can generate one from [your settings page](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained). |

#### Response

| Body |  |
| :--- | :--- | :--- |
| **video** | _unknown_ | The generated video returned as raw bytes in the payload. |

### Question Answering
https://huggingface.co/docs/inference-providers/tasks/question-answering.md

## Question Answering

Question Answering models can retrieve the answer to a question from a given text, which is useful for searching for an answer in a document.

> [!TIP]
> For more details about the `question-answering` task, check out its [dedicated page](https://huggingface.co/tasks/question-answering)! You will find examples and related materials.

### Recommended models

- [deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2): A robust baseline model for most question answering domains.
- [distilbert/distilbert-base-cased-distilled-squad](https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad): Small yet robust model that can answer questions.
- [google/tapas-base-finetuned-wtq](https://huggingface.co/google/tapas-base-finetuned-wtq): A special model that can answer questions from tables.

Explore all available models and find the one that suits you best [here](https://huggingface.co/models?inference=warm&pipeline_tag=question-answering&sort=trending), or from the terminal with the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-models-list):

```bash
hf models ls --warm --pipeline-tag question-answering --sort trending_score
```

### Using the API

<InferenceSnippet
    pipeline=question-answering
    providersMapping={ {"hf-inference":{"modelId":"deepset/roberta-base-squad2","providerModelId":"deepset/roberta-base-squad2"}} }
/>

### API specification

#### Request

| Headers |   |    |
| :--- | :--- | :--- |
| **authorization** | _string_ | Authentication header in the form `'Bearer: hf_****'` when `hf_****` is a personal user access token with "Inference Providers" permission. You can generate one from [your settings page](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained). |

| Payload |  |  |
| :--- | :--- | :--- |
| **inputs*** | _object_ | One (context, question) pair to answer |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;context*** | _string_ | The context to be used for answering the question |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;question*** | _string_ | The question to be answered |
| **parameters** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;top_k** | _integer_ | The number of answers to return (will be chosen by order of likelihood). Note that we return less than topk answers if there are not enough options available within the context. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;doc_stride** | _integer_ | If the context is too long to fit with the question for the model, it will be split in several chunks with some overlap. This argument controls the size of that overlap. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;max_answer_len** | _integer_ | The maximum length of predicted answers (e.g., only answers with a shorter length are considered). |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;max_seq_len** | _integer_ | The maximum length of the total sentence (context + question) in tokens of each chunk passed to the model. The context will be split in several chunks (using docStride as overlap) if needed. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;max_question_len** | _integer_ | The maximum length of the question after tokenization. It will be truncated if needed. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;handle_impossible_answer** | _boolean_ | Whether to accept impossible as an answer. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;align_to_words** | _boolean_ | Attempts to align the answer to real words. Improves quality on space separated languages. Might hurt on non-space-separated languages (like Japanese or Chinese) |

#### Response

| Body |  |
| :--- | :--- | :--- |
| **(array)** | _object[]_ | Output is an array of objects. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;answer** | _string_ | The answer to the question. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;score** | _number_ | The probability associated to the answer. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;start** | _integer_ | The character position in the input where the answer begins. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;end** | _integer_ | The character position in the input where the answer ends. |

### Summarization
https://huggingface.co/docs/inference-providers/tasks/summarization.md

## Summarization

Summarization is the task of producing a shorter version of a document while preserving its important information. Some models can extract text from the original input, while other models can generate entirely new text.

> [!TIP]
> For more details about the `summarization` task, check out its [dedicated page](https://huggingface.co/tasks/summarization)! You will find examples and related materials.

### Recommended models

- [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn): A strong summarization model trained on English news articles. Excels at generating factual summaries.
- [Falconsai/medical_summarization](https://huggingface.co/Falconsai/medical_summarization): A summarization model trained on medical articles.

Explore all available models and find the one that suits you best [here](https://huggingface.co/models?inference=warm&pipeline_tag=summarization&sort=trending), or from the terminal with the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-models-list):

```bash
hf models ls --warm --pipeline-tag summarization --sort trending_score
```

### Using the API

<InferenceSnippet
    pipeline=summarization
    providersMapping={ {"hf-inference":{"modelId":"human-centered-summarization/financial-summarization-pegasus","providerModelId":"human-centered-summarization/financial-summarization-pegasus"}} }
/>

### API specification

#### Request

| Headers |   |    |
| :--- | :--- | :--- |
| **authorization** | _string_ | Authentication header in the form `'Bearer: hf_****'` when `hf_****` is a personal user access token with "Inference Providers" permission. You can generate one from [your settings page](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained). |

| Payload |  |  |
| :--- | :--- | :--- |
| **inputs*** | _string_ | The input text to summarize. |
| **parameters** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;clean_up_tokenization_spaces** | _boolean_ | Whether to clean up the potential extra spaces in the text output. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;truncation** | _enum_ | Possible values: do_not_truncate, longest_first, only_first, only_second. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;generate_parameters** | _object_ | Additional parametrization of the text generation algorithm. |

#### Response

| Body |  |
| :--- | :--- | :--- |
| **summary_text** | _string_ | The summarized text. |

### Automatic Speech Recognition
https://huggingface.co/docs/inference-providers/tasks/automatic-speech-recognition.md

## Automatic Speech Recognition

Automatic Speech Recognition (ASR), also known as Speech to Text (STT), is the task of transcribing a given audio to text.

Example applications:
* Transcribing a podcast
* Building a voice assistant
* Generating subtitles for a video

> [!TIP]
> For more details about the `automatic-speech-recognition` task, check out its [dedicated page](https://huggingface.co/tasks/automatic-speech-recognition)! You will find examples and related materials.

### Recommended models

- [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3): A powerful ASR model by OpenAI.

Explore all available models and find the one that suits you best [here](https://huggingface.co/models?inference=warm&pipeline_tag=automatic-speech-recognition&sort=trending), or from the terminal with the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-models-list):

```bash
hf models ls --warm --pipeline-tag automatic-speech-recognition --sort trending_score
```

### Using the API

<InferenceSnippet
    pipeline=automatic-speech-recognition
    providersMapping={ {"deepinfra":{"modelId":"nvidia/nemotron-3.5-asr-streaming-0.6b","providerModelId":"nvidia/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b"},"fal-ai":{"modelId":"nvidia/nemotron-3.5-asr-streaming-0.6b","providerModelId":"nvidia/nemotron-asr-multilingual/asr"},"hf-inference":{"modelId":"openai/whisper-large-v3","providerModelId":"openai/whisper-large-v3"},"replicate":{"modelId":"openai/whisper-large-v3","providerModelId":"openai/whisper:8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e"},"together":{"modelId":"nvidia/nemotron-3.5-asr-streaming-0.6b","providerModelId":"nvidia/nemotron-3.5-asr-streaming-0.6b"}} }
/>

### API specification

#### Request

| Headers |   |    |
| :--- | :--- | :--- |
| **authorization** | _string_ | Authentication header in the form `'Bearer: hf_****'` when `hf_****` is a personal user access token with "Inference Providers" permission. You can generate one from [your settings page](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained). |

| Payload |  |  |
| :--- | :--- | :--- |
| **inputs*** | _string_ | The input audio data as a base64-encoded string. If no `parameters` are provided, you can also provide the audio data as a raw bytes payload. |
| **parameters** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return_timestamps** | _boolean_ | Whether to output corresponding timestamps with the generated text |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;generation_parameters** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;temperature** | _number_ | The value used to modulate the next token probabilities. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;top_k** | _integer_ | The number of highest probability vocabulary tokens to keep for top-k-filtering. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;top_p** | _number_ | If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;typical_p** | _number_ |  Local typicality measures how similar the conditional probability of predicting a target token next is to the expected conditional probability of predicting a random token next, given the partial text already generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that add up to typical_p or higher are kept for generation. See [this paper](https://hf.co/papers/2202.00666) for more details. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;epsilon_cutoff** | _number_ | If set to float strictly between 0 and 1, only tokens with a conditional probability greater than epsilon_cutoff will be sampled. In the paper, suggested values range from 3e-4 to 9e-4, depending on the size of the model. See [Truncation Sampling as Language Model Desmoothing](https://hf.co/papers/2210.15191) for more details. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;eta_cutoff** | _number_ | Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to float strictly between 0 and 1, a token is only considered if it is greater than either eta_cutoff or sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits))). The latter term is intuitively the expected next token probability, scaled by sqrt(eta_cutoff). In the paper, suggested values range from 3e-4 to 2e-3, depending on the size of the model. See [Truncation Sampling as Language Model Desmoothing](https://hf.co/papers/2210.15191) for more details. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;max_length** | _integer_ | The maximum length (in tokens) of the generated text, including the input. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;max_new_tokens** | _integer_ | The maximum number of tokens to generate. Takes precedence over max_length. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;min_length** | _integer_ | The minimum length (in tokens) of the generated text, including the input. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;min_new_tokens** | _integer_ | The minimum number of tokens to generate. Takes precedence over min_length. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;do_sample** | _boolean_ | Whether to use sampling instead of greedy decoding when generating new tokens. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;early_stopping** | _enum_ | Possible values: never, true, false. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;num_beams** | _integer_ | Number of beams to use for beam search. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;num_beam_groups** | _integer_ | Number of groups to divide num_beams into in order to ensure diversity among different groups of beams. See [this paper](https://hf.co/papers/1610.02424) for more details. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;penalty_alpha** | _number_ | The value balances the model confidence and the degeneration penalty in contrastive search decoding. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;use_cache** | _boolean_ | Whether the model should use the past last key/values attentions to speed up decoding |

#### Response

| Body |  |
| :--- | :--- | :--- |
| **text** | _string_ | The recognized text. |
| **chunks** | _object[]_ | When returnTimestamps is enabled, chunks contains a list of audio chunks identified by the model. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;text** | _string_ | A chunk of text identified by the model |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;timestamp** | _number[]_ | The start and end timestamps corresponding with the text |

### Translation
https://huggingface.co/docs/inference-providers/tasks/translation.md

## Translation

Translation is the task of converting text from one language to another.

> [!TIP]
> For more details about the `translation` task, check out its [dedicated page](https://huggingface.co/tasks/translation)! You will find examples and related materials.

### Recommended models

- [google-t5/t5-base](https://huggingface.co/google-t5/t5-base): A general-purpose Transformer that can be used to translate from English to German, French, or Romanian.

Explore all available models and find the one that suits you best [here](https://huggingface.co/models?inference=warm&pipeline_tag=translation&sort=trending), or from the terminal with the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-models-list):

```bash
hf models ls --warm --pipeline-tag translation --sort trending_score
```

### Using the API

<InferenceSnippet
    pipeline=translation
    providersMapping={ {"hf-inference":{"modelId":"google-t5/t5-small","providerModelId":"google-t5/t5-small"}} }
/>

### API specification

#### Request

| Headers |   |    |
| :--- | :--- | :--- |
| **authorization** | _string_ | Authentication header in the form `'Bearer: hf_****'` when `hf_****` is a personal user access token with "Inference Providers" permission. You can generate one from [your settings page](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained). |

| Payload |  |  |
| :--- | :--- | :--- |
| **inputs*** | _string_ | The text to translate. |
| **parameters** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;src_lang** | _string_ | The source language of the text. Required for models that can translate from multiple languages. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tgt_lang** | _string_ | Target language to translate to. Required for models that can translate to multiple languages. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;clean_up_tokenization_spaces** | _boolean_ | Whether to clean up the potential extra spaces in the text output. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;truncation** | _enum_ | Possible values: do_not_truncate, longest_first, only_first, only_second. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;generate_parameters** | _object_ | Additional parametrization of the text generation algorithm. |

#### Response

| Body |  |
| :--- | :--- | :--- |
| **translation_text** | _string_ | The translated text. |

### Image-Text to Text
https://huggingface.co/docs/inference-providers/tasks/image-text-to-text.md

## Image-Text to Text

Image-text-to-text models take in an image and text prompt and output text. These models are also called vision-language models, or VLMs. The difference from image-to-text models is that these models take an additional text input, not restricting the model to certain use cases like image captioning, and may also be trained to accept a conversation as input.

> [!TIP]
> For more details about the `image-text-to-text` task, check out its [dedicated page](https://huggingface.co/tasks/image-text-to-text)! You will find examples and related materials.

### Recommended models

- [zai-org/GLM-4.5V](https://huggingface.co/zai-org/GLM-4.5V): Cutting-edge reasoning vision language model.
- [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct): Small yet powerful model.

Explore all available models and find the one that suits you best [here](https://huggingface.co/models?inference=warm&pipeline_tag=image-text-to-text&sort=trending), or from the terminal with the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-models-list):

```bash
hf models ls --warm --pipeline-tag image-text-to-text --sort trending_score
```

### Using the API

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"baseten":{"modelId":"moonshotai/Kimi-K3","providerModelId":"moonshotai/Kimi-K3"},"cerebras":{"modelId":"google/gemma-4-31B-it","providerModelId":"gemma-4-31b"},"cohere":{"modelId":"CohereLabs/command-a-plus-05-2026-w4a4","providerModelId":"command-a-plus-05-2026"},"deepinfra":{"modelId":"moonshotai/Kimi-K3","providerModelId":"moonshotai/Kimi-K3"},"featherless-ai":{"modelId":"moonshotai/Kimi-K3","providerModelId":"moonshotai/Kimi-K3"},"fireworks-ai":{"modelId":"moonshotai/Kimi-K3","providerModelId":"accounts/fireworks/models/kimi-k3"},"novita":{"modelId":"google/gemma-4-31B-it","providerModelId":"google/gemma-4-31b-it"},"nscale":{"modelId":"meta-llama/Llama-4-Scout-17B-16E-Instruct","providerModelId":"meta-llama/Llama-4-Scout-17B-16E-Instruct"},"ovhcloud":{"modelId":"Qwen/Qwen3.6-27B","providerModelId":"Qwen3.6-27B"},"publicai":{"modelId":"swiss-ai/Apertus-v1.5-70B","providerModelId":"swiss-ai/apertus-v1.5-70b"},"scaleway":{"modelId":"Qwen/Qwen3.6-35B-A3B","providerModelId":"qwen3.6-35b-a3b"},"together":{"modelId":"moonshotai/Kimi-K3","providerModelId":"moonshotai/Kimi-K3"},"zai-org":{"modelId":"zai-org/GLM-4.6V-Flash","providerModelId":"glm-4.6v-flash"}} }
conversational />

### API specification

For the API specification of conversational image-text-to-text models, please refer to the [Chat Completion API documentation](https://huggingface.co/docs/inference-providers/tasks/chat-completion#api-specification).

### API Reference
https://huggingface.co/docs/inference-providers/tasks/index.md
