# API Reference

## Popular tasks

  

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./chat-completion">
      
        Chat Completion
      
        Generate a response given a list of messages in a conversational context.
      
    

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./feature-extraction">
      
        Feature Extraction
      
        Converting a text into a vector often called "embedding".
      
    

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./text-to-image">
      
        Text to Image
      
        Generate an image based on a given text prompt.
      
    

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./text-to-video">
      
        Text to Video
      
        Generate an video based on a given text prompt.
      
    

  

## Other tasks

  

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./audio-classification">
      
        Audio Classification
      
        Audio classification is the task of assigning a label or class to a given audio.
      
    

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./automatic-speech-recognition">
      
        Automatic Speech Recognition
      
        Automatic Speech Recognition (ASR), also known as Speech to Text (STT), is the task of transcribing a given audio to text.
      
    

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./fill-mask">
      
        Fill Mask
      
        Mask filling is the task of predicting the right word (token to be precise) in the middle of a sequence.
      
    

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./image-classification">
      
        Image Classification
      
        Image classification is the task of assigning a label or class to an entire image. Images are expected to have only one class for each image.
      
    

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./image-segmentation">
      
        Image Segmentation
      
        Image Segmentation divides an image into segments where each pixel in the image is mapped to an object.
      
    

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./image-to-image">
      
        Image to Image
      
        Image-to-image is the task of transforming a source image to match the characteristics of a target image or a target image domain.
      
    

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./object-detection">
      
        Object Detection
      
        Object Detection models allow users to identify objects of certain defined classes.
      
    

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./question-answering">
      
        Question Answering
      
        Question Answering models can retrieve the answer to a question from a given text, which is useful for searching for an answer in a document.
      
    

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./summarization">
      
        Summarization
      
        Summarization is the task of producing a shorter version of a document while preserving its important information.
      
    

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./table-question-answering">
      
        Table Question Answering
      
        Table Question Answering (Table QA) is the answering a question about an information on a given table.
      
    

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./text-classification">
      
        Text Classification
      
        Text Classification is the task of assigning a label or class to a given text.
      
    

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./text-generation">
      
        Text Generation
      
        Generate text based on a prompt.
      
    

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./token-classification">
      
        Token Classification
      
        Token classification is a task in which a label is assigned to some tokens in a text.
      
    

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./translation">
      
        Translation
      
        Translation is the task of converting text from one language to another.
      
    

    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg"
       href="./zero-shot-classification">
      
        Zero Shot Classification
      
        Zero shot classification is the task to classify text without specific training for the task.

### Text to Image
https://huggingface.co/docs/inference-providers/tasks/text-to-image.md

## Text to Image

Generate an image based on a given text prompt.

> [!TIP]
> For more details about the `text-to-image` task, check out its [dedicated page](https://huggingface.co/tasks/text-to-image)! You will find examples and related materials.

### Recommended models

- [black-forest-labs/FLUX.1-Krea-dev](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev): One of the most powerful image generation models that can generate realistic outputs.
- [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image): A powerful image generation model.
- [ByteDance/Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD): A powerful text-to-image model.

Explore all available models and find the one that suits you best [here](https://huggingface.co/models?inference=warm&pipeline_tag=text-to-image&sort=trending), or from the terminal with the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-models-list):

```bash
hf models ls --warm --pipeline-tag text-to-image --sort trending_score
```

### Using the API

<InferenceSnippet
    pipeline=text-to-image
    providersMapping={ {"fal-ai":{"modelId":"krea/Krea-2-Turbo","providerModelId":"fal-ai/krea-2/turbo"},"hf-inference":{"modelId":"stabilityai/stable-diffusion-3-medium-diffusers","providerModelId":"stabilityai/stable-diffusion-3-medium-diffusers"},"nscale":{"modelId":"black-forest-labs/FLUX.1-schnell","providerModelId":"black-forest-labs/FLUX.1-schnell"},"replicate":{"modelId":"black-forest-labs/FLUX.1-dev","providerModelId":"black-forest-labs/flux-dev"},"wavespeed":{"modelId":"krea/Krea-2-Turbo","providerModelId":"wavespeed-ai/krea-v2-medium-turbo/text-to-image"}} }
/>

### API specification

#### Request

| Headers |   |    |
| :--- | :--- | :--- |
| **authorization** | _string_ | Authentication header in the form `'Bearer: hf_****'` when `hf_****` is a personal user access token with "Inference Providers" permission. You can generate one from [your settings page](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained). |

| Payload |  |  |
| :--- | :--- | :--- |
| **inputs*** | _string_ | The input text data (sometimes called "prompt") |
| **parameters** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;guidance_scale** | _number_ | A higher guidance scale value encourages the model to generate images closely linked to the text prompt, but values too high may cause saturation and other artifacts. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;negative_prompt** | _string_ | One prompt to guide what NOT to include in image generation. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;num_inference_steps** | _integer_ | The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;width** | _integer_ | The width in pixels of the output image |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;height** | _integer_ | The height in pixels of the output image |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;scheduler** | _string_ | Override the scheduler with a compatible one. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;seed** | _integer_ | Seed for the random number generator. |

#### Response

| Body |  |
| :--- | :--- | :--- |
| **image** | _unknown_ | The generated image returned as raw bytes in the payload. |

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
    providersMapping={ {"fal-ai":{"modelId":"briaai/RMBG-2.0","providerModelId":"fal-ai/bria/background/remove"},"hf-inference":{"modelId":"nvidia/segformer-b2-finetuned-ade-512-512","providerModelId":"nvidia/segformer-b2-finetuned-ade-512-512"}} }
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
    providersMapping={ {"hf-inference":{"modelId":"jeff-legacy/Jeff-Qwen3.5-0.8B","providerModelId":"jeff-legacy/Jeff-Qwen3.5-0.8B"}} }
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
    providersMapping={ {"fal-ai":{"modelId":"Wan-AI/Wan2.2-TI2V-5B","providerModelId":"fal-ai/wan/v2.2-5b/text-to-video"},"replicate":{"modelId":"Wan-AI/Wan2.2-TI2V-5B","providerModelId":"wan-video/wan-2.2-5b-fast"},"wavespeed":{"modelId":"ZhengmingYu/DMAD","providerModelId":"wavespeed-ai/minimax-h3/text-to-video-lora"}} }
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
    providersMapping={ {"hf-inference":{"modelId":"meta-llama/Llama-Prompt-Guard-2-86M","providerModelId":"meta-llama/Llama-Prompt-Guard-2-86M"}} }
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

### Fill-mask
https://huggingface.co/docs/inference-providers/tasks/fill-mask.md

## Fill-mask

Mask filling is the task of predicting the right word (token to be precise) in the middle of a sequence.

> [!TIP]
> For more details about the `fill-mask` task, check out its [dedicated page](https://huggingface.co/tasks/fill-mask)! You will find examples and related materials.

### Recommended models

- [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base): A multilingual model trained on 100 languages.

Explore all available models and find the one that suits you best [here](https://huggingface.co/models?inference=warm&pipeline_tag=fill-mask&sort=trending), or from the terminal with the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-models-list):

```bash
hf models ls --warm --pipeline-tag fill-mask --sort trending_score
```

### Using the API

<InferenceSnippet
    pipeline=fill-mask
    providersMapping={ {"hf-inference":{"modelId":"google-bert/bert-base-uncased","providerModelId":"google-bert/bert-base-uncased"}} }
/>

### API specification

#### Request

| Headers |   |    |
| :--- | :--- | :--- |
| **authorization** | _string_ | Authentication header in the form `'Bearer: hf_****'` when `hf_****` is a personal user access token with "Inference Providers" permission. You can generate one from [your settings page](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained). |

| Payload |  |  |
| :--- | :--- | :--- |
| **inputs*** | _string_ | The text with masked tokens |
| **parameters** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;top_k** | _integer_ | When passed, overrides the number of predictions to return. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;targets** | _string[]_ | When passed, the model will limit the scores to the passed targets instead of looking up in the whole vocabulary. If the provided targets are not in the model vocab, they will be tokenized and the first resulting token will be used (with a warning, and that might be slower). |

#### Response

| Body |  |
| :--- | :--- | :--- |
| **(array)** | _object[]_ | Output is an array of objects. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;sequence** | _string_ | The corresponding input with the mask token prediction. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;score** | _number_ | The corresponding probability |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;token** | _integer_ | The predicted token id (to replace the masked one). |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;token_str** | _string_ | The predicted token (to replace the masked one). |

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
    providersMapping={ {"baseten":{"modelId":"zai-org/GLM-5.3","providerModelId":"zai-org/GLM-5.3"},"cerebras":{"modelId":"openai/gpt-oss-120b","providerModelId":"gpt-oss-120b"},"cohere":{"modelId":"CohereLabs/tiny-aya-global","providerModelId":"tiny-aya-global"},"deepinfra":{"modelId":"zai-org/GLM-5.3","providerModelId":"zai-org/GLM-5.3"},"featherless-ai":{"modelId":"zai-org/GLM-5.3","providerModelId":"zai-org/GLM-5.3"},"fireworks-ai":{"modelId":"zai-org/GLM-5.3","providerModelId":"accounts/fireworks/models/glm-5p3"},"groq":{"modelId":"openai/gpt-oss-120b","providerModelId":"openai/gpt-oss-120b"},"novita":{"modelId":"XiaomiMiMo/MiMo-V2.6-Pro-RL","providerModelId":"xiaomimimo/mimo-v2.6-pro"},"nscale":{"modelId":"meta-llama/Llama-3.1-8B-Instruct","providerModelId":"meta-llama/Llama-3.1-8B-Instruct"},"ovhcloud":{"modelId":"openai/gpt-oss-120b","providerModelId":"gpt-oss-120b"},"publicai":{"modelId":"speakleash/Bielik-11B-v3.0-Instruct","providerModelId":"speakleash/Bielik-11B-v3.0-Instruct"},"scaleway":{"modelId":"openai/gpt-oss-120b","providerModelId":"gpt-oss-120b"},"together":{"modelId":"zai-org/GLM-5.3","providerModelId":"zai-org/GLM-5.3"},"zai-org":{"modelId":"zai-org/GLM-5.3","providerModelId":"glm-5.3"}} }
conversational />

#### Code snippet example for conversational VLMs

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"baseten":{"modelId":"deepseek-ai/DeepSeek-V4.1-Flash-1017","providerModelId":"deepseek-ai/DeepSeek-V4.1-Flash-1017"},"cerebras":{"modelId":"Qwen/Qwen3.8-27B","providerModelId":"qwen-3.8-27b"},"cohere":{"modelId":"CohereLabs/aya-vision-32b","providerModelId":"c4ai-aya-vision-32b"},"deepinfra":{"modelId":"Qwen/Qwen3.8-27B","providerModelId":"Qwen/Qwen3.8-27B"},"featherless-ai":{"modelId":"Qwen/Qwen3.8-27B","providerModelId":"Qwen/Qwen3.8-27B"},"fireworks-ai":{"modelId":"deepseek-ai/DeepSeek-V4.1-Flash","providerModelId":"accounts/fireworks/models/deepseek-v4p1-flash"},"novita":{"modelId":"deepseek-ai/DeepSeek-V4.1-Flash-1017","providerModelId":"deepseek/deepseek-v4.1-flash-1017"},"nscale":{"modelId":"meta-llama/Llama-4-Scout-17B-16E-Instruct","providerModelId":"meta-llama/Llama-4-Scout-17B-16E-Instruct"},"ovhcloud":{"modelId":"Qwen/Qwen3.8-27B","providerModelId":"Qwen3.8-27B"},"publicai":{"modelId":"swiss-ai/Apertus-v1.5-70B","providerModelId":"swiss-ai/apertus-v1.5-70b"},"scaleway":{"modelId":"Qwen/Qwen3.6-35B-A3B","providerModelId":"qwen3.6-35b-a3b"},"together":{"modelId":"deepseek-ai/DeepSeek-V4.1-Flash-1017","providerModelId":"deepseek-ai/DeepSeek-V4.1-Flash-1017"},"zai-org":{"modelId":"zai-org/GLM-5.3-Flash","providerModelId":"glm-5.3-flash"}} }
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
    providersMapping={ {"deepinfra":{"modelId":"openai/whisper-large-v3","providerModelId":"openai/whisper-large-v3"},"fal-ai":{"modelId":"openai/whisper-large-v3","providerModelId":"fal-ai/whisper"},"hf-inference":{"modelId":"openai/whisper-large-v3","providerModelId":"openai/whisper-large-v3"},"replicate":{"modelId":"openai/whisper-large-v3","providerModelId":"openai/whisper:8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e"},"together":{"modelId":"openai/whisper-large-v3","providerModelId":"openai/whisper-large-v3"}} }
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

### Image Classification
https://huggingface.co/docs/inference-providers/tasks/image-classification.md

## Image Classification

Image classification is the task of assigning a label or class to an entire image. Images are expected to have only one class for each image.

> [!TIP]
> For more details about the `image-classification` task, check out its [dedicated page](https://huggingface.co/tasks/image-classification)! You will find examples and related materials.

### Recommended models

- [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224): A strong image classification model.
- [facebook/deit-base-distilled-patch16-224](https://huggingface.co/facebook/deit-base-distilled-patch16-224): A robust image classification model.

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
    providersMapping={ {"hf-inference":{"modelId":"NYTK/named-entity-recognition-nerkor-hubert-hungarian","providerModelId":"NYTK/named-entity-recognition-nerkor-hubert-hungarian"}} }
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
    providersMapping={ {"featherless-ai":{"modelId":"zai-org/GLM-5.3","providerModelId":"zai-org/GLM-5.3"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"featherless-ai":{"modelId":"Qwen/Qwen3.8-27B","providerModelId":"Qwen/Qwen3.8-27B"} } }
conversational />

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
    providersMapping={ {"cohere":{"modelId":"CohereLabs/tiny-aya-global","providerModelId":"tiny-aya-global"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"cohere":{"modelId":"CohereLabs/aya-vision-32b","providerModelId":"c4ai-aya-vision-32b"} } }
conversational />

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
    providersMapping={ {"hf-inference":{"modelId":"perplexity-ai/pplx-embed-v2-late-0.6b","providerModelId":"perplexity-ai/pplx-embed-v2-late-0.6b"} } }
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
    providersMapping={ {"hf-inference":{"modelId":"nvidia/segformer-b2-finetuned-ade-512-512","providerModelId":"nvidia/segformer-b2-finetuned-ade-512-512"} } }
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
    providersMapping={ {"hf-inference":{"modelId":"facebook/bart-large-cnn","providerModelId":"facebook/bart-large-cnn"} } }
/>

### Table Question Answering

Find out more about Table Question Answering [here](../tasks/table-question-answering).

<InferenceSnippet
    pipeline=table-question-answering
    providersMapping={ {"hf-inference":{"modelId":"google/tapas-large-finetuned-wtq","providerModelId":"google/tapas-large-finetuned-wtq"} } }
/>

### Text Classification

Find out more about Text Classification [here](../tasks/text-classification).

<InferenceSnippet
    pipeline=text-classification
    providersMapping={ {"hf-inference":{"modelId":"meta-llama/Llama-Prompt-Guard-2-86M","providerModelId":"meta-llama/Llama-Prompt-Guard-2-86M"} } }
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
    providersMapping={ {"hf-inference":{"modelId":"NYTK/named-entity-recognition-nerkor-hubert-hungarian","providerModelId":"NYTK/named-entity-recognition-nerkor-hubert-hungarian"} } }
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
    providersMapping={ {"hf-inference":{"modelId":"jeff-legacy/Jeff-Qwen3.5-0.8B","providerModelId":"jeff-legacy/Jeff-Qwen3.5-0.8B"} } }
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
    providersMapping={ {"wavespeed":{"modelId":"black-forest-labs/FLUX.2-klein-9B","providerModelId":"wavespeed-ai/flux-2-klein-9b/edit"} } }
/>

### Text To Image

Find out more about Text To Image [here](../tasks/text-to-image).

<InferenceSnippet
    pipeline=text-to-image
    providersMapping={ {"wavespeed":{"modelId":"krea/Krea-2-Turbo","providerModelId":"wavespeed-ai/krea-v2-medium-turbo/text-to-image"} } }
/>

### Text To Video

Find out more about Text To Video [here](../tasks/text-to-video).

<InferenceSnippet
    pipeline=text-to-video
    providersMapping={ {"wavespeed":{"modelId":"ZhengmingYu/DMAD","providerModelId":"wavespeed-ai/minimax-h3/text-to-video-lora"} } }
/>

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
    providersMapping={ {"deepinfra":{"modelId":"zai-org/GLM-5.3","providerModelId":"zai-org/GLM-5.3"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"deepinfra":{"modelId":"Qwen/Qwen3.8-27B","providerModelId":"Qwen/Qwen3.8-27B"} } }
conversational />

### Feature Extraction

Find out more about Feature Extraction [here](../tasks/feature-extraction).

<InferenceSnippet
    pipeline=feature-extraction
    providersMapping={ {"deepinfra":{"modelId":"Qwen/Qwen3-Embedding-0.6B","providerModelId":"Qwen/Qwen3-Embedding-0.6B"} } }
/>

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
    providersMapping={ {"groq":{"modelId":"openai/gpt-oss-120b","providerModelId":"openai/gpt-oss-120b"} } }
conversational />

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
    providersMapping={ {"ovhcloud":{"modelId":"openai/gpt-oss-120b","providerModelId":"gpt-oss-120b"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"ovhcloud":{"modelId":"Qwen/Qwen3.8-27B","providerModelId":"Qwen3.8-27B"} } }
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
    providersMapping={ {"replicate":{"modelId":"black-forest-labs/FLUX.2-klein-9B","providerModelId":"black-forest-labs/flux-2-klein-9b"} } }
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
    providersMapping={ {"cerebras":{"modelId":"Qwen/Qwen3.8-27B","providerModelId":"qwen-3.8-27b"} } }
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
    providersMapping={ {"baseten":{"modelId":"zai-org/GLM-5.3","providerModelId":"zai-org/GLM-5.3"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"baseten":{"modelId":"deepseek-ai/DeepSeek-V4.1-Flash-1017","providerModelId":"deepseek-ai/DeepSeek-V4.1-Flash-1017"} } }
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

    
        
        
    

    
        
        
    

[fal](https://fal.ai/) is the generative media inference platform: call 1,000+ models by API, or deploy your own models, pipelines and apps on fal's autoscaling GPUs with [fal Serverless](https://fal.ai/docs/documentation/serverless). Serverless deployments scale from zero to thousands of runners and back, are billed per second of runner lifetime ([pricing](https://fal.ai/docs/documentation/serverless/pricing)), and can be deployed from a Python `fal.App`, a custom Docker container, or an existing HTTP server.

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
    providersMapping={ {"fal-ai":{"modelId":"black-forest-labs/FLUX.2-klein-9B","providerModelId":"fal-ai/flux-2/klein/9b/edit"} } }
/>

### Text To Image

Find out more about Text To Image [here](../tasks/text-to-image).

<InferenceSnippet
    pipeline=text-to-image
    providersMapping={ {"fal-ai":{"modelId":"krea/Krea-2-Turbo","providerModelId":"fal-ai/krea-2/turbo"} } }
/>

### Text To Video

Find out more about Text To Video [here](../tasks/text-to-video).

<InferenceSnippet
    pipeline=text-to-video
    providersMapping={ {"fal-ai":{"modelId":"Wan-AI/Wan2.2-TI2V-5B","providerModelId":"fal-ai/wan/v2.2-5b/text-to-video"} } }
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
    providersMapping={ {"zai-org":{"modelId":"zai-org/GLM-5.3","providerModelId":"glm-5.3"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"zai-org":{"modelId":"zai-org/GLM-5.3-Flash","providerModelId":"glm-5.3-flash"} } }
conversational />

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
    providersMapping={ {"scaleway":{"modelId":"openai/gpt-oss-120b","providerModelId":"gpt-oss-120b"} } }
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
    providersMapping={ {"fireworks-ai":{"modelId":"zai-org/GLM-5.3","providerModelId":"accounts/fireworks/models/glm-5p3"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"fireworks-ai":{"modelId":"deepseek-ai/DeepSeek-V4.1-Flash","providerModelId":"accounts/fireworks/models/deepseek-v4p1-flash"} } }
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
    providersMapping={ {"together":{"modelId":"zai-org/GLM-5.3","providerModelId":"zai-org/GLM-5.3"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"together":{"modelId":"deepseek-ai/DeepSeek-V4.1-Flash-1017","providerModelId":"deepseek-ai/DeepSeek-V4.1-Flash-1017"} } }
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
    providersMapping={ {"novita":{"modelId":"XiaomiMiMo/MiMo-V2.6-Pro-RL","providerModelId":"xiaomimimo/mimo-v2.6-pro"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"novita":{"modelId":"deepseek-ai/DeepSeek-V4.1-Flash-1017","providerModelId":"deepseek/deepseek-v4.1-flash-1017"} } }
conversational />
