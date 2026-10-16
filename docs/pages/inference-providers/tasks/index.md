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
    providersMapping={ {"fal-ai":{"modelId":"black-forest-labs/FLUX.1-dev","providerModelId":"fal-ai/flux/dev"},"hf-inference":{"modelId":"stabilityai/stable-diffusion-3-medium-diffusers","providerModelId":"stabilityai/stable-diffusion-3-medium-diffusers"},"nscale":{"modelId":"black-forest-labs/FLUX.1-schnell","providerModelId":"black-forest-labs/FLUX.1-schnell"},"replicate":{"modelId":"black-forest-labs/FLUX.1-dev","providerModelId":"black-forest-labs/flux-dev"},"wavespeed":{"modelId":"black-forest-labs/FLUX.1-dev","providerModelId":"wavespeed-ai/flux-dev"}} }
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

### Audio Classification
https://huggingface.co/docs/inference-providers/tasks/audio-classification.md

## Audio Classification

Audio classification is the task of assigning a label or class to a given audio.

Example applications:
* Recognizing which command a user is giving
* Identifying a speaker
* Detecting the genre of a song

> [!TIP]
> For more details about the `audio-classification` task, check out its [dedicated page](https://huggingface.co/tasks/audio-classification)! You will find examples and related materials.

### Recommended models

Explore all available models and find the one that suits you best [here](https://huggingface.co/models?inference=warm&pipeline_tag=audio-classification&sort=trending), or from the terminal with the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-models-list):

```bash
hf models ls --warm --pipeline-tag audio-classification --sort trending_score
```

### Using the API

There are currently no snippet examples for the **audio-classification** task, as no providers support it yet.

### API specification

#### Request

| Headers |   |    |
| :--- | :--- | :--- |
| **authorization** | _string_ | Authentication header in the form `'Bearer: hf_****'` when `hf_****` is a personal user access token with "Inference Providers" permission. You can generate one from [your settings page](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained). |

| Payload |  |  |
| :--- | :--- | :--- |
| **inputs*** | _string_ | The input audio data as a base64-encoded string. If no `parameters` are provided, you can also provide the audio data as a raw bytes payload. |
| **parameters** | _object_ |  |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;function_to_apply** | _enum_ | Possible values: sigmoid, softmax, none. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;top_k** | _integer_ | When specified, limits the output to the top K most probable classes. |

#### Response

| Body |  |
| :--- | :--- | :--- |
| **(array)** | _object[]_ | Output is an array of objects. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;label** | _string_ | The predicted class label. |
| **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;score** | _number_ | The corresponding probability. |

### Pi
https://huggingface.co/docs/inference-providers/integrations/pi.md
