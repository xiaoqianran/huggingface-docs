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
    providersMapping={ {"baseten":{"modelId":"moonshotai/Kimi-K3","providerModelId":"moonshotai/Kimi-K3"},"cerebras":{"modelId":"google/gemma-4-31B-it","providerModelId":"gemma-4-31b"},"cohere":{"modelId":"CohereLabs/aya-vision-32b","providerModelId":"c4ai-aya-vision-32b"},"deepinfra":{"modelId":"thinkingmachines/Inkling","providerModelId":"thinkingmachines/Inkling"},"featherless-ai":{"modelId":"moonshotai/Kimi-K3","providerModelId":"moonshotai/Kimi-K3"},"fireworks-ai":{"modelId":"moonshotai/Kimi-K3","providerModelId":"accounts/fireworks/models/kimi-k3"},"novita":{"modelId":"moonshotai/Kimi-K2.7-Code","providerModelId":"moonshotai/kimi-k2.7-code"},"nscale":{"modelId":"meta-llama/Llama-4-Scout-17B-16E-Instruct","providerModelId":"meta-llama/Llama-4-Scout-17B-16E-Instruct"},"ovhcloud":{"modelId":"Qwen/Qwen3.6-27B","providerModelId":"Qwen3.6-27B"},"publicai":{"modelId":"swiss-ai/Apertus-v1.5-70B","providerModelId":"swiss-ai/apertus-v1.5-70b"},"scaleway":{"modelId":"Qwen/Qwen3.6-35B-A3B","providerModelId":"qwen3.6-35b-a3b"},"together":{"modelId":"moonshotai/Kimi-K3","providerModelId":"moonshotai/Kimi-K3"},"zai-org":{"modelId":"zai-org/GLM-4.6V-Flash","providerModelId":"glm-4.6v-flash"}} }
conversational />

### API specification

For the API specification of conversational image-text-to-text models, please refer to the [Chat Completion API documentation](https://huggingface.co/docs/inference-providers/tasks/chat-completion#api-specification).

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
    providersMapping={ {"hf-inference":{"modelId":"facebook/bart-large-cnn","providerModelId":"facebook/bart-large-cnn"}} }
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

### MacWhisper
https://huggingface.co/docs/inference-providers/integrations/macwhisper.md
