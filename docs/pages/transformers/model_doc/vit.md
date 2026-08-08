# Vision Transformer (ViT)

[Vision Transformer (ViT)](https://huggingface.co/papers/2010.11929) is a transformer adapted for computer vision tasks. An image is split into smaller fixed-sized patches which are treated as a sequence of tokens, similar to words for NLP tasks. ViT requires less resources to pretrain compared to convolutional architectures and its performance on large datasets can be transferred to smaller downstream tasks.

You can find all the original ViT checkpoints under the [Google](https://huggingface.co/google?search_models=vit) organization.

> [!TIP]
> Click on the ViT models in the right sidebar for more examples of how to apply ViT to different computer vision tasks.

The example below demonstrates how to classify an image with [Pipeline](/docs/transformers/v5.14.0/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v5.14.0/en/model_doc/auto#transformers.AutoModel) class.

```python
from transformers import pipeline

pipeline = pipeline(
    task="image-classification",
    model="google/vit-base-patch16-224",
    device=0
)
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

```python
import requests
import torch
from PIL import Image

from transformers import AutoImageProcessor, AutoModelForImageClassification

image_processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224",
    use_fast=True,
)
model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    device_map="auto",
    attn_implementation="sdpa"
)
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = image_processor(image, return_tensors="pt").to(model.device)

with torch.no_grad():
  logits = model(**inputs).logits
predicted_class_id = logits.argmax(dim=-1).item()

class_labels = model.config.id2label
predicted_class_label = class_labels[predicted_class_id]
print(f"The predicted class label is: {predicted_class_label}")
```

## Notes

- The best results are obtained with supervised pretraining, and during fine-tuning, it may be better to use images with a resolution higher than 224x224.
- Use [ViTImageProcessor](/docs/transformers/v5.14.0/en/model_doc/vit#transformers.ViTImageProcessor) to resize (or rescale) and normalize images to the expected size.
- The patch and image resolution are reflected in the checkpoint name. For example, google/vit-base-patch16-224, is the **base-sized** architecture with a patch resolution of 16x16 and fine-tuning resolution of 224x224.

## ViTConfig[[transformers.ViTConfig]]

- **hidden_size** (`int`, *optional*, defaults to `768`) --
  Dimension of the hidden representations.
- **num_hidden_layers** (`int`, *optional*, defaults to `12`) --
  Number of hidden layers in the Transformer decoder.
- **num_attention_heads** (`int`, *optional*, defaults to `12`) --
  Number of attention heads for each attention layer in the Transformer decoder.
- **intermediate_size** (`int`, *optional*, defaults to `3072`) --
  Dimension of the MLP representations.
- **hidden_act** (`str`, *optional*, defaults to `gelu`) --
  The non-linear activation function (function or string) in the decoder. For example, `"gelu"`,
  `"relu"`, `"silu"`, etc.
- **hidden_dropout_prob** (`Union[float, int]`, *optional*, defaults to `0.0`) --
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
- **attention_probs_dropout_prob** (`Union[float, int]`, *optional*, defaults to `0.0`) --
  The dropout ratio for the attention probabilities.
- **initializer_range** (`float`, *optional*, defaults to `0.02`) --
  The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
- **layer_norm_eps** (`float`, *optional*, defaults to `1e-12`) --
  The epsilon used by the layer normalization layers.
- **image_size** (`Union[int, list[int], tuple[int, int]]`, *optional*, defaults to `224`) --
  The size (resolution) of each image.
- **patch_size** (`Union[int, list[int], tuple[int, int]]`, *optional*, defaults to `16`) --
  The size (resolution) of each patch.
- **num_channels** (`int`, *optional*, defaults to `3`) --
  The number of input channels.
- **qkv_bias** (`bool`, *optional*, defaults to `True`) --
  Whether to add a bias to the queries, keys and values.
- **encoder_stride** (`int`, *optional*, defaults to 16) --
  Factor to increase the spatial resolution by in the decoder head for masked image modeling.
- **pooler_output_size** (`int`, *optional*) --
  Dimensionality of the pooler layer. If None, defaults to `hidden_size`.
- **pooler_act** (`str`, *optional*, defaults to `"tanh"`) --
  The activation function to be used by the pooler.

This is the configuration class to store the configuration of a ViTModel. It is used to instantiate a Vit
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.14.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.14.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import ViTConfig, ViTModel

>>> # Initializing a ViT vit-base-patch16-224 style configuration
>>> configuration = ViTConfig()

>>> # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
>>> model = ViTModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## ViTImageProcessor[[transformers.ViTImageProcessor]]

- **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]`) --
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
- **return_tensors** (`str` or [TensorType](/docs/transformers/v5.14.0/en/internal/file_utils#transformers.TensorType), *optional*) --
  Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors.
- ****kwargs** ([ImagesKwargs](/docs/transformers/v5.14.0/en/main_classes/processors#transformers.ImagesKwargs), *optional*) --
  Additional image preprocessing options. Model-specific kwargs are listed above; see the TypedDict class
  for the complete list of supported arguments.`~image_processing_base.BatchFeature`- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

## ViTImageProcessorPil[[transformers.ViTImageProcessorPil]]

- **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]`) --
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
- **return_tensors** (`str` or [TensorType](/docs/transformers/v5.14.0/en/internal/file_utils#transformers.TensorType), *optional*) --
  Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors.
- ****kwargs** ([ImagesKwargs](/docs/transformers/v5.14.0/en/main_classes/processors#transformers.ImagesKwargs), *optional*) --
  Additional image preprocessing options. Model-specific kwargs are listed above; see the TypedDict class
  for the complete list of supported arguments.`~image_processing_base.BatchFeature`- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

## ViTModel[[transformers.ViTModel]]

- **config** ([ViTConfig](/docs/transformers/v5.14.0/en/model_doc/vit#transformers.ViTConfig)) --
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from_pretrained()](/docs/transformers/v5.14.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
- **add_pooling_layer** (`bool`, *optional*, defaults to `True`) --
  Whether to add a pooling layer
- **use_mask_token** (`bool`, *optional*, defaults to `False`) --
  Whether to use a mask token for masked image modeling.

The bare Vit Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.14.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

- **pixel_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ViTImageProcessor](/docs/transformers/v5.14.0/en/model_doc/vit#transformers.ViTImageProcessor). See `ViTImageProcessor.__call__()` for details (`processor_class` uses
  [ViTImageProcessor](/docs/transformers/v5.14.0/en/model_doc/vit#transformers.ViTImageProcessor) for processing images).
- **bool_masked_pos** (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*) --
  Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
- **interpolate_pos_encoding** (`bool`, *optional*) --
  Whether to interpolate the pre-trained position encodings.
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)[BaseModelOutputWithPooling](/docs/transformers/v5.14.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`A [BaseModelOutputWithPooling](/docs/transformers/v5.14.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ViTConfig](/docs/transformers/v5.14.0/en/model_doc/vit#transformers.ViTConfig)) and inputs.
The [ViTModel](/docs/transformers/v5.14.0/en/model_doc/vit#transformers.ViTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Example:

```python
```

## ViTForMaskedImageModeling[[transformers.ViTForMaskedImageModeling]]

- **config** ([ViTConfig](/docs/transformers/v5.14.0/en/model_doc/vit#transformers.ViTConfig)) --
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from_pretrained()](/docs/transformers/v5.14.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

ViT Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://huggingface.co/papers/2111.09886).

Note that we provide a script to pre-train this model on custom data in our [examples
directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

This model inherits from [PreTrainedModel](/docs/transformers/v5.14.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

- **pixel_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ViTImageProcessor](/docs/transformers/v5.14.0/en/model_doc/vit#transformers.ViTImageProcessor). See `ViTImageProcessor.__call__()` for details (`processor_class` uses
  [ViTImageProcessor](/docs/transformers/v5.14.0/en/model_doc/vit#transformers.ViTImageProcessor) for processing images).
- **bool_masked_pos** (`torch.BoolTensor` of shape `(batch_size, num_patches)`) --
  Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
- **interpolate_pos_encoding** (`bool`, *optional*) --
  Whether to interpolate the pre-trained position encodings.
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)`MaskedImageModelingOutput` or `tuple(torch.FloatTensor)`A `MaskedImageModelingOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ViTConfig](/docs/transformers/v5.14.0/en/model_doc/vit#transformers.ViTConfig)) and inputs.
The [ViTForMaskedImageModeling](/docs/transformers/v5.14.0/en/model_doc/vit#transformers.ViTForMaskedImageModeling) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `bool_masked_pos` is provided) -- Reconstruction loss.
- **reconstruction** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) -- Reconstructed / completed images.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
  when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
  (also called feature maps) of the model at the output of each stage.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when
  `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.

Examples:
```python
>>> from transformers import AutoImageProcessor, ViTForMaskedImageModeling
>>> import torch
>>> from PIL import Image
>>> import httpx
>>> from io import BytesIO

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> with httpx.stream("GET", url) as response:
...     image = Image.open(BytesIO(response.read()))

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
>>> model = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k")

>>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
>>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
>>> # create random boolean mask of shape (batch_size, num_patches)
>>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

>>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
>>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
>>> list(reconstructed_pixel_values.shape)
[1, 3, 224, 224]
```

## ViTForImageClassification[[transformers.ViTForImageClassification]]

- **config** ([ViTConfig](/docs/transformers/v5.14.0/en/model_doc/vit#transformers.ViTConfig)) --
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from_pretrained()](/docs/transformers/v5.14.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
the [CLS] token) e.g. for ImageNet.

Note that it's possible to fine-tune ViT on higher resolution images than the ones it has been trained on, by
setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
position embeddings to the higher resolution.

This model inherits from [PreTrainedModel](/docs/transformers/v5.14.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

- **pixel_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ViTImageProcessor](/docs/transformers/v5.14.0/en/model_doc/vit#transformers.ViTImageProcessor). See `ViTImageProcessor.__call__()` for details (`processor_class` uses
  [ViTImageProcessor](/docs/transformers/v5.14.0/en/model_doc/vit#transformers.ViTImageProcessor) for processing images).
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
  config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
- **interpolate_pos_encoding** (`bool`, *optional*) --
  Whether to interpolate the pre-trained position encodings.
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)[ImageClassifierOutput](/docs/transformers/v5.14.0/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`A [ImageClassifierOutput](/docs/transformers/v5.14.0/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ViTConfig](/docs/transformers/v5.14.0/en/model_doc/vit#transformers.ViTConfig)) and inputs.
The [ViTForImageClassification](/docs/transformers/v5.14.0/en/model_doc/vit#transformers.ViTForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
  (also called feature maps) of the model at the output of each stage.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Example:

```python
>>> from transformers import AutoImageProcessor, ViTForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
>>> model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
...
```

### Overview
https://huggingface.co/docs/transformers/v5.14.0/model_doc/qwen3_next.md

## Overview

The Qwen3-Next series represents our next-generation foundation models, optimized for extreme context length and large-scale parameter efficiency.
The series introduces a suite of architectural innovations designed to maximize performance while minimizing computational cost:

- **Hybrid Attention**: Replaces standard attention with the combination of **Gated DeltaNet** and **Gated Attention**, enabling efficient context modeling.
- **High-Sparsity MoE**: Achieves an extreme low activation ratio as 1:50 in MoE layers — drastically reducing FLOPs per token while preserving model capacity.
- **Multi-Token Prediction(MTP)**: Boosts pretraining model performance, and accelerates inference.
- **Other Optimizations**: Includes techniques such as **zero-centered and weight-decayed layernorm**, **Gated Attention**, and other stabilizing enhancements for robust training.

Built on this architecture, we trained and open-sourced Qwen3-Next-80B-A3B — 80B total parameters, only 3B active — achieving extreme sparsity and efficiency.

Despite its ultra-efficiency, it outperforms Qwen3-32B on downstream tasks — while requiring **less than 1/10 of the training cost**.
Moreover, it delivers over **10x higher inference throughput** than Qwen3-32B when handling contexts longer than 32K tokens.

For more details, please visit our blog [Qwen3-Next](qwen3_next) ([blog post](https://qwenlm.github.io/blog/qwen3_next/)).

## Usage examples

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-Next-80B-A3B-Instruct"

<CopyLLMTxtMenu containerStyle="float: right; margin-left: 10px; display: inline-flex; position: relative; z-index: 10;"></CopyLLMTxtMenu>

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("content:", content)
```

## Qwen3NextConfig[[transformers.Qwen3NextConfig]]

- **vocab_size** (`int`, *optional*, defaults to `151936`) --
  Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.
- **hidden_size** (`int`, *optional*, defaults to `2048`) --
  Dimension of the hidden representations.
- **intermediate_size** (`int`, *optional*, defaults to `5632`) --
  Dimension of the MLP representations.
- **num_hidden_layers** (`int`, *optional*, defaults to `48`) --
  Number of hidden layers in the Transformer decoder.
- **num_attention_heads** (`int`, *optional*, defaults to `16`) --
  Number of attention heads for each attention layer in the Transformer decoder.
- **num_key_value_heads** (`int`, *optional*, defaults to `2`) --
  This is the number of key_value heads that should be used to implement Grouped Query Attention. If
  `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
  converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
  by meanpooling all the original heads within that group. For more details, check out [this
  paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
  `num_attention_heads`.
- **hidden_act** (`str`, *optional*, defaults to `silu`) --
  The non-linear activation function (function or string) in the decoder. For example, `"gelu"`,
  `"relu"`, `"silu"`, etc.
- **max_position_embeddings** (`int`, *optional*, defaults to `32768`) --
  The maximum sequence length that this model might ever be used with.
- **initializer_range** (`float`, *optional*, defaults to `0.02`) --
  The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
- **rms_norm_eps** (`float`, *optional*, defaults to `1e-06`) --
  The epsilon used by the rms normalization layers.
- **use_cache** (`bool`, *optional*, defaults to `True`) --
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.
- **tie_word_embeddings** (`bool`, *optional*, defaults to `False`) --
  Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.
- **rope_parameters** (`Union[~modeling_rope_utils.RopeParameters, dict]`, *optional*) --
  Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
  a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
  with longer `max_position_embeddings`.
- **attention_bias** (`bool`, *optional*, defaults to `False`) --
  Whether to use a bias in the query, key, value and output projection layers during self-attention.
- **attention_dropout** (`Union[float, int]`, *optional*, defaults to `0.0`) --
  The dropout ratio for the attention probabilities.
- **head_dim** (`int`, *optional*, defaults to `256`) --
  The attention head dimension. If None, it will default to hidden_size // num_attention_heads
- **linear_conv_kernel_dim** (`int`, *optional*, defaults to 4) --
  Kernel size of the convolution used in linear attention layers.
- **linear_key_head_dim** (`int`, *optional*, defaults to 128) --
  Dimension of each key head in linear attention.
- **linear_value_head_dim** (`int`, *optional*, defaults to 128) --
  Dimension of each value head in linear attention.
- **linear_num_key_heads** (`int`, *optional*, defaults to 16) --
  Number of key heads used in linear attention layers.
- **linear_num_value_heads** (`int`, *optional*, defaults to 32) --
  Number of value heads used in linear attention layers.
- **decoder_sparse_step** (`int`, *optional*, defaults to 1) --
  The frequency of the MoE layer.
- **moe_intermediate_size** (`int`, *optional*, defaults to `512`) --
  Intermediate size of the routed expert MLPs.
- **shared_expert_intermediate_size** (`int`, *optional*, defaults to `512`) --
  Intermediate size of the shared expert MLPs.
- **num_experts_per_tok** (`int`, *optional*, defaults to `10`) --
  Number of experts to route each token to. This is the top-k value for the token-choice routing.
- **num_experts** (`int`, *optional*, defaults to `512`) --
  Number of routed experts in MoE layers.

- **norm_topk_prob** (`bool`, *optional*, defaults to `True`) --
  Whether to normalize the weights of the routed experts.

- **output_router_logits** (`bool`, *optional*, defaults to `False`) --
  Whether or not the router logits should be returned by the model. Enabling this will also allow the model
  to output the auxiliary loss, including load balancing loss and router z-loss.
- **router_aux_loss_coef** (`float`, *optional*, defaults to `0.001`) --
  Auxiliary load balancing loss coefficient. Used to penalize uneven expert routing in MoE models.
- **mlp_only_layers** (`list[int]`, *optional*, defaults to `[]`) --
  Indicate which layers use Qwen3NextMLP rather than Qwen3NextSparseMoeBlock
  The list contains layer index, from 0 to num_layers-1 if we have num_layers layers
  If `mlp_only_layers` is empty, `decoder_sparse_step` is used to determine the sparsity.
- **layer_types** (`list[str]`, *optional*) --
  A list that explicitly maps each layer index with its layer type. If not provided, it will be automatically
  generated based on config values.
- **pad_token_id** (`int`, *optional*) --
  Token id used for padding in the vocabulary.
- **bos_token_id** (`int`, *optional*) --
  Token id used for beginning-of-stream in the vocabulary.
- **eos_token_id** (`Union[int, list[int]]`, *optional*) --
  Token id used for end-of-stream in the vocabulary.

This is the configuration class to store the configuration of a Qwen3NextModel. It is used to instantiate a Qwen3 Next
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.14.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.14.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

```python
>>> from transformers import Qwen3NextModel, Qwen3NextConfig

>>> # Initializing a Qwen3Next style configuration
>>> configuration =  Qwen3NextConfig()

>>> # Initializing a model from the Qwen3-Next-80B-A3B style configuration
>>> model = Qwen3NextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Qwen3NextModel[[transformers.Qwen3NextModel]]

- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.14.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.14.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/v5.14.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **past_key_values** (`~cache_utils.Cache`, *optional*) --
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v5.14.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.14.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).`MoeModelOutputWithPast` or `tuple(torch.FloatTensor)`A `MoeModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Qwen3NextConfig](/docs/transformers/v5.14.0/en/model_doc/qwen3_next#transformers.Qwen3NextConfig)) and inputs.
The [Qwen3NextModel](/docs/transformers/v5.14.0/en/model_doc/qwen3_next#transformers.Qwen3NextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.14.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **router_logits** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

  Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
  loss for Mixture of Experts models.

## Qwen3NextForCausalLM[[transformers.Qwen3NextForCausalLM]]

- **config** ([Qwen3NextForCausalLM](/docs/transformers/v5.14.0/en/model_doc/qwen3_next#transformers.Qwen3NextForCausalLM)) --
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from_pretrained()](/docs/transformers/v5.14.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Qwen3 Next Model for causal language modeling.

This model inherits from [PreTrainedModel](/docs/transformers/v5.14.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.14.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.14.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/v5.14.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **past_key_values** (`~cache_utils.Cache`, *optional*) --
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v5.14.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.14.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
  config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
- **output_router_logits** (`bool`, *optional*) --
  Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
  should not be returned during inference.
- **logits_to_keep** (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) --
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).`MoeCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`A `MoeCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Qwen3NextConfig](/docs/transformers/v5.14.0/en/model_doc/qwen3_next#transformers.Qwen3NextConfig)) and inputs.
The [Qwen3NextForCausalLM](/docs/transformers/v5.14.0/en/model_doc/qwen3_next#transformers.Qwen3NextForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **aux_loss** (`torch.FloatTensor`, *optional*, returned when `labels` is provided) -- aux_loss for the sparse modules.
- **router_logits** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

  Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
  loss for Mixture of Experts models.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.14.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Example:

```python
>>> from transformers import AutoTokenizer, Qwen3NextForCausalLM

>>> model = Qwen3NextForCausalLM.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")
>>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")

>>> prompt = "Hey, are you conscious? Can you talk to me?"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
```

## Qwen3NextForSequenceClassification[[transformers.Qwen3NextForSequenceClassification]]

- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.14.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.14.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/v5.14.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **past_key_values** (`Cache`, *optional*) --
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v5.14.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.14.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
  config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).`SequenceClassifierOutputWithPast`
The `GenericForSequenceClassification` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## Qwen3NextForQuestionAnswering[[transformers.Qwen3NextForQuestionAnswering]]

- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.14.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.14.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/v5.14.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **past_key_values** (`Cache`, *optional*) --
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v5.14.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.14.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **start_positions** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for position (index) of the start of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
- **end_positions** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for position (index) of the end of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.`QuestionAnsweringModelOutput`
The `GenericForQuestionAnswering` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## Qwen3NextForTokenClassification[[transformers.Qwen3NextForTokenClassification]]

- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.14.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.14.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/v5.14.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **past_key_values** (`Cache`, *optional*) --
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v5.14.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.14.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
  config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).`TokenClassifierOutput`
The `GenericForTokenClassification` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

### Janus
https://huggingface.co/docs/transformers/v5.14.0/model_doc/janus.md
