# LFM3-VL

## Overview

LFM3-VL is the vision-language model family built on [LFM3](./lfm3)
by [Liquid AI](https://liquid.ai/). It takes images and videos at their native resolution together with text.

## Architecture

LFM3-VL consists of a vision encoder, a projector and an LFM3 language model. The vision encoder is itself a Liquid
hybrid: a linear patchifier followed by gated depthwise 2D convolution layers and bidirectional grouped-query attention
layers with 2D rotary embeddings over the patch grid of each frame. Images are split into 16x16 patches with the
dynamic-resolution preprocessing of Qwen2-VL, videos additionally pair consecutive frames and prefix each frame with its
timestamp. The projector merges every 2x2 block of patch features into one language token with a layer norm and a
two-layer MLP.

## Example

The following example shows how to generate an answer using the `AutoModelForImageTextToText` class.

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

# Load model and processor
model_id = "LiquidAI/LFM3-VL-2B"
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

# Load image and create conversation
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://www.ilankelman.org/stopsigns/australia.jpg"},
            {"type": "text", "text": "What is in this image?"},
        ],
    },
]

# Generate answer
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    tokenize=True,
).to(model.device, model.dtype)

outputs = model.generate(**inputs, max_new_tokens=64)
processor.batch_decode(outputs, skip_special_tokens=True)[0]
```

## Lfm3VlProcessor[[transformers.Lfm3VlProcessor]]

#### transformers.Lfm3VlProcessor[[transformers.Lfm3VlProcessor]]

```python
transformers.Lfm3VlProcessor(image_processor = None, tokenizer = None, video_processor = None, chat_template = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/lfm3_vl/processing_lfm3_vl.py#L30)

**Parameters:**

image_processor (`Qwen2VLImageProcessor`) : The image processor is a required input.

tokenizer (`tokenizer_class`) : The tokenizer is a required input.

video_processor (`Qwen3VLVideoProcessor`) : The video processor is a required input.

chat_template (`str`) : A Jinja template to convert lists of messages in a chat into a tokenizable string.

Constructs a Lfm3VlProcessor which wraps a image processor, a tokenizer, and a video processor into a single processor.

[Lfm3VlProcessor](/docs/transformers/v5.23.0/en/model_doc/lfm3_vl#transformers.Lfm3VlProcessor) offers all the functionalities of [Qwen2VLImageProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor), `tokenizer_class`, and [Qwen3VLVideoProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen3_vl#transformers.Qwen3VLVideoProcessor). See the
[~Qwen2VLImageProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor), `~tokenizer_class`, and [~Qwen3VLVideoProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen3_vl#transformers.Qwen3VLVideoProcessor) for more information.

#### post_process_image_text_to_text[[transformers.Lfm3VlProcessor.post_process_image_text_to_text]]

```python
post_process_image_text_to_text(generated_outputs, skip_special_tokens = True, clean_up_tokenization_spaces = False, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/lfm3_vl/processing_lfm3_vl.py#L140)

**Parameters:**

generated_outputs (`torch.Tensor` or `np.ndarray`) : The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)` or `(sequence_length,)`.

skip_special_tokens (`bool`, *optional*, defaults to `True`) : Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.

clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`) : Whether or not to clean up the tokenization spaces. Argument passed to the tokenizer's `batch_decode` method.

- ****kwargs** : Additional arguments to be passed to the tokenizer's `batch_decode method`.

**Returns:** `list[str]`

The decoded text.

Post-process the output of the model to decode the text.

## Lfm3VlConfig[[transformers.Lfm3VlConfig]]

#### transformers.Lfm3VlConfig[[transformers.Lfm3VlConfig]]

```python
transformers.Lfm3VlConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vision_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, text_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, image_token_id: int = 125030, video_token_id: int = 125029, vision_start_token_id: int = 125009, vision_end_token_id: int = 125010, projector_hidden_act: str = 'gelu', projector_hidden_size: int = 4096, projector_bias: bool = True, tie_word_embeddings: bool = True)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/lfm3_vl/configuration_lfm3_vl.py#L63)

**Parameters:**

vision_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the vision backbone.

text_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the text backbone.

image_token_id (`int`, *optional*, defaults to `125030`) : The image token index used as a placeholder for input images.

video_token_id (`int`, *optional*, defaults to `125029`) : The video token index used as a placeholder for input videos.

vision_start_token_id (`int`, *optional*, defaults to `125009`) : Token ID that marks the start of a visual segment in the multimodal input sequence.

vision_end_token_id (`int`, *optional*, defaults to `125010`) : Token ID that marks the end of a visual segment in the multimodal input sequence.

projector_hidden_act (`str`, *optional*, defaults to `gelu`) : The activation function used by the multimodal projector.

projector_hidden_size (`int`, *optional*, defaults to `4096`) : Dimensionality of text and vision projection layers.

projector_bias (`bool`, *optional*, defaults to `True`) : Whether to use bias in the multimodal projector.

tie_word_embeddings (`bool`, *optional*, defaults to `True`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

This is the configuration class to store the configuration of a Lfm3VlModel. It is used to instantiate a Lfm3 Vl
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [LiquidAI/LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Lfm3VlVisionConfig[[transformers.Lfm3VlVisionConfig]]

#### transformers.Lfm3VlVisionConfig[[transformers.Lfm3VlVisionConfig]]

```python
transformers.Lfm3VlVisionConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, hidden_size: int = 1024, intermediate_size: int = 2560, in_channels: int = 3, patch_size: int | list[int] | tuple[int, int] = 16, spatial_merge_size: int = 2, temporal_patch_size: int | list[int] | tuple[int, int] = 2, initializer_range: float = 0.02, rope_parameters: dict | None = None, num_hidden_layers: int = 14, num_attention_heads: int = 16, num_key_value_heads: int = 8, layer_types: list[str] | None = None, conv_kernel_size: int = 3, norm_eps: float = 1e-05)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/lfm3_vl/configuration_lfm3_vl.py#L29)

**Parameters:**

hidden_size (`int`, *optional*, defaults to `1024`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `2560`) : Dimension of the MLP representations.

in_channels (`int`, *optional*, defaults to `3`) : The number of input channels.

patch_size (`Union[int, list[int], tuple[int, int]]`, *optional*, defaults to `16`) : The size (resolution) of each patch.

spatial_merge_size (`int`, *optional*, defaults to `2`) : The size of the spatial merge window used to reduce the number of visual tokens by merging neighboring patches.

temporal_patch_size (`Union[int, list[int], tuple[int, int]]`, *optional*, defaults to `2`) : Temporal patch size used in the 3D patch embedding for video inputs.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rope_parameters (`dict`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

num_hidden_layers (`int`, *optional*, defaults to `14`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `16`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*, defaults to `8`) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

layer_types (`list[str]`, *optional*) : A list that explicitly maps each layer index with its layer type. If not provided, it will be automatically generated based on config values.

conv_kernel_size (`int`, *optional*, defaults to 3) : Kernel size of the depthwise convolution over the patch grid of each frame.

norm_eps (`float`, *optional*, defaults to `1e-05`) : The epsilon used by the layer normalization layers.

This is the configuration class to store the configuration of a Lfm3VlModel. It is used to instantiate a Lfm3 Vl
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [LiquidAI/LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Lfm3VlVisionModel[[transformers.Lfm3VlVisionModel]]

#### transformers.Lfm3VlVisionModel[[transformers.Lfm3VlVisionModel]]

```python
transformers.Lfm3VlVisionModel(config: Lfm3VlVisionConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/lfm3_vl/modeling_lfm3_vl.py#L398)

**Parameters:**

config ([Lfm3VlVisionConfig](/docs/transformers/v5.23.0/en/model_doc/lfm3_vl#transformers.Lfm3VlVisionConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The LFM3-VL vision encoder: a stack of gated convolution and bidirectional attention layers over the patches of
each image or video frame, embedded by a linear patch embedding.

This model inherits from [PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Lfm3VlVisionModel.forward]]

```python
forward(pixel_values: Tensor, grid_thw: Tensor, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/lfm3_vl/modeling_lfm3_vl.py#L424)

**Parameters:**

pixel_values (`torch.Tensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`) : The flattened patches of every image or video, grouped by merge block.

grid_thw (`torch.LongTensor` of shape `(num_images_or_videos, 3)`) : The temporal, height and width dimensions of the patch grid of each image or video.

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.23.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.Tensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.23.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Lfm3VlConfig](/docs/transformers/v5.23.0/en/model_doc/lfm3_vl#transformers.Lfm3VlConfig)) and inputs.

The [Lfm3VlVisionModel](/docs/transformers/v5.23.0/en/model_doc/lfm3_vl#transformers.Lfm3VlVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.Tensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## Lfm3VlModel[[transformers.Lfm3VlModel]]

#### transformers.Lfm3VlModel[[transformers.Lfm3VlModel]]

```python
transformers.Lfm3VlModel(config: Lfm3VlConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/lfm3_vl/modeling_lfm3_vl.py#L521)

**Parameters:**

config ([Lfm3VlConfig](/docs/transformers/v5.23.0/en/model_doc/lfm3_vl#transformers.Lfm3VlConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Lfm3 Vl Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Lfm3VlModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, pixel_values: typing.Optional[torch.FloatTensor] = None, image_grid_thw: typing.Optional[torch.LongTensor] = None, pixel_values_videos: typing.Optional[torch.FloatTensor] = None, video_grid_thw: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/lfm3_vl/modeling_lfm3_vl.py#L602)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.23.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [Qwen2VLImageProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor). See [Qwen2VLImageProcessor.__call__()](/docs/transformers/v5.23.0/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([Lfm3VlProcessor](/docs/transformers/v5.23.0/en/model_doc/lfm3_vl#transformers.Lfm3VlProcessor) uses [Qwen2VLImageProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor) for processing images).

image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) : The temporal, height and width of feature shape of each image in LLM.

pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) : The tensors corresponding to the input video. Pixel values for videos can be obtained using [Qwen3VLVideoProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen3_vl#transformers.Qwen3VLVideoProcessor). See [Qwen3VLVideoProcessor.__call__()](/docs/transformers/v5.23.0/en/model_doc/molmo2#transformers.Molmo2VideoProcessor.__call__) for details ([Lfm3VlProcessor](/docs/transformers/v5.23.0/en/model_doc/lfm3_vl#transformers.Lfm3VlProcessor) uses [Qwen3VLVideoProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen3_vl#transformers.Qwen3VLVideoProcessor) for processing videos).

video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) : The temporal, height and width of feature shape of each video in LLM.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** `Lfm3VlModelOutputWithPast` or `tuple(torch.Tensor)`

A `Lfm3VlModelOutputWithPast` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Lfm3VlConfig](/docs/transformers/v5.23.0/en/model_doc/lfm3_vl#transformers.Lfm3VlConfig)) and inputs.

The [Lfm3VlModel](/docs/transformers/v5.23.0/en/model_doc/lfm3_vl#transformers.Lfm3VlModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_pooler_output** (`torch.FloatTensor`, *optional*) -- A `torch.FloatTensor` of size `(num_image_tokens, hidden_size)`.
  Image features produced by the vision encoder and projected to the language model hidden size.
- **video_pooler_output** (`torch.FloatTensor`, *optional*) -- A `torch.FloatTensor` of size `(num_video_tokens, hidden_size)`.
  Video features produced by the vision encoder and projected to the language model hidden size.

## Lfm3VlForConditionalGeneration[[transformers.Lfm3VlForConditionalGeneration]]

#### transformers.Lfm3VlForConditionalGeneration[[transformers.Lfm3VlForConditionalGeneration]]

```python
transformers.Lfm3VlForConditionalGeneration(config: Lfm3VlConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/lfm3_vl/modeling_lfm3_vl.py#L682)

**Parameters:**

config ([Lfm3VlConfig](/docs/transformers/v5.23.0/en/model_doc/lfm3_vl#transformers.Lfm3VlConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The LFM3_VL model which consists of a vision backbone and a language model.

This model inherits from [PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Lfm3VlForConditionalGeneration.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, pixel_values: typing.Optional[torch.FloatTensor] = None, image_grid_thw: typing.Optional[torch.LongTensor] = None, pixel_values_videos: typing.Optional[torch.FloatTensor] = None, video_grid_thw: typing.Optional[torch.LongTensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/lfm3_vl/modeling_lfm3_vl.py#L723)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.23.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [Qwen2VLImageProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor). See [Qwen2VLImageProcessor.__call__()](/docs/transformers/v5.23.0/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([Lfm3VlProcessor](/docs/transformers/v5.23.0/en/model_doc/lfm3_vl#transformers.Lfm3VlProcessor) uses [Qwen2VLImageProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor) for processing images).

image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) : The temporal, height and width of feature shape of each image in LLM.

pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) : The tensors corresponding to the input video. Pixel values for videos can be obtained using [Qwen3VLVideoProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen3_vl#transformers.Qwen3VLVideoProcessor). See [Qwen3VLVideoProcessor.__call__()](/docs/transformers/v5.23.0/en/model_doc/molmo2#transformers.Molmo2VideoProcessor.__call__) for details ([Lfm3VlProcessor](/docs/transformers/v5.23.0/en/model_doc/lfm3_vl#transformers.Lfm3VlProcessor) uses [Qwen3VLVideoProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen3_vl#transformers.Qwen3VLVideoProcessor) for processing videos).

video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) : The temporal, height and width of feature shape of each video in LLM.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** `Lfm3VlCausalLMOutputWithPast` or `tuple(torch.Tensor)`

A `Lfm3VlCausalLMOutputWithPast` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Lfm3VlConfig](/docs/transformers/v5.23.0/en/model_doc/lfm3_vl#transformers.Lfm3VlConfig)) and inputs.

The [Lfm3VlForConditionalGeneration](/docs/transformers/v5.23.0/en/model_doc/lfm3_vl#transformers.Lfm3VlForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_pooler_output** (`torch.FloatTensor`, *optional*) -- A `torch.FloatTensor` of size `(num_image_tokens, hidden_size)`.
  Image features produced by the vision encoder and projected to the language model hidden size.
- **video_pooler_output** (`torch.FloatTensor`, *optional*) -- A `torch.FloatTensor` of size `(num_video_tokens, hidden_size)`.
  Video features produced by the vision encoder and projected to the language model hidden size.

Example:

```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText

>>> model = AutoModelForImageTextToText.from_pretrained("LiquidAI/LFM3-VL-2B")
>>> processor = AutoProcessor.from_pretrained("LiquidAI/LFM3-VL-2B")

>>> conversation = [
...     {
...         "role": "user",
...         "content": [
...             {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
...             {"type": "text", "text": "What is in this image?"},
...         ],
...     },
... ]

>>> inputs = processor.apply_chat_template(
...     conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
... ).to(model.device, model.dtype)

>>> outputs = model.generate(**inputs, max_new_tokens=45)
>>> processor.batch_decode(outputs, skip_special_tokens=True)[0]
```

### SeamlessM4T-v2
https://huggingface.co/docs/transformers/v5.23.0/model_doc/seamless_m4t_v2.md
