# UltraVBert

## Overview

UltraVBert extends [ModernVBert](./modernvbert) by pairing the [UltraBERT](./ultrabert) text encoder with a [SigLIP](./siglip) vision encoder for visual document understanding and retrieval. The image processor splits a page into 512-pixel tiles plus a global view. The connector merges each 4 x 4 block of vision patches into one token, so a tile becomes 64 image tokens, and RMS-normalizes them to the scale of UltraBERT's token embeddings. `max_image_tiles`, 27 by default, caps the tiles per page by lowering the resize edge in 512-pixel steps.

## Usage example

```python
import torch

from transformers import AutoModelForMaskedLM, AutoProcessor
from transformers.image_utils import load_image

processor = AutoProcessor.from_pretrained("athrael-soju/UltraVBERT")
model = AutoModelForMaskedLM.from_pretrained("athrael-soju/UltraVBERT", device_map="auto")

image = load_image(
    "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/receipt_00008.png"
)
inputs = processor(text="<image>Rounding 0 <|mask|> 50,000", images=[image], return_tensors="pt").to(model.device)

with torch.no_grad():
    logits = model(**inputs).logits

masked_index = torch.where(inputs["input_ids"] == processor.tokenizer.mask_token_id)
predicted_token_id = logits[masked_index].argmax(dim=-1)
print(processor.tokenizer.decode(predicted_token_id))
```

The `fill-mask` pipeline reads text only.

```python
from transformers import pipeline

fill_mask = pipeline(task="fill-mask", model="athrael-soju/UltraVBERT", device_map="auto")
print(fill_mask("Plants create <|mask|> through a process known as photosynthesis."))
```

## UltraVBertConfig[[transformers.UltraVBertConfig]]

#### transformers.UltraVBertConfig[[transformers.UltraVBertConfig]]

```python
transformers.UltraVBertConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, text_config: transformers.configuration_utils.PreTrainedConfig | dict | None = None, vision_config: transformers.configuration_utils.PreTrainedConfig | dict | None = None, image_token_id: int = 49191, pixel_shuffle_factor: int = 4, initializer_range: float = 0.02, initializer_cutoff_factor: float = 2.0, classifier_pooling: typing.Literal['cls', 'mean'] = 'cls', classifier_dropout: float | int = 0.0, tie_word_embeddings: bool = False)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/ultravbert/configuration_ultravbert.py#L32)

**Parameters:**

text_config (`Union[~configuration_utils.PreTrainedConfig, dict]`, *optional*) : The config object or dictionary of the text backbone.

vision_config (`Union[~configuration_utils.PreTrainedConfig, dict]`, *optional*) : The config object or dictionary of the vision backbone.

image_token_id (`int`, *optional*, defaults to `49191`) : The image token index used as a placeholder for input images.

pixel_shuffle_factor (`int`, *optional*, defaults to 4) : Side of the square of vision patches that the connector merges into one image token.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

initializer_cutoff_factor (`float`, *optional*, defaults to 2.0) : The cutoff factor for the truncated_normal_initializer for initializing all weight matrices.

classifier_pooling (`Literal["cls", "mean"]`, *optional*, defaults to `"cls"`) : The pooling strategy to use for classification tasks.

classifier_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for classifier.

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

This is the configuration class to store the configuration of a UltraVBertModel. It is used to instantiate a Ultravbert
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [athrael-soju/UltraVBERT](https://huggingface.co/athrael-soju/UltraVBERT)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## UltraVBertImageProcessor[[transformers.UltraVBertImageProcessor]]

#### transformers.UltraVBertImageProcessor[[transformers.UltraVBertImageProcessor]]

```python
transformers.UltraVBertImageProcessor(**kwargs: *<class 'transformers.models.ultravbert.image_processing_ultravbert.UltraVBertImageProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/ultravbert/image_processing_ultravbert.py#L216)

**Parameters:**

do_convert_rgb (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to convert the image to RGB.

do_resize (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to resize the image.

size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*, defaults to `{'longest_edge' : 3072}`): Describes the maximum input dimensions to the model.

default_to_square (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to default to a square image when resizing, if size is an int.

crop_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : Size of the output image after applying `center_crop`.

resample (`Annotated[Union[int, PILImageResampling, NoneType], None]`, *kwargs*, defaults to `1`) : Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.

do_rescale (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to rescale the image.

rescale_factor (`float`, *kwargs*, *optional*, defaults to `0.00392156862745098`) : Rescale factor to rescale the image by if `do_rescale` is set to `True`.

do_normalize (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to normalize the image.

image_mean (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*, defaults to `[0.5, 0.5, 0.5]`) : Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.

image_std (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*, defaults to `[0.5, 0.5, 0.5]`) : Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to `True`.

do_pad (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to pad the image. Padding is done either to the largest size in the batch or to a fixed square size per image. The exact padding strategy depends on the model.

pad_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : The size in `{"height": int, "width" int}` to pad the images to. Must be larger than any image size provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest height and width in the batch. Applied only when `do_pad=True.`

do_center_crop (`bool`, *kwargs*, *optional*) : Whether to center crop the image.

data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.

input_data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format. - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

device (`Annotated[Union[str, torch.device, NoneType], None]`, *kwargs*) : The device to process the videos on. If unset, the device is inferred from the input videos.

return_tensors (`Annotated[str | ~utils.generic.TensorType | None, None]`, *kwargs*) : Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors.

disable_grouping (`bool`, *kwargs*, *optional*) : Whether to disable grouping of images by size to process them individually and not in batches. If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157

image_seq_length (`int`, *kwargs*, *optional*) : The number of image tokens to be used for each image in the input. Added for backward compatibility but this should be set as a processor attribute in future models.

do_image_splitting (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to split the image into sub-images concatenated with the original image. They are split into patches such that each patch has a size of `max_image_size["height"]` x `max_image_size["width"]`.

max_image_size (`Dict`, *kwargs*, *optional*, defaults to `{"longest_edge" : 512}`): Maximum resolution of the patches of images accepted by the model. This is a dictionary containing the key "longest_edge".

return_row_col_info (`bool`, *kwargs*, *optional*, defaults to `False`) : Whether to return the row and column information of the images.

max_image_tiles (`int`, *kwargs*, *optional*, defaults to 27) : Largest number of tiles an image splits into, the global image excluded. An image that would split into more is resized to a longest edge lowered in steps of `max_image_size["longest_edge"]` until it fits.

Constructs a UltraVBertImageProcessor image processor.

#### preprocess[[transformers.UltraVBertImageProcessor.preprocess]]

```python
preprocess(images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']], **kwargs: *<class 'transformers.models.ultravbert.image_processing_ultravbert.UltraVBertImageProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/ultravbert/image_processing_ultravbert.py#L236)

**Parameters:**

images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]`) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

do_convert_rgb (`bool`, *kwargs*, *optional*) : Whether to convert the image to RGB.

do_resize (`bool`, *kwargs*, *optional*) : Whether to resize the image.

size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : Describes the maximum input dimensions to the model.

default_to_square (`bool`, *kwargs*, *optional*) : Whether to default to a square image when resizing, if size is an int.

crop_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : Size of the output image after applying `center_crop`.

resample (`Annotated[Union[int, PILImageResampling, NoneType], None]`, *kwargs*) : Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.

do_rescale (`bool`, *kwargs*, *optional*) : Whether to rescale the image.

rescale_factor (`float`, *kwargs*, *optional*) : Rescale factor to rescale the image by if `do_rescale` is set to `True`.

do_normalize (`bool`, *kwargs*, *optional*) : Whether to normalize the image.

image_mean (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*) : Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.

image_std (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*) : Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to `True`.

do_pad (`bool`, *kwargs*, *optional*) : Whether to pad the image. Padding is done either to the largest size in the batch or to a fixed square size per image. The exact padding strategy depends on the model.

pad_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : The size in `{"height": int, "width" int}` to pad the images to. Must be larger than any image size provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest height and width in the batch. Applied only when `do_pad=True.`

do_center_crop (`bool`, *kwargs*, *optional*) : Whether to center crop the image.

data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.

input_data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format. - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

device (`Annotated[Union[str, torch.device, NoneType], None]`, *kwargs*) : The device to process the videos on. If unset, the device is inferred from the input videos.

return_tensors (`Annotated[str | ~utils.generic.TensorType | None, None]`, *kwargs*) : Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors.

disable_grouping (`bool`, *kwargs*, *optional*) : Whether to disable grouping of images by size to process them individually and not in batches. If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157

image_seq_length (`int`, *kwargs*, *optional*) : The number of image tokens to be used for each image in the input. Added for backward compatibility but this should be set as a processor attribute in future models.

do_image_splitting (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to split the image into sub-images concatenated with the original image. They are split into patches such that each patch has a size of `max_image_size["height"]` x `max_image_size["width"]`.

max_image_size (`Dict`, *kwargs*, *optional*, defaults to `{"longest_edge" : 512}`): Maximum resolution of the patches of images accepted by the model. This is a dictionary containing the key "longest_edge".

return_row_col_info (`bool`, *kwargs*, *optional*, defaults to `False`) : Whether to return the row and column information of the images.

max_image_tiles (`int`, *kwargs*, *optional*, defaults to 27) : Largest number of tiles an image splits into, the global image excluded. An image that would split into more is resized to a longest edge lowered in steps of `max_image_size["longest_edge"]` until it fits.

**Returns:** `~image_processing_base.BatchFeature`

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

## UltraVBertModel[[transformers.UltraVBertModel]]

#### transformers.UltraVBertModel[[transformers.UltraVBertModel]]

```python
transformers.UltraVBertModel(config: UltraVBertConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/ultravbert/modeling_ultravbert.py#L222)

**Parameters:**

config ([UltraVBertConfig](/docs/transformers/v5.21.1/en/model_doc/ultravbert#transformers.UltraVBertConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Ultravbert Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.UltraVBertModel.forward]]

```python
forward(input_ids: LongTensor = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, pixel_values: typing.Optional[torch.FloatTensor] = None, pixel_attention_mask: typing.Optional[torch.BoolTensor] = None, image_hidden_states: typing.Optional[torch.FloatTensor] = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/ultravbert/modeling_ultravbert.py#L306)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.21.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [UltraVBertImageProcessor](/docs/transformers/v5.21.1/en/model_doc/ultravbert#transformers.UltraVBertImageProcessor). See [UltraVBertImageProcessor.__call__()](/docs/transformers/v5.21.1/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([Idefics3Processor](/docs/transformers/v5.21.1/en/model_doc/idefics3#transformers.Idefics3Processor) uses [UltraVBertImageProcessor](/docs/transformers/v5.21.1/en/model_doc/ultravbert#transformers.UltraVBertImageProcessor) for processing images).

pixel_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_images, image_size, image_size)`, *optional*) : Mask over the real pixels of each image tile.

image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The hidden states of the image encoder after modality projection.

**Returns:** `UltraVBertBaseModelOutput` or `tuple(torch.FloatTensor)`

A `UltraVBertBaseModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([UltraVBertConfig](/docs/transformers/v5.21.1/en/model_doc/ultravbert#transformers.UltraVBertConfig)) and inputs.

The [UltraVBertModel](/docs/transformers/v5.21.1/en/model_doc/ultravbert#transformers.UltraVBertModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.
  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_hidden_states** (`tuple(torch.FloatTensor)`, *optional*) -- Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
  sequence_length, hidden_size)`.
  image_hidden_states of the model produced by the vision encoder

## UltraVBertForMaskedLM[[transformers.UltraVBertForMaskedLM]]

#### transformers.UltraVBertForMaskedLM[[transformers.UltraVBertForMaskedLM]]

```python
transformers.UltraVBertForMaskedLM(config: UltraVBertConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/ultravbert/modeling_ultravbert.py#L396)

**Parameters:**

config ([UltraVBertConfig](/docs/transformers/v5.21.1/en/model_doc/ultravbert#transformers.UltraVBertConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Ultravbert Model with a `language modeling` head on top."

This model inherits from [PreTrainedModel](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.UltraVBertForMaskedLM.forward]]

```python
forward(input_ids: LongTensor = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, pixel_values: typing.Optional[torch.FloatTensor] = None, pixel_attention_mask: typing.Optional[torch.BoolTensor] = None, image_hidden_states: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.LongTensor] = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/ultravbert/modeling_ultravbert.py#L417)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.21.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [UltraVBertImageProcessor](/docs/transformers/v5.21.1/en/model_doc/ultravbert#transformers.UltraVBertImageProcessor). See [UltraVBertImageProcessor.__call__()](/docs/transformers/v5.21.1/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([Idefics3Processor](/docs/transformers/v5.21.1/en/model_doc/idefics3#transformers.Idefics3Processor) uses [UltraVBertImageProcessor](/docs/transformers/v5.21.1/en/model_doc/ultravbert#transformers.UltraVBertImageProcessor) for processing images).

pixel_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_images, image_size, image_size)`, *optional*) : Mask over the real pixels of each image tile.

image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The hidden states of the image encoder after modality projection.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

**Returns:** `UltraVBertMaskedLMOutput` or `tuple(torch.FloatTensor)`

A `UltraVBertMaskedLMOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([UltraVBertConfig](/docs/transformers/v5.21.1/en/model_doc/ultravbert#transformers.UltraVBertConfig)) and inputs.

The [UltraVBertForMaskedLM](/docs/transformers/v5.21.1/en/model_doc/ultravbert#transformers.UltraVBertForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor`, *optional*, returned when `labels` is provided) -- Masked language modeling (MLM) loss.
- **logits** (`torch.FloatTensor`, *optional*) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.
  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_hidden_states** (`tuple(torch.FloatTensor)`, *optional*) -- Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
  sequence_length, hidden_size)`.
  image_hidden_states of the model produced by the vision encoder

Example:

```python
>>> from transformers import AutoTokenizer, UltraVBertForMaskedLM
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("athrael-soju/UltraVBERT")
>>> model = UltraVBertForMaskedLM.from_pretrained("athrael-soju/UltraVBERT")

>>> inputs = tokenizer("The capital of France is <mask>.", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # retrieve index of <mask>
>>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

>>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
>>> tokenizer.decode(predicted_token_id)
...

>>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
>>> # mask labels of non-<mask> tokens
>>> labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

>>> outputs = model(**inputs, labels=labels)
>>> round(outputs.loss.item(), 2)
...
```

## UltraVBertForSequenceClassification[[transformers.UltraVBertForSequenceClassification]]

#### transformers.UltraVBertForSequenceClassification[[transformers.UltraVBertForSequenceClassification]]

```python
transformers.UltraVBertForSequenceClassification(config: UltraVBertConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/ultravbert/modeling_ultravbert.py#L471)

**Parameters:**

config ([UltraVBertConfig](/docs/transformers/v5.21.1/en/model_doc/ultravbert#transformers.UltraVBertConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The UltraVBert Model with a sequence classification head on top that performs pooling.

This model inherits from [PreTrainedModel](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.UltraVBertForSequenceClassification.forward]]

```python
forward(input_ids: LongTensor = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, pixel_values: typing.Optional[torch.FloatTensor] = None, pixel_attention_mask: typing.Optional[torch.BoolTensor] = None, image_hidden_states: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.LongTensor] = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/ultravbert/modeling_ultravbert.py#L485)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.21.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [UltraVBertImageProcessor](/docs/transformers/v5.21.1/en/model_doc/ultravbert#transformers.UltraVBertImageProcessor). See [UltraVBertImageProcessor.__call__()](/docs/transformers/v5.21.1/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([Idefics3Processor](/docs/transformers/v5.21.1/en/model_doc/idefics3#transformers.Idefics3Processor) uses [UltraVBertImageProcessor](/docs/transformers/v5.21.1/en/model_doc/ultravbert#transformers.UltraVBertImageProcessor) for processing images).

pixel_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_images, image_size, image_size)`, *optional*) : Mask over the real pixels of each image tile.

image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The hidden states of the image encoder after modality projection.

labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*) : Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

**Returns:** [SequenceClassifierOutput](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

A [SequenceClassifierOutput](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([UltraVBertConfig](/docs/transformers/v5.21.1/en/model_doc/ultravbert#transformers.UltraVBertConfig)) and inputs.

The [UltraVBertForSequenceClassification](/docs/transformers/v5.21.1/en/model_doc/ultravbert#transformers.UltraVBertForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Example of single-label classification:

```python
>>> import torch
>>> from transformers import AutoTokenizer, UltraVBertForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("athrael-soju/UltraVBERT")
>>> model = UltraVBertForSequenceClassification.from_pretrained("athrael-soju/UltraVBERT")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
...

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = UltraVBertForSequenceClassification.from_pretrained("athrael-soju/UltraVBERT", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
...
```

Example of multi-label classification:

```python
>>> import torch
>>> from transformers import AutoTokenizer, UltraVBertForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("athrael-soju/UltraVBERT")
>>> model = UltraVBertForSequenceClassification.from_pretrained("athrael-soju/UltraVBERT", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = UltraVBertForSequenceClassification.from_pretrained(
...     "athrael-soju/UltraVBERT", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## UltraVBertForTokenClassification[[transformers.UltraVBertForTokenClassification]]

#### transformers.UltraVBertForTokenClassification[[transformers.UltraVBertForTokenClassification]]

```python
transformers.UltraVBertForTokenClassification(config: UltraVBertConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/ultravbert/modeling_ultravbert.py#L572)

**Parameters:**

config ([UltraVBertConfig](/docs/transformers/v5.21.1/en/model_doc/ultravbert#transformers.UltraVBertConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The UltraVBert Model with a token classification head on top, e.g. for Named Entity Recognition (NER) tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.UltraVBertForTokenClassification.forward]]

```python
forward(input_ids: LongTensor = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, pixel_values: typing.Optional[torch.FloatTensor] = None, pixel_attention_mask: typing.Optional[torch.BoolTensor] = None, image_hidden_states: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.LongTensor] = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/ultravbert/modeling_ultravbert.py#L585)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.21.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [UltraVBertImageProcessor](/docs/transformers/v5.21.1/en/model_doc/ultravbert#transformers.UltraVBertImageProcessor). See [UltraVBertImageProcessor.__call__()](/docs/transformers/v5.21.1/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([Idefics3Processor](/docs/transformers/v5.21.1/en/model_doc/idefics3#transformers.Idefics3Processor) uses [UltraVBertImageProcessor](/docs/transformers/v5.21.1/en/model_doc/ultravbert#transformers.UltraVBertImageProcessor) for processing images).

pixel_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_images, image_size, image_size)`, *optional*) : Mask over the real pixels of each image tile.

image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The hidden states of the image encoder after modality projection.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

**Returns:** [TokenClassifierOutput](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

A [TokenClassifierOutput](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([UltraVBertConfig](/docs/transformers/v5.21.1/en/model_doc/ultravbert#transformers.UltraVBertConfig)) and inputs.

The [UltraVBertForTokenClassification](/docs/transformers/v5.21.1/en/model_doc/ultravbert#transformers.UltraVBertForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) -- Classification scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Example:

```python
>>> from transformers import AutoTokenizer, UltraVBertForTokenClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("athrael-soju/UltraVBERT")
>>> model = UltraVBertForTokenClassification.from_pretrained("athrael-soju/UltraVBERT")

>>> inputs = tokenizer(
...     "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
... )

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_token_class_ids = logits.argmax(-1)

>>> # Note that tokens are classified rather then input words which means that
>>> # there might be more predicted token classes than words.
>>> # Multiple token classes might account for the same word
>>> predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
>>> predicted_tokens_classes
...

>>> labels = predicted_token_class_ids
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
...
```

### GPT
https://huggingface.co/docs/transformers/v5.21.1/model_doc/openai-gpt.md
