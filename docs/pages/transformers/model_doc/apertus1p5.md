# Apertus 1.5

## Overview

Apertus 1.5 is a multimodal model from the [Swiss AI Initiative](https://huggingface.co/swiss-ai) that accepts
text, images, and audio and generates text. It extends the [Apertus](./apertus) language model with the
[EMU3.5 vision tokenizer](https://huggingface.co/BAAI/Emu3.5-VisionTokenizer) and [WavTokenizer](./wavtokenizer)
for audio. The EMU3.5 vision tokenizer is an image-only VQGAN with the same encoder as [Chameleon](./chameleon)'s,
unlike the [Emu3](./emu3) tokenizer, which adds temporal layers for video. It assigns codes by the largest inner
product with the codebook instead of the smallest L2 distance.

The media tokenizers convert images and audio into discrete tokens, which the language backbone processes
alongside text in a shared input vocabulary. The output vocabulary contains only text tokens. The integration
includes the media encoders and quantizers; it does not support image or audio generation, or training the
vision tokenizer.

## Usage example

Use the processor's chat template to prepare a conversation with images and audio. It loads the referenced
media, resamples audio, and constructs the model inputs automatically.

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model = AutoModelForMultimodalLM.from_pretrained(
    "swiss-ai/Apertus-v1.5-8B", device_map="auto"
)
processor = AutoProcessor.from_pretrained("swiss-ai/Apertus-v1.5-8B")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            {"type": "audio", "url": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/belinda.wav"},
            {"type": "text", "text": "Describe the image and transcribe the audio."},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=128)
print(processor.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

## Usage notes

- **Tokenizer precision:** Both media tokenizers require `float32` for stable code assignments.
  `from_pretrained` preserves this precision when loading the language backbone in half precision. Avoid
  casting the entire model to a lower precision or running the tokenizers under mixed-precision autocast.
  If quantizing the model, ensure the backend excludes both media tokenizers.
- **Output vocabulary:** Returned logits cover only `text_config.output_vocab_size` when set, otherwise
  `text_config.vocab_size`. Input-only multimodal tokens cannot be generated. Mask these tokens with `-100`
  in training labels, and restrict generation constraints to tokens within the output vocabulary.

## Apertus1p5Config[[transformers.Apertus1p5Config]]

#### transformers.Apertus1p5Config[[transformers.Apertus1p5Config]]

```python
transformers.Apertus1p5Config(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, text_config: dict | transformers.models.apertus1p5.configuration_apertus1p5.Apertus1p5TextConfig | None = None, vision_config: dict | transformers.models.apertus1p5.configuration_apertus1p5.Apertus1p5VisionTokenizerConfig | None = None, audio_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, image_token_id: int = 131079, audio_token_id: int = 131085, image_token_offset: int = 131272, audio_token_offset: int = 262344, tie_word_embeddings: bool = False)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/apertus1p5/configuration_apertus1p5.py#L182)

**Parameters:**

text_config (`Union[dict, ~models.apertus1p5.configuration_apertus1p5.Apertus1p5TextConfig]`, *optional*) : The config object or dictionary of the text backbone.

vision_config (`Union[dict, ~models.apertus1p5.configuration_apertus1p5.Apertus1p5VisionTokenizerConfig]`, *optional*) : The config object or dictionary of the vision backbone.

audio_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the audio backbone.

image_token_id (`int`, *optional*, defaults to `131079`) : The image token index used as a placeholder for input images.

audio_token_id (`int`, *optional*, defaults to `131085`) : The audio token index used as a placeholder for input audio.

image_token_offset (`int`, *optional*, defaults to 131272) : Offset mapping a visual codebook index `i` to the vocabulary id of `<|visual token i|>`.

audio_token_offset (`int`, *optional*, defaults to 262344) : Offset mapping an audio codebook index `i` to the vocabulary id of `<|audio token i|>`.

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

This is the configuration class to store the configuration of a Apertus1p5Model. It is used to instantiate a Apertus1P5
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [swiss-ai/Apertus-v1.5-8B](https://huggingface.co/swiss-ai/Apertus-v1.5-8B)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.24.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.24.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import Apertus1p5Config, Apertus1p5ForConditionalGeneration

>>> # Initializing a configuration (Apertus 1.5 backbone + vision/audio tokenizer defaults)
>>> configuration = Apertus1p5Config()

>>> # Initializing a model (with random weights) from the configuration
>>> model = Apertus1p5ForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Apertus1p5TextConfig[[transformers.Apertus1p5TextConfig]]

#### transformers.Apertus1p5TextConfig[[transformers.Apertus1p5TextConfig]]

```python
transformers.Apertus1p5TextConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vocab_size: int = 266752, hidden_size: int = 4096, intermediate_size: int = 14336, num_hidden_layers: int = 32, num_attention_heads: int = 32, num_key_value_heads: int | None = None, hidden_act: str = 'xielu', max_position_embeddings: int = 65536, initializer_range: float = 0.02, rms_norm_eps: float = 1e-05, use_cache: bool = True, pad_token_id: int | None = 3, bos_token_id: int | None = 1, eos_token_id: int | list[int] | None = 2, tie_word_embeddings: bool = False, rope_parameters: transformers.modeling_rope_utils.RopeParameters | dict | None = None, attention_bias: bool = False, attention_dropout: float | int = 0.0, output_vocab_size: int | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/apertus1p5/configuration_apertus1p5.py#L89)

**Parameters:**

vocab_size (`int`, *optional*, defaults to `266752`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

hidden_size (`int`, *optional*, defaults to `4096`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `14336`) : Dimension of the MLP representations.

num_hidden_layers (`int`, *optional*, defaults to `32`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `32`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

hidden_act (`str`, *optional*, defaults to `xielu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

max_position_embeddings (`int`, *optional*, defaults to `65536`) : The maximum sequence length that this model might ever be used with.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rms_norm_eps (`float`, *optional*, defaults to `1e-05`) : The epsilon used by the rms normalization layers.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

pad_token_id (`int`, *optional*, defaults to `3`) : Token id used for padding in the vocabulary.

bos_token_id (`int`, *optional*, defaults to `1`) : Token id used for beginning-of-stream in the vocabulary.

eos_token_id (`Union[int, list[int]]`, *optional*, defaults to `2`) : Token id used for end-of-stream in the vocabulary.

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

rope_parameters (`Union[~modeling_rope_utils.RopeParameters, dict]`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

attention_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

output_vocab_size (`int`, *optional*) : Number of LM-head rows kept after pruning the multimodal token rows from the output projection; the retained output ids are `0..output_vocab_size - 1`. `None` means the head is unpruned (`vocab_size` rows). Input embeddings always use `vocab_size`, while returned logits use the LM-head width, with or without `labels`. Released checkpoints use `tie_word_embeddings=False` to keep the pruned head separate from the input embeddings.

This is the configuration class to store the configuration of a Apertus1p5Model. It is used to instantiate a Apertus1P5
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [swiss-ai/Apertus-v1.5-8B](https://huggingface.co/swiss-ai/Apertus-v1.5-8B)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.24.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.24.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import Apertus1p5TextConfig, Apertus1p5TextForCausalLM

>>> # Initializing an Apertus 1.5 text configuration (extended vocabulary, optionally pruned output head)
>>> configuration = Apertus1p5TextConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = Apertus1p5TextForCausalLM(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Apertus1p5VisionTokenizerConfig[[transformers.Apertus1p5VisionTokenizerConfig]]

#### transformers.Apertus1p5VisionTokenizerConfig[[transformers.Apertus1p5VisionTokenizerConfig]]

```python
transformers.Apertus1p5VisionTokenizerConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, codebook_size: int = 131072, embed_dim: int = 256, latent_channels: int = 256, in_channels: int = 3, base_channels: int = 256, channel_multiplier: list[int] | tuple[int, ...] = (1, 1, 2, 2, 4), num_res_blocks: int = 4, attn_resolutions: list[int] | tuple[int, ...] = (16,), resolution: int = 256, dropout: float = 0.0)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/apertus1p5/configuration_apertus1p5.py#L30)

**Parameters:**

codebook_size (`int`, *optional*, defaults to 131072) : Number of entries in the IBQ codebook.

embed_dim (`int`, *optional*, defaults to 256) : Dimension of the codebook vectors (the quantizer input after `quant_conv`).

latent_channels (`int`, *optional*, defaults to 256) : Number of channels output by the encoder (before `quant_conv`).

in_channels (`int`, *optional*, defaults to `3`) : The number of input channels.

base_channels (`int`, *optional*, defaults to 256) : Base channel count of the encoder; per-stage widths are `base_channels * channel_multiplier[stage]`.

channel_multiplier (`list[int]`, *optional*, defaults to `(1, 1, 2, 2, 4)`) : Channel scaling factor per encoder stage. The number of stages is `len(channel_multiplier)`, giving a spatial downsampling factor of `2**(len(channel_multiplier) - 1)` (16 by default).

num_res_blocks (`int`, *optional*, defaults to 4) : Number of residual blocks per encoder stage.

attn_resolutions (`list[int]`, *optional*, defaults to `(16,)`) : Feature-map resolutions (relative to a `resolution`-sized input) at which attention blocks are inserted.

resolution (`int`, *optional*, defaults to 256) : Reference input resolution used (statically, together with `attn_resolutions`) to decide which encoder stages carry attention blocks. Does not restrict the actual input size.

dropout (`float`, *optional*, defaults to 0.0) : Dropout inside the residual blocks.

This is the configuration class to store the configuration of a Apertus1p5Model. It is used to instantiate a Apertus1P5
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [swiss-ai/Apertus-v1.5-8B](https://huggingface.co/swiss-ai/Apertus-v1.5-8B)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.24.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.24.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import Apertus1p5VisionTokenizerConfig, Apertus1p5VisionTokenizerModel

>>> # Initializing a vision tokenizer configuration (Apertus 1.5 defaults)
>>> configuration = Apertus1p5VisionTokenizerConfig()

>>> # Initializing the encode-only tokenizer (with random weights) from the configuration
>>> model = Apertus1p5VisionTokenizerModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Apertus1p5Processor[[transformers.Apertus1p5Processor]]

#### transformers.Apertus1p5Processor[[transformers.Apertus1p5Processor]]

```python
transformers.Apertus1p5Processor(image_processor = None, feature_extractor = None, tokenizer = None, chat_template = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/apertus1p5/processing_apertus1p5.py#L26)

**Parameters:**

image_processor (`Apertus1p5ImageProcessor`) : The image processor is a required input.

feature_extractor (`WavTokenizerFeatureExtractor`) : The feature extractor is a required input.

tokenizer (`tokenizer_class`) : The tokenizer is a required input.

chat_template (`str`) : A Jinja template to convert lists of messages in a chat into a tokenizable string.

Constructs an Apertus 1.5 processor for text, images, and audio.

#### __call__[[transformers.Apertus1p5Processor.__call__]]

```python
__call__(images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None, text: str | list[str] | list[list[str]] | None = None, videos: typing.Union[list['PIL.Image.Image'], numpy.ndarray, ForwardRef('torch.Tensor'), list[numpy.ndarray], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list[numpy.ndarray]], list[list['torch.Tensor']], transformers.video_utils.URL, list[transformers.video_utils.URL], list[list[transformers.video_utils.URL]], transformers.video_utils.Path, list[transformers.video_utils.Path], list[list[transformers.video_utils.Path]], NoneType] = None, audio: typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence['torch.Tensor'], NoneType] = None, **kwargs: *<class 'transformers.processing_utils.ProcessingKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/processing_utils.py#L671)

**Parameters:**

images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]`, *optional*) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

text (`Union[str, list[str], list[list[str]]]`, *optional*) : The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings (pretokenized string). If you pass a pretokenized input, set `is_split_into_words=True` to avoid ambiguity with batched inputs.

videos (`Union[list[PIL.Image.Image], numpy.ndarray, torch.Tensor, list[numpy.ndarray], list[torch.Tensor], list[list[PIL.Image.Image]], list[list[numpy.ndarray]], list[list[torch.Tensor]], ~video_utils.URL, list[~video_utils.URL], list[list[~video_utils.URL]], ~video_utils.Path, list[~video_utils.Path], list[list[~video_utils.Path]]]`, *optional*) : Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If passing in videos with pixel values between 0 and 1, set `do_rescale=False`.

audio (`Union[numpy.ndarray, torch.Tensor, collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence[torch.Tensor]]`, *optional*) : The audio or batch of audios to be prepared. Each audio can be a NumPy array or PyTorch tensor. In case of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels, and T is the sample length of the audio.

return_mm_token_type_ids (`bool`, *kwargs*) : Whether to return multimodal token type ids indicating multimodal placeholder token positions.

return_text_replacement_offsets (`bool`, *kwargs*) : Whether to return character offsets for each multimodal placeholder and its replacement.

return_tensors (`str` or [TensorType](/docs/transformers/v5.24.0/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

- ****kwargs** ([ProcessingKwargs](/docs/transformers/v5.24.0/en/main_classes/processors#transformers.ProcessingKwargs), *optional*) : Additional processing options for each modality (text, images, videos, audio). Model-specific parameters are listed above; see the TypedDict class for the complete list of supported arguments.

## Apertus1p5ImageProcessor[[transformers.Apertus1p5ImageProcessor]]

#### transformers.Apertus1p5ImageProcessor[[transformers.Apertus1p5ImageProcessor]]

```python
transformers.Apertus1p5ImageProcessor(**kwargs: *<class 'transformers.models.apertus1p5.image_processing_apertus1p5.Apertus1p5ImageProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/apertus1p5/image_processing_apertus1p5.py#L76)

**Parameters:**

do_convert_rgb (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to convert the image to RGB.

do_resize (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to resize the image.

size (`dict`, *kwargs*, *optional*, defaults to `{"min_pixels" : 128 * 128, "max_pixels": 1400 * 1400}`): Minimum and maximum pixel areas. Images are resized to approximately this area range.

default_to_square (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to default to a square image when resizing, if size is an int.

crop_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : Size of the output image after applying `center_crop`.

resample (`Annotated[Union[int, PILImageResampling, NoneType], None]`, *kwargs*, defaults to `3`) : Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.

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

spatial_factor (`int`, *kwargs*, *optional*, defaults to 16) : Both output sides are rounded to multiples of this factor (the vision tokenizer's downsampling factor); each `spatial_factor x spatial_factor` patch becomes one discrete code.

Constructs an Apertus 1.5 image processor with the training pipeline's resize policy.

#### preprocess[[transformers.Apertus1p5ImageProcessor.preprocess]]

```python
preprocess(images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']], **kwargs: *<class 'transformers.models.apertus1p5.image_processing_apertus1p5.Apertus1p5ImageProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/apertus1p5/image_processing_apertus1p5.py#L95)

**Parameters:**

images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]`) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

do_convert_rgb (`bool`, *kwargs*, *optional*) : Whether to convert the image to RGB.

do_resize (`bool`, *kwargs*, *optional*) : Whether to resize the image.

size (`dict`, *kwargs*, *optional*, defaults to `{"min_pixels" : 128 * 128, "max_pixels": 1400 * 1400}`): Minimum and maximum pixel areas. Images are resized to approximately this area range.

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

spatial_factor (`int`, *kwargs*, *optional*, defaults to 16) : Both output sides are rounded to multiples of this factor (the vision tokenizer's downsampling factor); each `spatial_factor x spatial_factor` patch becomes one discrete code.

**Returns:** `~image_processing_base.BatchFeature`

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

## Apertus1p5VisionTokenizerModel[[transformers.Apertus1p5VisionTokenizerModel]]

#### transformers.Apertus1p5VisionTokenizerModel[[transformers.Apertus1p5VisionTokenizerModel]]

```python
transformers.Apertus1p5VisionTokenizerModel(config: Apertus1p5VisionTokenizerConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/apertus1p5/modeling_apertus1p5.py#L316)

**Parameters:**

config ([Apertus1p5VisionTokenizerConfig](/docs/transformers/v5.24.0/en/model_doc/apertus1p5#transformers.Apertus1p5VisionTokenizerConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Encode-only EMU3.5 vision tokenizer used by Apertus 1.5: a 2D VQGAN encoder as in Chameleon, followed by a
quantizer that picks the codebook entry with the largest inner product. It returns hard codebook indices; run it in
`float32` for stable code assignment (Apertus 1.5 does this automatically).

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### encode[[transformers.Apertus1p5VisionTokenizerModel.encode]]

```python
encode(pixel_values: Tensor)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/apertus1p5/modeling_apertus1p5.py#L324)

**Parameters:**

pixel_values (`torch.Tensor` of shape `(batch_size, channels, height, width)`) : RGB images normalized to `[-1, 1]`.

**Returns:** `Apertus1p5VisionTokenizerEncoderOutput`

Discrete image-code grids in `image_tokens`.

Tokenizes images into a grid of discrete codebook indices.

## Apertus1p5TextModel[[transformers.Apertus1p5TextModel]]

#### transformers.Apertus1p5TextModel[[transformers.Apertus1p5TextModel]]

```python
transformers.Apertus1p5TextModel(config: Apertus1p5TextConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/apertus1p5/modeling_apertus1p5.py#L650)

**Parameters:**

config ([Apertus1p5TextConfig](/docs/transformers/v5.24.0/en/model_doc/apertus1p5#transformers.Apertus1p5TextConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Apertus1P5 Text Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Apertus1p5TextModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, cache: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.Tensor] = None, use_cache: bool | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/apertus1p5/modeling_apertus1p5.py#L667)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.24.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.24.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.24.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

cache (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `cache` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.24.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `cache` are passed, [DynamicCache](/docs/transformers/v5.24.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `cache` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `cache` key value states are returned and can be used to speed up decoding (see `cache`).

**Returns:** [BaseModelOutputWithPast](/docs/transformers/v5.24.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.Tensor)`

A [BaseModelOutputWithPast](/docs/transformers/v5.24.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Apertus1p5Config](/docs/transformers/v5.24.0/en/model_doc/apertus1p5#transformers.Apertus1p5Config)) and inputs.

The [Apertus1p5TextModel](/docs/transformers/v5.24.0/en/model_doc/apertus1p5#transformers.Apertus1p5TextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.

  If `cache` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **cache** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.24.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `cache`
  input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## Apertus1p5TextForCausalLM[[transformers.Apertus1p5TextForCausalLM]]

#### transformers.Apertus1p5TextForCausalLM[[transformers.Apertus1p5TextForCausalLM]]

```python
transformers.Apertus1p5TextForCausalLM(config: Apertus1p5TextConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/apertus1p5/modeling_apertus1p5.py#L725)

**Parameters:**

config ([Apertus1p5TextConfig](/docs/transformers/v5.24.0/en/model_doc/apertus1p5#transformers.Apertus1p5TextConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Apertus1P5 Model for causal language modeling.

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Apertus1p5TextForCausalLM.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, cache: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.Tensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/apertus1p5/modeling_apertus1p5.py#L742)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.24.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.24.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.24.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

cache (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `cache` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.24.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `cache` are passed, [DynamicCache](/docs/transformers/v5.24.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `cache` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Target token ids. `-100` ignores a position; other values must be within the physical LM-head range. With a pruned head, mask multimodal/input-only ids with `-100`.

use_cache (`bool`, *optional*) : If set to `True`, `cache` key value states are returned and can be used to speed up decoding (see `cache`).

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

**Returns:** [CausalLMOutputWithPast](/docs/transformers/v5.24.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.Tensor)`

Logits use the physical LM-head width (`config.output_vocab_size` when set, otherwise
`config.vocab_size`), with or without `labels`.

The [Apertus1p5TextForCausalLM](/docs/transformers/v5.24.0/en/model_doc/apertus1p5#transformers.Apertus1p5TextForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## Apertus1p5Model[[transformers.Apertus1p5Model]]

#### transformers.Apertus1p5Model[[transformers.Apertus1p5Model]]

```python
transformers.Apertus1p5Model(config: Apertus1p5Config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/apertus1p5/modeling_apertus1p5.py#L795)

**Parameters:**

config ([Apertus1p5Config](/docs/transformers/v5.24.0/en/model_doc/apertus1p5#transformers.Apertus1p5Config)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Apertus1P5 Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Apertus1p5Model.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, pixel_values: typing.Optional[torch.Tensor] = None, image_sizes: typing.Optional[torch.Tensor] = None, input_values: typing.Optional[torch.Tensor] = None, padding_mask: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, cache: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.Tensor] = None, use_cache: bool | None = None, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/apertus1p5/modeling_apertus1p5.py#L924)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.24.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.24.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.24.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses `image_processor_class` for processing images).

image_sizes (`torch.Tensor` of shape `(batch_size, 2)`, *optional*) : The sizes of the images in the batch, being (height, width) for each image.

input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library (`pip install torchcodec`) or the soundfile library (`pip install soundfile`). To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v5.24.0/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion into a tensor of type `torch.Tensor`. See `processor_class.__call__` for details.

padding_mask (`torch.Tensor` of shape `(num_clips, max_length)`, *optional*) : Right-padded mask with 1 for valid samples and 0 for padding, required with `input_values`.

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

cache (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `cache` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.24.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `cache` are passed, [DynamicCache](/docs/transformers/v5.24.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `cache` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `cache` key value states are returned and can be used to speed up decoding (see `cache`).

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** `MultimodalBaseModelOutputWithPast` or `tuple(torch.Tensor)`

A `MultimodalBaseModelOutputWithPast` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

The [Apertus1p5Model](/docs/transformers/v5.24.0/en/model_doc/apertus1p5#transformers.Apertus1p5Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the language model backbone.
  If `cache` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **cache** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.24.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `cache`
  input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
  Hidden-states of the language model backbone at the output of each layer plus the optional initial
  embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attention weights of the language model backbone, after the attention softmax.
- **image_outputs** (`BaseModelOutputWithPooling`, *optional*) -- Output of the image encoder: `last_hidden_state`, `pooler_output`, `hidden_states` and `attentions`.
  `None` if no images were processed.
- **video_outputs** (`BaseModelOutputWithPooling`, *optional*) -- Output of the video encoder: `last_hidden_state`, `pooler_output`, `hidden_states` and `attentions`.
  `None` if no videos were processed.
- **audio_outputs** (`BaseModelOutputWithPooling`, *optional*) -- Output of the audio encoder: `last_hidden_state`, `pooler_output`, `hidden_states` and `attentions`.
  `None` if no audio was processed.

## Apertus1p5ForConditionalGeneration[[transformers.Apertus1p5ForConditionalGeneration]]

#### transformers.Apertus1p5ForConditionalGeneration[[transformers.Apertus1p5ForConditionalGeneration]]

```python
transformers.Apertus1p5ForConditionalGeneration(config: Apertus1p5Config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/apertus1p5/modeling_apertus1p5.py#L993)

**Parameters:**

config ([Apertus1p5Config](/docs/transformers/v5.24.0/en/model_doc/apertus1p5#transformers.Apertus1p5Config)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Apertus1P5 Model for token generation conditioned on other modalities (e.g. image-text-to-text generation).

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Apertus1p5ForConditionalGeneration.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, pixel_values: typing.Optional[torch.Tensor] = None, image_sizes: typing.Optional[torch.Tensor] = None, input_values: typing.Optional[torch.Tensor] = None, padding_mask: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, cache: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.Tensor] = None, use_cache: bool | None = None, labels: typing.Optional[torch.LongTensor] = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/apertus1p5/modeling_apertus1p5.py#L1030)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.24.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.24.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.24.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses `image_processor_class` for processing images).

image_sizes (`torch.Tensor` of shape `(batch_size, 2)`, *optional*) : The sizes of the images in the batch, being (height, width) for each image.

input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library (`pip install torchcodec`) or the soundfile library (`pip install soundfile`). To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v5.24.0/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion into a tensor of type `torch.Tensor`. See `processor_class.__call__` for details.

padding_mask (`torch.Tensor` of shape `(num_clips, max_length)`, *optional*) : Audio sample mask: 1 for valid samples, 0 for right padding. Required with `input_values`.

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

cache (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `cache` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.24.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `cache` are passed, [DynamicCache](/docs/transformers/v5.24.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `cache` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `cache` key value states are returned and can be used to speed up decoding (see `cache`).

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Next-token labels in `[0, output_vocab_size)` (or `[0, text_config.vocab_size)` when unset). Use `-100` for input-only multimodal tokens and other ignored positions.

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** [CausalLMOutputWithPast](/docs/transformers/v5.24.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.Tensor)`

Logits have width `text_config.output_vocab_size`, or `text_config.vocab_size` when unset,
with or without `labels`.

The [Apertus1p5ForConditionalGeneration](/docs/transformers/v5.24.0/en/model_doc/apertus1p5#transformers.Apertus1p5ForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

#### get_image_features[[transformers.Apertus1p5ForConditionalGeneration.get_image_features]]

```python
get_image_features(pixel_values: Tensor, image_sizes: LongTensor, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/apertus1p5/modeling_apertus1p5.py#L1008)

**Parameters:**

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input images. Pixel values can be obtained using [Apertus1p5ImageProcessor](/docs/transformers/v5.24.0/en/model_doc/apertus1p5#transformers.Apertus1p5ImageProcessor). See [Apertus1p5ImageProcessor.__call__()](/docs/transformers/v5.24.0/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([Apertus1p5Processor](/docs/transformers/v5.24.0/en/model_doc/apertus1p5#transformers.Apertus1p5Processor) uses [Apertus1p5ImageProcessor](/docs/transformers/v5.24.0/en/model_doc/apertus1p5#transformers.Apertus1p5ImageProcessor) for processing images).

image_sizes (`torch.LongTensor` of shape `(batch_size, 2)`) : The sizes of the images in the batch, being (height, width) for each image.

**Returns:** `Apertus1p5VisionTokenizerModelOutput` or `tuple(torch.Tensor)`

A `Apertus1p5VisionTokenizerModelOutput` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Apertus1p5Config](/docs/transformers/v5.24.0/en/model_doc/apertus1p5#transformers.Apertus1p5Config)) and inputs.

- **last_hidden_state** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`tuple[torch.Tensor]`, *optional*) -- One `(num_tokens, hidden_size)` tensor of embedded image tokens per input image.
- **hidden_states** (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_tokens** (`list[torch.LongTensor]`, *optional*) -- One flat tensor of shared-vocabulary token ids per input image: its `(height // factor) * (width // factor)`
  codes in row-major order, offset by `config.image_token_offset`, where `factor` is the vision tokenizer's
  `spatial_scale_factor` (16 by default).

Example:

```python
>>> from PIL import Image
>>> from transformers import AutoProcessor, Apertus1p5ForConditionalGeneration

>>> model = Apertus1p5ForConditionalGeneration.from_pretrained("swiss-ai/Apertus-v1.5-8B")
>>> processor = AutoProcessor.from_pretrained("swiss-ai/Apertus-v1.5-8B")

>>> messages = [
...     {
...         "role": "user", "content": [
...             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
...             {"type": "text", "text": "Where is the cat standing?"},
...         ]
...     },
... ]

>>> inputs = processor.apply_chat_template(
...     messages,
...     tokenize=True,
...     return_dict=True,
...     return_tensors="pt",
...     add_generation_prompt=True
... )
>>> # Generate
>>> generate_ids = model.generate(**inputs)
>>> processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
```

#### get_audio_features[[transformers.Apertus1p5ForConditionalGeneration.get_audio_features]]

```python
get_audio_features(input_values: Tensor, padding_mask: Tensor, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/apertus1p5/modeling_apertus1p5.py#L1017)

**Parameters:**

input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`) : Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library (`pip install torchcodec`) or the soundfile library (`pip install soundfile`). To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v5.24.0/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion into a tensor of type `torch.Tensor`. See [Apertus1p5Processor.__call__()](/docs/transformers/v5.24.0/en/main_classes/processors#transformers.ProcessorMixin.__call__) for details.

padding_mask (`torch.Tensor` of shape `(num_clips, max_length)`) : Mask with 1 for valid samples and 0 for padding, as returned by the feature extractor.

**Returns:** `Apertus1p5AudioTokenizerModelOutput` or `tuple(torch.Tensor)`

A `Apertus1p5AudioTokenizerModelOutput` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Apertus1p5Config](/docs/transformers/v5.24.0/en/model_doc/apertus1p5#transformers.Apertus1p5Config)) and inputs.

- **last_hidden_state** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`tuple[torch.Tensor]`, *optional*) -- One `(num_tokens, hidden_size)` tensor of embedded audio tokens per input clip.
- **hidden_states** (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **audio_tokens** (`list[torch.LongTensor]`, *optional*) -- One tensor of `ceil(num_valid_samples / hop_length)` shared-vocabulary token ids per input clip, offset by
  `config.audio_token_offset`.

Example:

```python
>>> from PIL import Image
>>> from transformers import AutoProcessor, Apertus1p5ForConditionalGeneration

>>> model = Apertus1p5ForConditionalGeneration.from_pretrained("swiss-ai/Apertus-v1.5-8B")
>>> processor = AutoProcessor.from_pretrained("swiss-ai/Apertus-v1.5-8B")

>>> messages = [
...     {
...         "role": "user", "content": [
...             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
...             {"type": "text", "text": "Where is the cat standing?"},
...         ]
...     },
... ]

>>> inputs = processor.apply_chat_template(
...     messages,
...     tokenize=True,
...     return_dict=True,
...     return_tensors="pt",
...     add_generation_prompt=True
... )
>>> # Generate
>>> generate_ids = model.generate(**inputs)
>>> processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
```

### GptOss
https://huggingface.co/docs/transformers/v5.24.0/model_doc/gpt_oss.md
