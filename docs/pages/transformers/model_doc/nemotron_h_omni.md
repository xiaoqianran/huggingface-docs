# NemotronH Omni

NemotronH Omni is a multimodal reasoning model from NVIDIA that pairs the [NemotronH](./nemotron_h) hybrid
Mamba-Transformer language model with a [RADIO](./radio) vision encoder and an optional Parakeet-based sound encoder.
Image (and video) patches are projected through a RADIO tower and a pixel-shuffle MLP into the language model's
embedding space at the `<image>` / `<video>` context-token positions; audio clips are projected in the same way at
`<audio>` positions. The result is a single autoregressive model that reasons jointly over text, images, video and
sound.

The example below demonstrates how to reason over an image and a text prompt with [AutoModelForImageTextToText](/docs/transformers/v5.19.0/en/model_doc/auto#transformers.AutoModelForImageTextToText).

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto",
    # the Parakeet audio encoder has no flash-attention kernel, so pin that tower to sdpa
    attn_implementation={"": "flash_attention_2", "audio_config": "sdpa"},
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(processor.decode(output[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True))
```

## NemotronH_Omni_Reasoning_V3_Config[[transformers.NemotronH_Omni_Reasoning_V3_Config]]

#### transformers.NemotronH_Omni_Reasoning_V3_Config[[transformers.NemotronH_Omni_Reasoning_V3_Config]]

```python
transformers.NemotronH_Omni_Reasoning_V3_Config(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vision_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, text_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, audio_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, force_image_size: int | None = None, downsample_ratio: float = 0.5, projector_hidden_size: int = 4096, vision_hidden_size: int = 1280, video_pruning_rate: float = 0.0, video_temporal_patch_size: int = 2, image_token_id: int | None = None, video_token_id: int | None = None, audio_token_id: int | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron_h_omni/configuration_nemotron_h_omni.py#L28)

**Parameters:**

vision_config (`dict` or `RadioConfig`, *optional*) : Configuration for the RADIO vision encoder. Defaults to a default [RadioConfig](/docs/transformers/v5.19.0/en/model_doc/radio#transformers.RadioConfig).

text_config (`dict` or `NemotronHConfig`, *optional*) : Configuration for the NemotronH language model. Defaults to a default [NemotronHConfig](/docs/transformers/v5.19.0/en/model_doc/nemotron_h#transformers.NemotronHConfig).

audio_config (`dict` or `ParakeetEncoderConfig`, *optional*) : Configuration for the optional Parakeet sound encoder. `None` disables the audio branch.

force_image_size (`int`, *optional*) : Fixed input image resolution (in pixels) the vision tower expects.

downsample_ratio (`float`, *optional*, defaults to 0.5) : Pixel-shuffle spatial downsample ratio applied to the vision features.

projector_hidden_size (`int`, *optional*, defaults to 4096) : Hidden size of the vision-to-LLM MLP projector.

vision_hidden_size (`int`, *optional*, defaults to 1280) : Hidden size of the RADIO vision features.

video_pruning_rate (`float`, *optional*, defaults to 0.0) : Efficient-Video-Sampling token pruning rate; `0.0` disables pruning.

video_temporal_patch_size (`int`, *optional*, defaults to 2) : Number of frames collapsed into a single temporal patch by the video embedder.

image_token_id (`int`, *optional*) : Token id used as the image-context placeholder in `input_ids`.

video_token_id (`int`, *optional*) : Token id used as the video-context placeholder in `input_ids`.

audio_token_id (`int`, *optional*) : Token id used as the audio-context placeholder in `input_ids`.

This is the configuration class to store the configuration of a Nemotron H OmniModel. It is used to instantiate a Nemotron H Omni
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## NemotronH_Omni_Reasoning_V3ImageProcessor[[transformers.NemotronH_Omni_Reasoning_V3ImageProcessor]]

#### transformers.NemotronH_Omni_Reasoning_V3ImageProcessor[[transformers.NemotronH_Omni_Reasoning_V3ImageProcessor]]

```python
transformers.NemotronH_Omni_Reasoning_V3ImageProcessor(**kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron_h_omni/image_processing_nemotron_h_omni.py#L98)

**Parameters:**

do_convert_rgb (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to convert the image to RGB.

do_resize (`bool`, *kwargs*, *optional*) : Whether to resize the image.

size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : Describes the maximum input dimensions to the model.

default_to_square (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to default to a square image when resizing, if size is an int.

crop_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : Size of the output image after applying `center_crop`.

resample (`Annotated[Union[int, PILImageResampling, NoneType], None]`, *kwargs*, defaults to `Resampling.BICUBIC`) : Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.

do_rescale (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to rescale the image.

rescale_factor (`float`, *kwargs*, *optional*, defaults to `0.00392156862745098`) : Rescale factor to rescale the image by if `do_rescale` is set to `True`.

do_normalize (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to normalize the image.

image_mean (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`) : Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.

image_std (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`) : Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to `True`.

do_pad (`bool`, *kwargs*, *optional*) : Whether to pad the image. Padding is done either to the largest size in the batch or to a fixed square size per image. The exact padding strategy depends on the model.

pad_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : The size in `{"height": int, "width" int}` to pad the images to. Must be larger than any image size provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest height and width in the batch. Applied only when `do_pad=True.`

do_center_crop (`bool`, *kwargs*, *optional*) : Whether to center crop the image.

data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.

input_data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format. - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

device (`Annotated[Union[str, torch.device, NoneType], None]`, *kwargs*) : The device to process the videos on. If unset, the device is inferred from the input videos.

return_tensors (`Annotated[str | ~utils.generic.TensorType | None, None]`, *kwargs*) : Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors.

disable_grouping (`bool`, *kwargs*, *optional*) : Whether to disable grouping of images by size to process them individually and not in batches. If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157

image_seq_length (`int`, *kwargs*, *optional*) : The number of image tokens to be used for each image in the input. Added for backward compatibility but this should be set as a processor attribute in future models.

patch_size (`int`, *kwargs*, *optional*, defaults to 16) : Side length, in pixels, of one vision-tower patch.

downsample_ratio (`float`, *kwargs*, *optional*, defaults to 0.5) : Pixel-shuffle spatial downsample ratio applied by the model; each side of the patch grid is a multiple of its reciprocal.

min_num_patches (`int`, *kwargs*, *optional*, defaults to 1024) : Minimum number of `patch_size` patches an image is resized to.

max_num_patches (`int`, *kwargs*, *optional*, defaults to 13312) : Maximum number of `patch_size` patches an image is resized to; `0` disables the cap.

max_model_len (`int`, *kwargs*, *optional*, defaults to 16384) : Context length of the language model, which caps the patch budget of the images in one call.

Constructs a NemotronH_Omni_Reasoning_V3ImageProcessor image processor.

#### preprocess[[transformers.NemotronH_Omni_Reasoning_V3ImageProcessor.preprocess]]

```python
preprocess(images, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron_h_omni/image_processing_nemotron_h_omni.py#L122)

**Parameters:**

images (``) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

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

patch_size (`int`, *kwargs*, *optional*, defaults to 16) : Side length, in pixels, of one vision-tower patch.

downsample_ratio (`float`, *kwargs*, *optional*, defaults to 0.5) : Pixel-shuffle spatial downsample ratio applied by the model; each side of the patch grid is a multiple of its reciprocal.

min_num_patches (`int`, *kwargs*, *optional*, defaults to 1024) : Minimum number of `patch_size` patches an image is resized to.

max_num_patches (`int`, *kwargs*, *optional*, defaults to 13312) : Maximum number of `patch_size` patches an image is resized to; `0` disables the cap.

max_model_len (`int`, *kwargs*, *optional*, defaults to 16384) : Context length of the language model, which caps the patch budget of the images in one call.

**Returns:** `~image_processing_base.BatchFeature`

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

## NemotronH_Omni_Reasoning_V3VideoProcessor[[transformers.NemotronH_Omni_Reasoning_V3VideoProcessor]]

#### transformers.NemotronH_Omni_Reasoning_V3VideoProcessor[[transformers.NemotronH_Omni_Reasoning_V3VideoProcessor]]

```python
transformers.NemotronH_Omni_Reasoning_V3VideoProcessor(**kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron_h_omni/video_processing_nemotron_h_omni.py#L54)

**Parameters:**

- ****kwargs** (`NemotronH_Omni_Reasoning_V3VideoProcessorInitKwargs`, *optional*) : Additional processing options for each modality (text, images, videos, audio). Model-specific parameters are listed above; see the TypedDict class for the complete list of supported arguments.

Constructs a NemotronH_Omni_Reasoning_V3VideoProcessor video processor.

## NemotronH_Omni_Reasoning_V3Processor[[transformers.NemotronH_Omni_Reasoning_V3Processor]]

#### transformers.NemotronH_Omni_Reasoning_V3Processor[[transformers.NemotronH_Omni_Reasoning_V3Processor]]

```python
transformers.NemotronH_Omni_Reasoning_V3Processor(image_processor = None, video_processor = None, tokenizer = None, feature_extractor = None, chat_template = None, audio_sampling_rate: int = 16000, audio_subsampling_factor: int = 8, audio_hop_length: int = 160, video_temporal_patch_dim: int = 2, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron_h_omni/processing_nemotron_h_omni.py#L44)

**Parameters:**

image_processor (`NemotronH_Omni_Reasoning_V3ImageProcessor`) : The image processor is a required input.

video_processor (`NemotronH_Omni_Reasoning_V3VideoProcessor`) : The video processor is a required input.

tokenizer (`tokenizer_class`) : The tokenizer is a required input.

feature_extractor (`feature_extractor_class`) : The feature extractor is a required input.

chat_template (`str`) : A Jinja template to convert lists of messages in a chat into a tokenizable string.

audio_sampling_rate (`int`, *optional*, defaults to 16000) : Sampling rate, in Hz, the audio waveforms are expected to be at.

audio_subsampling_factor (`int`, *optional*, defaults to 8) : Factor by which the sound encoder subsamples the mel frames, used to size the audio placeholder run.

audio_hop_length (`int`, *optional*, defaults to 160) : Hop length, in samples, between consecutive mel frames.

video_temporal_patch_dim (`int`, *optional*, defaults to 2) : Number of frames collapsed into a single temporal patch by the model's video embedder.

Constructs a NemotronH_Omni_Reasoning_V3Processor which wraps a image processor, a video processor, a tokenizer, and a feature extractor into a single processor.

[NemotronH_Omni_Reasoning_V3Processor](/docs/transformers/v5.19.0/en/model_doc/nemotron_h_omni#transformers.NemotronH_Omni_Reasoning_V3Processor) offers all the functionalities of [NemotronH_Omni_Reasoning_V3ImageProcessor](/docs/transformers/v5.19.0/en/model_doc/nemotron_h_omni#transformers.NemotronH_Omni_Reasoning_V3ImageProcessor), [NemotronH_Omni_Reasoning_V3VideoProcessor](/docs/transformers/v5.19.0/en/model_doc/nemotron_h_omni#transformers.NemotronH_Omni_Reasoning_V3VideoProcessor), `tokenizer_class`, and `feature_extractor_class`. See the
[~NemotronH_Omni_Reasoning_V3ImageProcessor](/docs/transformers/v5.19.0/en/model_doc/nemotron_h_omni#transformers.NemotronH_Omni_Reasoning_V3ImageProcessor), [~NemotronH_Omni_Reasoning_V3VideoProcessor](/docs/transformers/v5.19.0/en/model_doc/nemotron_h_omni#transformers.NemotronH_Omni_Reasoning_V3VideoProcessor), `~tokenizer_class`, and `~feature_extractor_class` for more information.

#### batch_decode[[transformers.NemotronH_Omni_Reasoning_V3Processor.batch_decode]]

```python
batch_decode(*args, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron_h_omni/processing_nemotron_h_omni.py#L287)

Forward to the tokenizer's [batch_decode()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode).

#### decode[[transformers.NemotronH_Omni_Reasoning_V3Processor.decode]]

```python
decode(*args, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron_h_omni/processing_nemotron_h_omni.py#L291)

Forward to the tokenizer's [decode()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.decode).

#### post_process_image_text_to_text[[transformers.NemotronH_Omni_Reasoning_V3Processor.post_process_image_text_to_text]]

```python
post_process_image_text_to_text(generated_outputs, skip_special_tokens = True, clean_up_tokenization_spaces = False, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron_h_omni/processing_nemotron_h_omni.py#L295)

Decode the model's generated token ids into text.

#### replace_video_token[[transformers.NemotronH_Omni_Reasoning_V3Processor.replace_video_token]]

```python
replace_video_token(video_inputs: dict, video_idx: int, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron_h_omni/processing_nemotron_h_omni.py#L169)

Expand `<video>` into one `<img>...</img>` chunk per temporal patch (tubelet).

Each chunk is labeled with the timestamps of the frames it packs, joined by " and "
("Frame" for the first frame in the tubelet, "frame" for the rest). The tokenizer has no
real `<video>` token, so the chunks use the image token; the model tells image from video
by which `pixel_values_*` argument was passed.

## NemotronH_Omni_Reasoning_V3[[transformers.NemotronH_Omni_Reasoning_V3]]

#### transformers.NemotronH_Omni_Reasoning_V3[[transformers.NemotronH_Omni_Reasoning_V3]]

```python
transformers.NemotronH_Omni_Reasoning_V3(config: NemotronH_Omni_Reasoning_V3_Config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron_h_omni/modeling_nemotron_h_omni.py#L142)

#### forward[[transformers.NemotronH_Omni_Reasoning_V3.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, pixel_values: typing.Optional[torch.FloatTensor] = None, image_grid_hw: typing.Optional[torch.LongTensor] = None, pixel_values_videos: typing.Optional[torch.FloatTensor] = None, input_features: typing.Optional[torch.FloatTensor] = None, input_features_mask: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, labels: typing.Optional[torch.LongTensor] = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron_h_omni/modeling_nemotron_h_omni.py#L347)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.19.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

pixel_values (`torch.FloatTensor` of shape `(total_patches, num_channels * patch_size**2)`, *optional*) : Flattened patches of all images, concatenated, as returned by the image processor.

image_grid_hw (`torch.LongTensor` of shape `(num_images, 2)`, *optional*) : Patch grid `(height, width)` of each image in `pixel_values`.

pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) : The tensors corresponding to the input video. Pixel values for videos can be obtained using [NemotronH_Omni_Reasoning_V3VideoProcessor](/docs/transformers/v5.19.0/en/model_doc/nemotron_h_omni#transformers.NemotronH_Omni_Reasoning_V3VideoProcessor). See `NemotronH_Omni_Reasoning_V3VideoProcessor.__call__()` for details ([NemotronH_Omni_Reasoning_V3Processor](/docs/transformers/v5.19.0/en/model_doc/nemotron_h_omni#transformers.NemotronH_Omni_Reasoning_V3Processor) uses [NemotronH_Omni_Reasoning_V3VideoProcessor](/docs/transformers/v5.19.0/en/model_doc/nemotron_h_omni#transformers.NemotronH_Omni_Reasoning_V3VideoProcessor) for processing videos).

input_features (`torch.FloatTensor` of shape `(batch_size, num_frames, num_mel_bins)`, *optional*) : Mel features produced by the processor, encoded and scattered onto the audio placeholder tokens.

input_features_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*) : Mask marking the real mel frames of each padded clip.

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** [CausalLMOutputWithPast](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

A [CausalLMOutputWithPast](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([NemotronH_Omni_Reasoning_V3_Config](/docs/transformers/v5.19.0/en/model_doc/nemotron_h_omni#transformers.NemotronH_Omni_Reasoning_V3_Config)) and inputs.

The [NemotronH_Omni_Reasoning_V3](/docs/transformers/v5.19.0/en/model_doc/nemotron_h_omni#transformers.NemotronH_Omni_Reasoning_V3) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

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
>>> from PIL import Image
>>> from transformers import AutoProcessor, NemotronH_Omni_Reasoning_V3

>>> model = NemotronH_Omni_Reasoning_V3.from_pretrained("nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16")
>>> processor = AutoProcessor.from_pretrained("nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16")

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

#### generate[[transformers.NemotronH_Omni_Reasoning_V3.generate]]

```python
generate(inputs: typing.Optional[torch.Tensor] = None, generation_config: transformers.generation.configuration_utils.GenerationConfig | None = None, logits_processor: transformers.generation.logits_process.LogitsProcessorList | None = None, stopping_criteria: transformers.generation.stopping_criteria.StoppingCriteriaList | None = None, prefix_allowed_tokens_fn: collections.abc.Callable[[int, torch.Tensor], list[int]] | None = None, synced_gpus: bool | None = None, assistant_model: typing.Optional[ForwardRef('PreTrainedModel')] = None, streamer: typing.Optional[ForwardRef('BaseStreamer')] = None, negative_prompt_ids: typing.Optional[torch.Tensor] = None, negative_prompt_attention_mask: typing.Optional[torch.Tensor] = None, custom_generate: str | collections.abc.Callable | None = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/generation/utils.py#L2562)

**Parameters:**

inputs (`torch.Tensor` of varying shape depending on the modality, *optional*) : The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs` should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of `input_ids`, `input_values`, `input_features`, or `pixel_values`.

generation_config ([GenerationConfig](/docs/transformers/v5.19.0/en/main_classes/text_generation#transformers.GenerationConfig), *optional*) : The generation configuration to be used as base parametrization for the generation call. `**kwargs` passed to generate matching the attributes of `generation_config` will override them. If `generation_config` is not provided, the default will be used, which has the following loading priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model configuration. Please note that unspecified parameters will inherit [GenerationConfig](/docs/transformers/v5.19.0/en/main_classes/text_generation#transformers.GenerationConfig)'s default values, whose documentation should be checked to parameterize generation.

logits_processor (`LogitsProcessorList`, *optional*) : Custom logits processors that complement the default logits processors built from arguments and generation config. If a logit processor is passed that is already created with the arguments or a generation config an error is thrown. This feature is intended for advanced users.

stopping_criteria (`StoppingCriteriaList`, *optional*) : Custom stopping criteria that complements the default stopping criteria built from arguments and a generation config. If a stopping criteria is passed that is already created with the arguments or a generation config an error is thrown. If your stopping criteria depends on the `scores` input, make sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is intended for advanced users.

prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], list[int]]`, *optional*) : If provided, this function constraints the beam search to allowed tokens only at each step. If not provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful for constrained generation conditioned on the prefix, as described in [Autoregressive Entity Retrieval](https://huggingface.co/papers/2010.00904).

synced_gpus (`bool`, *optional*) : Whether to continue running the while loop until max_length. Unless overridden, this flag will be set to `True` if using `FullyShardedDataParallel` or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to `False`.

assistant_model (`PreTrainedModel`, *optional*) : An assistant model that can be used to accelerate generation. The assistant model must have the exact same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model is much faster than running generation with the model you're calling generate from. As such, the assistant model should be much smaller.

streamer (`BaseStreamer`, *optional*) : Streamer object that will be used to stream the generated sequences. Generated tokens are passed through `streamer.put(token_ids)` and the streamer is responsible for any further processing.

negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : The negative prompt needed for some processors such as CFG. The batch size must match the input batch size. This is an experimental feature, subject to breaking API changes in future versions.

negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Attention_mask for `negative_prompt_ids`.

custom_generate (`str` or `Callable`, *optional*) : One of the following: - `str` (Hugging Face Hub repository name): runs the custom `generate` function defined at `custom_generate/generate.py` in that repository instead of the standard `generate` method. The repository fully replaces the generation logic, and the return type may differ. - `str` (local repository path): same as above but from a local path. Local directories also require `trust_remote_code=True` because the local `custom_generate/generate.py` is executed. - `Callable`: `generate` will perform the usual input preparation steps, then call the provided callable to run the decoding loop. For more information, see [the docs](../../generation_strategies#custom-generation-methods).

kwargs (`dict[str, Any]`, *optional*) : Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

**Returns:** [ModelOutput](/docs/transformers/v5.19.0/en/main_classes/output#transformers.utils.ModelOutput) or `torch.LongTensor`

A [ModelOutput](/docs/transformers/v5.19.0/en/main_classes/output#transformers.utils.ModelOutput) (if `return_dict_in_generate=True`
or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
[ModelOutput](/docs/transformers/v5.19.0/en/main_classes/output#transformers.utils.ModelOutput) types are:

- [GenerateDecoderOnlyOutput](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.generation.GenerateDecoderOnlyOutput),
- [GenerateBeamDecoderOnlyOutput](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.generation.GenerateBeamDecoderOnlyOutput)

If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
[ModelOutput](/docs/transformers/v5.19.0/en/main_classes/output#transformers.utils.ModelOutput) types are:

- [GenerateEncoderDecoderOutput](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.generation.GenerateEncoderDecoderOutput),
- [GenerateBeamEncoderDecoderOutput](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.generation.GenerateBeamEncoderDecoderOutput)

Generates sequences of token ids for models with a language modeling head.

Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
model's default generation configuration. You can override any `generation_config` by passing the corresponding
parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

For an overview of generation strategies and code examples, check out the [following
guide](../generation_strategies).

### LayoutLMv3
https://huggingface.co/docs/transformers/v5.19.0/model_doc/layoutlmv3.md
