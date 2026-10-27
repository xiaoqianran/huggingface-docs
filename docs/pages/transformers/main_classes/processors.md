# Processors

## Multi-modal processors[[transformers.ProcessorMixin]]

Any multi-modal model will require an object to encode or decode the data that groups several modalities (among text,
vision and audio). This is handled by objects called processors, which group together two or more processing objects
such as tokenizers (for the text modality), image processors (for vision) and feature extractors (for audio).

Those processors inherit from the following base class that implements the saving and loading functionality:

#### transformers.ProcessorMixin[[transformers.ProcessorMixin]]

```python
transformers.ProcessorMixin(*args, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.0/src/transformers/processing_utils.py#L598)

This is a mixin used to provide saving/loading functionality for all processor classes.

#### __call__[[transformers.ProcessorMixin.__call__]]

```python
__call__(images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None, text: str | list[str] | list[list[str]] | None = None, videos: typing.Union[list['PIL.Image.Image'], numpy.ndarray, ForwardRef('torch.Tensor'), list[numpy.ndarray], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list[numpy.ndarray]], list[list['torch.Tensor']], transformers.video_utils.URL, list[transformers.video_utils.URL], list[list[transformers.video_utils.URL]], transformers.video_utils.Path, list[transformers.video_utils.Path], list[list[transformers.video_utils.Path]], NoneType] = None, audio: typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence['torch.Tensor'], NoneType] = None, **kwargs: *<class 'transformers.processing_utils.ProcessingKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.0/src/transformers/processing_utils.py#L652)

**Parameters:**

images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]`, *optional*) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

text (`Union[str, list[str], list[list[str]]]`, *optional*) : The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings (pretokenized string). If you pass a pretokenized input, set `is_split_into_words=True` to avoid ambiguity with batched inputs.

videos (`Union[list[PIL.Image.Image], numpy.ndarray, torch.Tensor, list[numpy.ndarray], list[torch.Tensor], list[list[PIL.Image.Image]], list[list[numpy.ndarray]], list[list[torch.Tensor]], ~video_utils.URL, list[~video_utils.URL], list[list[~video_utils.URL]], ~video_utils.Path, list[~video_utils.Path], list[list[~video_utils.Path]]]`, *optional*) : Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If passing in videos with pixel values between 0 and 1, set `do_rescale=False`.

audio (`Union[numpy.ndarray, torch.Tensor, collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence[torch.Tensor]]`, *optional*) : The audio or batch of audios to be prepared. Each audio can be a NumPy array or PyTorch tensor. In case of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels, and T is the sample length of the audio.

return_tensors (`str` or [TensorType](/docs/transformers/v5.21.0/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

- ****kwargs** ([ProcessingKwargs](/docs/transformers/v5.21.0/en/main_classes/processors#transformers.ProcessingKwargs), *optional*) : Additional processing options for each modality (text, images, videos, audio). Model-specific parameters are listed above; see the TypedDict class for the complete list of supported arguments.

#### prepare_inputs_layout[[transformers.ProcessorMixin.prepare_inputs_layout]]

```python
prepare_inputs_layout(images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None, text: str | list[str] | list[list[str]] | None = None, videos: typing.Union[list['PIL.Image.Image'], numpy.ndarray, ForwardRef('torch.Tensor'), list[numpy.ndarray], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list[numpy.ndarray]], list[list['torch.Tensor']], transformers.video_utils.URL, list[transformers.video_utils.URL], list[list[transformers.video_utils.URL]], transformers.video_utils.Path, list[transformers.video_utils.Path], list[list[transformers.video_utils.Path]], NoneType] = None, audio: typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence['torch.Tensor'], NoneType] = None, **kwargs: *<class 'transformers.processing_utils.ProcessingKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.0/src/transformers/processing_utils.py#L713)

Normalize and prefetch inputs before processing. Wraps text in a list for multimodal
processors, fetches remote images and audio if URLs are provided, and ensures audio
is properly batched. Returns the normalized `(images, text, videos, audio)` tuple.

#### validate_inputs[[transformers.ProcessorMixin.validate_inputs]]

```python
validate_inputs(images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None, text: str | list[str] | list[list[str]] | None = None, videos: typing.Union[list['PIL.Image.Image'], numpy.ndarray, ForwardRef('torch.Tensor'), list[numpy.ndarray], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list[numpy.ndarray]], list[list['torch.Tensor']], transformers.video_utils.URL, list[transformers.video_utils.URL], list[list[transformers.video_utils.URL]], transformers.video_utils.Path, list[transformers.video_utils.Path], list[list[transformers.video_utils.Path]], NoneType] = None, audio: typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence['torch.Tensor'], NoneType] = None, **kwargs: *<class 'transformers.processing_utils.ProcessingKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.0/src/transformers/processing_utils.py#L743)

Validate that at least one input is provided and that no deprecated keyword arguments
are used. Raises `ValueError` otherwise.

Override when the processor needs additional validation on the input args.

#### get_text_with_replacements[[transformers.ProcessorMixin.get_text_with_replacements]]

```python
get_text_with_replacements(text: list, images_replacements: list = [], videos_replacements: list = [], audio_replacements: list = [])
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.0/src/transformers/processing_utils.py#L816)

**Parameters:**

text (*list[str]*) : Batch of raw text strings, each potentially containing multimodal placeholder tokens. Note that it will be modified in-place and returned.

images_replacements (*list[str]*, *optional*, defaults to *[]*) : Expanded replacement strings for each image, in the order they appear across the batch. Produced by *self._process_images*.

videos_replacements (*list[str]*, *optional*, defaults to *[]*) : Expanded replacement strings for each video. Produced by *self._process_videos*.

audio_replacements (*list[str]*, *optional*, defaults to *[]*) : Expanded replacement strings for each audio input. Produced by *self._process_audio*.

**Returns:** *tuple[list[str], list[dict[str, Any]]]*

A tuple of:
- The modified *text* batch with all placeholder tokens expanded.
- *batch_replacement_offsets*: one entry per batch item, each being a
list of dicts with keys:
- *"type"* (*str*): modality name — *"image"*, *"video"*, or *"audio"*
- *"span"* (*tuple[int, int]*): original *(start, end)* char offsets of the placeholder token
- *"new_span"* (*tuple[int, int]*): *(start, end)* offsets of placeholder in the expanded string
- *"text"* (*str*): the original placeholder token string that was matched
- *"replacement"* (*str*): the string it was replaced with

Replace multimodal placeholder tokens in a batch of text strings with their
expanded representations, and return the modified texts alongside offset metadata.

This method is the core text-side preprocessing step for multimodal inputs. It
scans each text in the batch for special tokens (image, video, audio) and replaces
them in-order with the pre-computed replacement strings produced by
*self.replace_image_token* / *self.replace_video_token* / *self.replace_audio_token*.
Replacements are consumed from each modality's list sequentially, so the i-th
occurrence of e.g. `self.image_token` is replaced by `images_replacements[i]`.

To add a new multimodal processor with placeholder tokens, you need to define a correct
*self.image_token* which is the same token that is embedded in input text and also used as
placeholder and repeated many times. Then you need to override *self.replace_image_token*
to return the correct replacement string for a given image at index *i*. Same goes for all
other supported modalities.

#### create_mm_token_type_ids[[transformers.ProcessorMixin.create_mm_token_type_ids]]

```python
create_mm_token_type_ids(input_ids: list)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.0/src/transformers/processing_utils.py#L921)

**Parameters:**

input_ids (*list[list[int]]*) : Batch of token ID sequences. May be unpadded (variable length), so a plain Python list of lists is expected rather than a tensor or uniformly-shaped array.

**Returns:** *list[list[int]]*

A list of the same structure as `input_ids`, where each
integer is the modality type ID for the corresponding token.

Build per-token modality type IDs for a batch of token_id sequences.

Each position is assigned an integer indicating which modality it belongs to:
`0` for regular text, `1` for image tokens, `2` for video tokens, and
`3` for audio tokens. Membership is determined by comparing against
`self.image_token_ids`, `self.video_token_ids`, and `self.audio_token_ids`.

#### apply_chat_template[[transformers.ProcessorMixin.apply_chat_template]]

```python
apply_chat_template(conversation: list[dict[str, str]] | list[list[dict[str, str]]], chat_template: str | None = None, tools: list[dict] | None = None, documents: list[dict[str, str]] | None = None, add_generation_prompt: bool = False, continue_final_message: bool | str = False, return_assistant_tokens_mask: bool = False, tokenize: bool = False, return_tensors: str | transformers.utils.generic.TensorType | None = None, return_dict: bool = False, load_audio_from_video: bool = False, processor_kwargs: dict | None = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.0/src/transformers/processing_utils.py#L1994)

**Parameters:**

conversation (`Union[list[Dict, [str, str]], list[list[dict[str, str]]]]`) : The conversation to format.

chat_template (`Optional[str]`, *optional*) : The Jinja template to use for formatting the conversation. If not provided, the tokenizer's chat template is used.

Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
conversations to turn them into a single tokenizable string.

The input is expected to be in the following format, where each message content is a list consisting of text and
optionally image or video inputs. One can also provide an image, video, URL or local path which will be used to form
`pixel_values` when `return_dict=True`. If not provided, one will get only the formatted text, optionally tokenized text.

conversation = [
{
"role": "user",
"content": [
{"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
{"type": "text", "text": "Please describe this image in detail."},
],
},
]

### Processing kwargs[[transformers.ProcessingKwargs]]

Processor `__call__` methods accept keyword arguments organized by modality. The following TypedDict classes define
the available keyword arguments for each modality. Model-specific processors may subclass these to add or override
fields.

#### transformers.ProcessingKwargs[[transformers.ProcessingKwargs]]

[Source](https://github.com/huggingface/transformers/blob/v5.21.0/src/transformers/processing_utils.py#L434)

Base class for kwargs passing to processors.
In case a model has specific kwargs that are not present in the base class or default values for existing keys,
it should have its own `ModelProcessorKwargs` class that inherits from `ProcessingKwargs` to provide:
1) Additional typed keys and that this model requires to process inputs.
2) Default values for existing keys under a `_defaults` attribute.
New keys have to be defined as follows to ensure type hinting is done correctly.

```python
# adding a new image kwarg for this model
class ModelImagesKwargs(ImagesKwargs, total=False):
    new_image_kwarg: Optional[bool]

class ModelProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: ModelImagesKwargs
    _defaults = {
        "images_kwargs: {
            "new_image_kwarg": False,
        }
        "text_kwargs": {
            "padding": "max_length",
        },
    }

```

For Python 3.8 compatibility, when inheriting from this class and overriding one of the kwargs,

you need to manually update the __annotations__ dictionary. This can be done as follows:

```python
class CustomProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: CustomImagesKwargs

CustomProcessorKwargs.__annotations__["images_kwargs"] = CustomImagesKwargs  # python 3.8 compatibility
```

#### transformers.TextKwargs[[transformers.TextKwargs]]

[Source](https://github.com/huggingface/transformers/blob/v5.21.0/src/transformers/processing_utils.py#L162)

**Parameters:**

add_special_tokens (`bool`, *optional*) : Whether or not to add special tokens when encoding the sequences.

padding (`bool`, `str` or [PaddingStrategy](/docs/transformers/v5.21.0/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*) : Activates and controls padding.

truncation (`bool`, `str` or [TruncationStrategy](/docs/transformers/v5.21.0/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*) : Activates and controls truncation.

max_length (`int`, *optional*) : Controls the maximum length to use by one of the truncation/padding parameters.

stride (`int`, *optional*) : If set, the overflowing tokens will contain some tokens from the end of the truncated sequence.

is_split_into_words (`bool`, *optional*) : Whether or not the input is already pre-tokenized.

pad_to_multiple_of (`int`, *optional*) : If set, will pad the sequence to a multiple of the provided value.

return_token_type_ids (`bool`, *optional*) : Whether to return token type IDs.

return_attention_mask (`bool`, *optional*) : Whether to return the attention mask.

return_overflowing_tokens (`bool`, *optional*) : Whether or not to return overflowing token sequences.

return_special_tokens_mask (`bool`, *optional*) : Whether or not to return special tokens mask information.

return_offsets_mapping (`bool`, *optional*) : Whether or not to return `(char_start, char_end)` for each token.

return_length (`bool`, *optional*) : Whether or not to return the lengths of the encoded inputs.

verbose (`bool`, *optional*) : Whether or not to print more information and warnings.

padding_side (`str`, *optional*) : The side on which padding will be applied.

return_mm_token_type_ids (`bool`, *optional*) : Whether to return multimodal token type ids indicating mm placeholder token positions.

return_text_replacement_offsets (`bool`, *optional*) : Whether to return character offsets for each mm placeholder and its replacement.

return_tensors (`str` or [TensorType](/docs/transformers/v5.21.0/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Acceptable values are: - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

Keyword arguments for text processing. For extended documentation, check out tokenization_utils_base methods and
docstrings associated.

#### transformers.ImagesKwargs[[transformers.ImagesKwargs]]

[Source](https://github.com/huggingface/transformers/blob/v5.21.0/src/transformers/processing_utils.py#L231)

**Parameters:**

do_convert_rgb (`bool`) : Whether to convert the image to RGB format.

do_resize (`bool`, *optional*) : Whether to resize the image.

size (`dict[str, int]`, *optional*) : Resize using one of the supported size dictionaries. Pixel-area bounds use `{"min_pixels": int, "max_pixels": int}`.

default_to_square (`bool`, *optional*, defaults to `self.default_to_square`) : Whether to default to a square when resizing, if size is an int.

crop_size (`dict[str, int]`, *optional*) : Desired output size when applying center-cropping.

resample (`PILImageResampling`, *optional*) : Resampling filter to use if resizing the image.

do_rescale (`bool`, *optional*) : Whether to rescale the image by the specified scale `rescale_factor`.

rescale_factor (`int` or `float`, *optional*) : Scale factor to use if rescaling the image.

do_normalize (`bool`, *optional*) : Whether to normalize the image.

image_mean (`float` or `list[float] or tuple[float, float, float]`, *optional*) : Mean to use if normalizing the image.

image_std (`float` or `list[float] or tuple[float, float, float]`, *optional*) : Standard deviation to use if normalizing the image.

do_pad (`bool`, *optional*) : Whether to pad the images in the batch.

pad_size (`dict[str, int]`, *optional*) : The size `{"height": int, "width" int}` to pad the images to.

do_center_crop (`bool`, *optional*) : Whether to center crop the image.

data_format (`ChannelDimension` or `str`, *optional*) : The channel dimension format for the output image.

input_data_format (`ChannelDimension` or `str`, *optional*) : The channel dimension format for the input image.

device (`Union[str, torch.Tensor]`, *optional*) : The device to use for processing (e.g. "cpu", "cuda"), only relevant for torchvision backend.

return_tensors (`str` or [TensorType](/docs/transformers/v5.21.0/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Acceptable values are: - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

disable_grouping (`bool`, *optional*) : Whether to group images by shapes when processing or not, only relevant for torchvision backend.

image_seq_length (`int`, *optional*) : The number of image tokens to be used for each image in the input. Added for backward compatibility but this should be set as a processor attribute in future models.

Keyword arguments for image processing. For extended documentation, check the appropriate ImageProcessor
class methods and docstrings.

#### transformers.VideosKwargs[[transformers.VideosKwargs]]

[Source](https://github.com/huggingface/transformers/blob/v5.21.0/src/transformers/processing_utils.py#L305)

**Parameters:**

do_convert_rgb (`bool`) : Whether to convert the video to RGB format.

do_resize (`bool`) : Whether to resize the video.

size (`dict[str, int]`, *optional*) : Resize the shorter side of the input to `size["shortest_edge"]`.

default_to_square (`bool`, *optional*, defaults to `self.default_to_square`) : Whether to default to a square when resizing, if size is an int.

resample (`PILImageResampling`, *optional*) : Resampling filter to use if resizing the video.

do_rescale (`bool`, *optional*) : Whether to rescale the video by the specified scale `rescale_factor`.

rescale_factor (`int` or `float`, *optional*) : Scale factor to use if rescaling the video.

do_normalize (`bool`, *optional*) : Whether to normalize the video.

image_mean (`float` or `list[float] or tuple[float, float, float]`, *optional*) : Mean to use if normalizing the video.

image_std (`float` or `list[float] or tuple[float, float, float]`, *optional*) : Standard deviation to use if normalizing the video.

do_center_crop (`bool`, *optional*) : Whether to center crop the video.

do_pad (`bool`, *optional*) : Whether to pad the images in the batch.

do_sample_frames (`bool`, *optional*) : Whether to sample frames from the video before processing or to process the whole video.

video_metadata (`Union[VideoMetadata, dict]`, *optional*) : Metadata of the video containing information about total duration, fps and total number of frames.

num_frames (`int`, *optional*) : Maximum number of frames to sample when `do_sample_frames=True`.

fps (`int` or `float`, *optional*) : Target frames to sample per second when `do_sample_frames=True`.

crop_size (`dict[str, int]`, *optional*) : Desired output size when applying center-cropping.

data_format (`ChannelDimension` or `str`, *optional*) : The channel dimension format for the output video.

input_data_format (`ChannelDimension` or `str`, *optional*) : The channel dimension format for the input video.

device (`Union[str, torch.Tensor]`, *optional*) : The device to use for processing (e.g. "cpu", "cuda"), only relevant for fast image processing.

return_metadata (`bool`, *optional*) : Whether to return video metadata or not.

return_tensors (`str` or [TensorType](/docs/transformers/v5.21.0/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Acceptable values are: - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

Keyword arguments for video processing.

#### transformers.AudioKwargs[[transformers.AudioKwargs]]

[Source](https://github.com/huggingface/transformers/blob/v5.21.0/src/transformers/processing_utils.py#L382)

**Parameters:**

sampling_rate (`int`, *optional*) : The sampling rate at which the `raw_speech` input was sampled.

raw_speech (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`) : The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not stereo, i.e. single float per timestep.

padding (`bool`, `str` or [PaddingStrategy](/docs/transformers/v5.21.0/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*) : Select a strategy to pad the returned sequences (according to the model's padding side and padding index) among:  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence if provided). - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. - `False` or `'do_not_pad'`

max_length (`int`, *optional*) : Maximum length of the returned list and optionally padding length (see above).

truncation (`bool`, *optional*) : Activates truncation to cut input sequences longer than *max_length* to *max_length*.

pad_to_multiple_of (`int`, *optional*) : If set, will pad the sequence to a multiple of the provided value.

return_attention_mask (`bool`, *optional*) : Whether or not [__call__()](/docs/transformers/v5.21.0/en/model_doc/audio-spectrogram-transformer#transformers.ASTFeatureExtractor.__call__) should return `attention_mask`.

device (`str` or `torch.device`, *optional*) : The device to compute the audio features on (e.g. "cpu", "cuda"), only relevant for feature extractors that compute them with torch.

return_tensors (`str` or [TensorType](/docs/transformers/v5.21.0/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Acceptable values are: - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

load_audio_backend (`str`, *optional*) : Backend used by `load_audio()` to decode/resample audio referenced by URL/path in `apply_chat_template`. One of `"auto"`, `"torchcodec"`, `"librosa"`, `"torchaudio"`.

Keyword arguments for audio processing.

### Configuration
https://huggingface.co/docs/transformers/v5.21.0/main_classes/configuration.md
