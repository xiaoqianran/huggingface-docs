# Molmo2

[Molmo2](https://huggingface.co/papers/2601.10611) is a family of open-weight vision-language models by AllenAI that are state-of-the-art among open-source models, with exceptional capabilities in point-driven grounding for single image, multi-image, and video tasks. The architecture combines a Vision Transformer (ViT) for image processing with an adapter layer connecting vision and text modalities, and a text decoder based on transformer architecture with rotary position embeddings.

The abstract from the paper is the following:

*Today's strongest video-language models (VLMs) remain proprietary. The strongest open-weight models either rely on synthetic data from proprietary VLMs, effectively distilling from them, or do not disclose their training data or recipe. As a result, the open-source community lacks the foundations needed to improve on the state-of-the-art video (and image) language models. Crucially, many downstream applications require more than just high-level video understanding; they require grounding -- either by pointing or by tracking in pixels. Even proprietary models lack this capability. We present Molmo2, a new family of VLMs that are state-of-the-art among open-source models and demonstrate exceptional new capabilities in point-driven grounding in single image, multi-image, and video tasks. Our key contribution is a collection of 7 new video datasets and 2 multi-image datasets, including a dataset of highly detailed video captions for pre-training, a free-form video Q&A dataset for fine-tuning, a new object tracking dataset with complex queries, and an innovative new video pointing dataset, all collected without the use of closed VLMs. We also present a training recipe for this data utilizing an efficient packing and message-tree encoding scheme, and show bi-directional attention on vision tokens and a novel token-weight strategy improves performance. Our best-in-class 8B model outperforms others in the class of open weight and data models on short videos, counting, and captioning, and is competitive on long-videos. On video-grounding Molmo2 significantly outperforms existing open-weight models like Qwen3-VL (35.5 vs 29.6 accuracy on video counting) and surpasses proprietary models like Gemini 3 Pro on some tasks (38.4 vs 20.0 F1 on video pointing and 56.2 vs 41.1 J&F on video tracking).*

You can find all the original Molmo2 checkpoints under the [Molmo2](https://huggingface.co/collections/allenai/molmo2-67d6b5b0e138c5d621de1e5d) collection.

## Usage example

### Image-text-to-text generation

The example below demonstrates how to generate text based on an image with [Pipeline](/docs/transformers/v5.20.1/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v5.20.1/en/model_doc/auto#transformers.AutoModel) class.

```python
from transformers import pipeline

pipeline = pipeline(
    task="image-text-to-text",
    model="allenai/Molmo2-8B",
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
pipeline(text=messages, max_new_tokens=128, return_full_text=False)
```

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

processor = AutoProcessor.from_pretrained("allenai/Molmo2-8B")
model = AutoModelForImageTextToText.from_pretrained(
    "allenai/Molmo2-8B",
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_text = processor.decode(
    generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
)
print(generated_text)
# In this captivating image, a snow leopard is captured ...
```

### Video pointing

Molmo2 can also generate grounded points for video inputs. The points are returned in the generated text as a
`<points coords="...">` payload, where each coordinate group contains a timestamp followed by point ids and normalized
image coordinates.

```python
import torch

from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "allenai/Molmo2-8B"
video_url = "https://storage.googleapis.com/oe-training-public/demo_videos/many_penguins.mp4"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Point to the penguins."},
            {"type": "video", "video": video_url},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)

generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

input_len = inputs["input_ids"].shape[1]
generated_text = processor.decode(generated_ids[0][input_len:], skip_special_tokens=True)
print(generated_text)
```

## Molmo2Config[[transformers.Molmo2Config]]

#### transformers.Molmo2Config[[transformers.Molmo2Config]]

```python
transformers.Molmo2Config(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vision_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, adapter_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, text_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, image_start_token_id: int = 151936, low_res_image_start_token_id: int = 151940, image_end_token_id: int = 151937, image_patch_id: int = 151938, frame_start_token_id: int = 151943, frame_end_token_id: int = 151944, initializer_range: float = 0.02, tie_word_embeddings: bool = False)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/molmo2/configuration_molmo2.py#L171)

**Parameters:**

vision_config (`Molmo2VisionConfig`, *optional*) : Configuration for the vision transformer backbone.

adapter_config (`Molmo2AdapterConfig`, *optional*) : Configuration for the vision-to-language adapter.

text_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the text backbone.

image_start_token_id (`int`, *optional*, defaults to 151936) : Token ID marking the start of an image region.

low_res_image_start_token_id (`int`, *optional*, defaults to 151940) : Token ID marking the start of a low-resolution image crop.

image_end_token_id (`int`, *optional*, defaults to 151937) : Token ID marking the end of an image region.

image_patch_id (`int`, *optional*, defaults to 151938) : Token ID for image patches.

frame_start_token_id (`int`, *optional*, defaults to 151943) : Token ID marking the start of a video frame.

frame_end_token_id (`int`, *optional*, defaults to 151944) : Token ID marking the end of a video frame.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether the model's input and output word embeddings should be tied.

This is the configuration class to store the configuration of a Molmo2Model. It is used to instantiate a Molmo2
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.20.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.20.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Molmo2VisionConfig[[transformers.Molmo2VisionConfig]]

#### transformers.Molmo2VisionConfig[[transformers.Molmo2VisionConfig]]

```python
transformers.Molmo2VisionConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, hidden_size: int = 1152, intermediate_size: int = 4304, num_hidden_layers: int = 25, num_attention_heads: int = 16, num_key_value_heads: int = 16, head_dim: int = 72, hidden_act: str = 'gelu_pytorch_tanh', layer_norm_eps: float = 1e-06, image_size: list[int] | None = None, patch_size: int = 14, num_position_embeddings: int | None = None, num_channels: int = 3, attention_dropout: float = 0.0, attention_bias: bool = True, initializer_range: float = 0.02)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/molmo2/configuration_molmo2.py#L30)

**Parameters:**

hidden_size (`int`, *optional*, defaults to `1152`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `4304`) : Dimension of the MLP representations.

num_hidden_layers (`int`, *optional*, defaults to `25`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `16`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*, defaults to `16`) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

head_dim (`int`, *optional*, defaults to `72`) : The attention head dimension. If None, it will default to hidden_size // num_attention_heads

hidden_act (`str`, *optional*, defaults to `gelu_pytorch_tanh`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

layer_norm_eps (`float`, *optional*, defaults to `1e-06`) : The epsilon used by the layer normalization layers.

image_size (`list[int]`, *optional*) : Input image size as (height, width), `[378, 378]` when not provided.

patch_size (`int`, *optional*, defaults to `14`) : The size (resolution) of each patch.

num_position_embeddings (`int`, *optional*) : Number of positional embeddings, one per image patch. Derived from `image_size` and `patch_size` when not provided.

num_channels (`int`, *optional*, defaults to `3`) : The number of input channels.

attention_dropout (`float`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

attention_bias (`bool`, *optional*, defaults to `True`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a Molmo2Model. It is used to instantiate a Molmo2
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.20.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.20.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Molmo2AdapterConfig[[transformers.Molmo2AdapterConfig]]

#### transformers.Molmo2AdapterConfig[[transformers.Molmo2AdapterConfig]]

```python
transformers.Molmo2AdapterConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vision_feature_layer: list[int] | None = None, hidden_size: int = 1152, num_attention_heads: int = 16, num_key_value_heads: int = 16, head_dim: int = 72, attention_dropout: float = 0.0, attention_bias: bool = True, hidden_act: str = 'silu', intermediate_size: int = 18944, text_hidden_size: int = 3584, mlp_bias: bool = False, initializer_range: float = 0.02)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/molmo2/configuration_molmo2.py#L75)

**Parameters:**

vision_feature_layer (`list[int]`, *optional*) : Indices of the ViT layers whose outputs are concatenated and pooled, `[24, 18]` when not provided. Negative indices count from the last layer.

hidden_size (`int`, *optional*, defaults to `1152`) : Dimension of the hidden representations.

num_attention_heads (`int`, *optional*, defaults to `16`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*, defaults to `16`) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

head_dim (`int`, *optional*, defaults to `72`) : The attention head dimension. If None, it will default to hidden_size // num_attention_heads

attention_dropout (`float`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

attention_bias (`bool`, *optional*, defaults to `True`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

hidden_act (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

intermediate_size (`int`, *optional*, defaults to `18944`) : Dimension of the MLP representations.

text_hidden_size (`int`, *optional*, defaults to 3584) : Hidden size of the text model (used for projection).

mlp_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a Molmo2Model. It is used to instantiate a Molmo2
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.20.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.20.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Molmo2TextConfig[[transformers.Molmo2TextConfig]]

#### transformers.Molmo2TextConfig[[transformers.Molmo2TextConfig]]

```python
transformers.Molmo2TextConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, hidden_size: int = 4096, num_attention_heads: int = 32, num_key_value_heads: int = 8, head_dim: int = 128, vocab_size: int = 151936, additional_vocab_size: int = 128, attention_bias: bool = False, qk_norm_type: str = 'qwen3', num_hidden_layers: int = 36, intermediate_size: int = 12288, hidden_act: str = 'silu', attention_dropout: float = 0.0, max_position_embeddings: int = 36864, rope_parameters: transformers.modeling_rope_utils.RopeParameters | dict | None = None, rope_scaling_layers: list[int] | None = None, rms_norm_eps: float = 1e-06, norm_after: bool = False, initializer_range: float = 0.02, use_cache: bool = True, tie_word_embeddings: bool = False)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/molmo2/configuration_molmo2.py#L111)

**Parameters:**

hidden_size (`int`, *optional*, defaults to `4096`) : Dimension of the hidden representations.

num_attention_heads (`int`, *optional*, defaults to `32`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*, defaults to `8`) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

head_dim (`int`, *optional*, defaults to `128`) : The attention head dimension. If None, it will default to hidden_size // num_attention_heads

vocab_size (`int`, *optional*, defaults to `151936`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

additional_vocab_size (`int`, *optional*, defaults to 128) : Number of additional vocabulary tokens beyond the base vocabulary.

attention_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

qk_norm_type (`str`, *optional*, defaults to `"qwen3"`) : Query/key normalization layout used by the checkpoint. `"qwen3"` normalizes per head; `"olmo"` normalizes the full projected query/key tensors.

num_hidden_layers (`int`, *optional*, defaults to `36`) : Number of hidden layers in the Transformer decoder.

intermediate_size (`int`, *optional*, defaults to `12288`) : Dimension of the MLP representations.

hidden_act (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

attention_dropout (`float`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

max_position_embeddings (`int`, *optional*, defaults to `36864`) : The maximum sequence length that this model might ever be used with.

rope_parameters (`Union[~modeling_rope_utils.RopeParameters, dict]`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

rope_scaling_layers (`list[int]`, *optional*) : Indices of the layers that apply the scaled RoPE described by `rope_parameters`. The remaining layers use an unscaled RoPE with the same theta. All layers are scaled when not provided.

rms_norm_eps (`float`, *optional*, defaults to `1e-06`) : The epsilon used by the rms normalization layers.

norm_after (`bool`, *optional*, defaults to `False`) : Whether to apply layer normalization after the attention/FFN blocks instead of before.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

This is the configuration class to store the configuration of a Molmo2Model. It is used to instantiate a Molmo2
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.20.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.20.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Molmo2Processor[[transformers.Molmo2Processor]]

#### transformers.Molmo2Processor[[transformers.Molmo2Processor]]

```python
transformers.Molmo2Processor(image_processor = None, video_processor = None, tokenizer = None, chat_template: str | None = None, image_use_col_tokens: bool | None = True, use_single_crop_col_tokens: bool | None = None, use_single_crop_start_token: bool | None = True, video_use_col_tokens: bool | None = False, use_frame_special_tokens: bool | None = True, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/molmo2/processing_molmo2.py#L44)

**Parameters:**

image_processor (`Molmo2ImageProcessor`) : The image processor is a required input.

video_processor (`Molmo2VideoProcessor`) : The video processor is a required input.

tokenizer (`TokenizersBackend`) : The tokenizer is a required input.

chat_template (`str`, *optional*) : A Jinja template to convert lists of messages in a chat into a tokenizable string.

image_use_col_tokens (`bool`, *optional*, defaults to `True`) : Whether to append column-separator tokens (`<im_col>`) after each patch row of the high-resolution image view.

use_single_crop_col_tokens (`bool`, *optional*) : Whether to append column-separator tokens after each patch row of the low-resolution (single-crop) image view. If `None`, falls back to `image_use_col_tokens`.

use_single_crop_start_token (`bool`, *optional*, defaults to `True`) : Whether to start the low-resolution image view with `<low_res_im_start>` instead of the regular `<im_start>`.

video_use_col_tokens (`bool`, *optional*, defaults to `False`) : Whether to append column-separator tokens after each patch row of video frames.

use_frame_special_tokens (`bool`, *optional*, defaults to `True`) : Whether to wrap each video frame with `<frame_start>` / `<frame_end>` tokens. If `False`, falls back to `<im_start>` / `<im_end>`.

Constructs a Molmo2Processor which wraps a image processor, a video processor, and a tokenizer into a single processor.

[Molmo2Processor](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2Processor) offers all the functionalities of [Molmo2ImageProcessor](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2ImageProcessor), [Molmo2VideoProcessor](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2VideoProcessor), and [TokenizersBackend](/docs/transformers/v5.20.1/en/main_classes/tokenizer#transformers.TokenizersBackend). See the
[~Molmo2ImageProcessor](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2ImageProcessor), [~Molmo2VideoProcessor](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2VideoProcessor), and [~TokenizersBackend](/docs/transformers/v5.20.1/en/main_classes/tokenizer#transformers.TokenizersBackend) for more information.

#### __call__[[transformers.Molmo2Processor.__call__]]

```python
__call__(images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None, text: str | list[str] | list[list[str]] | None = None, videos: typing.Union[list['PIL.Image.Image'], numpy.ndarray, ForwardRef('torch.Tensor'), list[numpy.ndarray], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list[numpy.ndarray]], list[list['torch.Tensor']], transformers.video_utils.URL, list[transformers.video_utils.URL], list[list[transformers.video_utils.URL]], transformers.video_utils.Path, list[transformers.video_utils.Path], list[list[transformers.video_utils.Path]], NoneType] = None, **kwargs: *<class 'transformers.models.molmo2.processing_molmo2.Molmo2ProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/molmo2/processing_molmo2.py#L103)

**Parameters:**

images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]`, *optional*) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

text (`Union[str, list[str], list[list[str]]]`, *optional*) : The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings (pretokenized string). If you pass a pretokenized input, set `is_split_into_words=True` to avoid ambiguity with batched inputs.

videos (`Union[list[PIL.Image.Image], numpy.ndarray, torch.Tensor, list[numpy.ndarray], list[torch.Tensor], list[list[PIL.Image.Image]], list[list[numpy.ndarray]], list[list[torch.Tensor]], ~video_utils.URL, list[~video_utils.URL], list[list[~video_utils.URL]], ~video_utils.Path, list[~video_utils.Path], list[list[~video_utils.Path]]]`, *optional*) : Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If passing in videos with pixel values between 0 and 1, set `do_rescale=False`.

return_tensors (`str` or [TensorType](/docs/transformers/v5.20.1/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

- ****kwargs** ([ProcessingKwargs](/docs/transformers/v5.20.1/en/main_classes/processors#transformers.ProcessingKwargs), *optional*) : Additional processing options for each modality (text, images, videos, audio). Model-specific parameters are listed above; see the TypedDict class for the complete list of supported arguments.

## Molmo2ImageProcessor[[transformers.Molmo2ImageProcessor]]

#### transformers.Molmo2ImageProcessor[[transformers.Molmo2ImageProcessor]]

```python
transformers.Molmo2ImageProcessor(**kwargs: *<class 'transformers.models.molmo2.image_processing_molmo2.Molmo2ImageProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/molmo2/image_processing_molmo2.py#L193)

**Parameters:**

do_convert_rgb (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to convert the image to RGB.

do_resize (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to resize the image.

size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*, defaults to `{'height' : 378, 'width': 378}`): Describes the maximum input dimensions to the model.

default_to_square (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to default to a square image when resizing, if size is an int.

crop_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : Size of the output image after applying `center_crop`.

resample (`Annotated[Union[int, PILImageResampling, NoneType], None]`, *kwargs*, defaults to `2`) : Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.

do_rescale (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to rescale the image.

rescale_factor (`float`, *kwargs*, *optional*, defaults to `0.00392156862745098`) : Rescale factor to rescale the image by if `do_rescale` is set to `True`.

do_normalize (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to normalize the image.

image_mean (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*, defaults to `[0.5, 0.5, 0.5]`) : Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.

image_std (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*, defaults to `[0.5, 0.5, 0.5]`) : Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to `True`.

do_pad (`bool`, *kwargs*, *optional*) : Whether to pad the image. Padding is done either to the largest size in the batch or to a fixed square size per image. The exact padding strategy depends on the model.

pad_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : The size in `{"height": int, "width" int}` to pad the images to. Must be larger than any image size provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest height and width in the batch. Applied only when `do_pad=True.`

do_center_crop (`bool`, *kwargs*, *optional*) : Whether to center crop the image.

data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.

input_data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format. - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

device (`Annotated[Union[str, torch.device, NoneType], None]`, *kwargs*) : The device to process the videos on. If unset, the device is inferred from the input videos.

return_tensors (`Annotated[str | ~utils.generic.TensorType | None, None]`, *kwargs*) : Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors.

disable_grouping (`bool`, *kwargs*, *optional*) : Whether to disable grouping of images by size to process them individually and not in batches. If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157

image_seq_length (`int`, *kwargs*, *optional*) : The number of image tokens to be used for each image in the input. Added for backward compatibility but this should be set as a processor attribute in future models.

max_crops (`int`, *kwargs*, *optional*, defaults to 8) : Maximum number of crops to use per image.

overlap_margins (`list[int]`, *kwargs*, *optional*, defaults to `[4, 4]`) : Overlap margins (in patches) for overlapping crop extraction.

patch_size (`int`, *kwargs*, *optional*, defaults to 14) : The spatial patch size of the vision encoder.

pooling_size (`list[int]`, *kwargs*, *optional*, defaults to `[2, 2]`) : The pooling size of the vision adapter.

Constructs a Molmo2ImageProcessor image processor.

#### __call__[[transformers.Molmo2ImageProcessor.__call__]]

```python
__call__(images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']], *args, **kwargs: *<class 'transformers.processing_utils.ImagesKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/image_processing_utils.py#L215)

Preprocess an image or a batch of images.

#### preprocess[[transformers.Molmo2ImageProcessor.preprocess]]

```python
preprocess(images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']], **kwargs: *<class 'transformers.models.molmo2.image_processing_molmo2.Molmo2ImageProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/molmo2/image_processing_molmo2.py#L353)

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

max_crops (`int`, *kwargs*, *optional*, defaults to 8) : Maximum number of crops to use per image.

overlap_margins (`list[int]`, *kwargs*, *optional*, defaults to `[4, 4]`) : Overlap margins (in patches) for overlapping crop extraction.

patch_size (`int`, *kwargs*, *optional*, defaults to 14) : The spatial patch size of the vision encoder.

pooling_size (`list[int]`, *kwargs*, *optional*, defaults to `[2, 2]`) : The pooling size of the vision adapter.

**Returns:** `~image_processing_base.BatchFeature`

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

## Molmo2VideoProcessor[[transformers.Molmo2VideoProcessor]]

#### transformers.Molmo2VideoProcessor[[transformers.Molmo2VideoProcessor]]

```python
transformers.Molmo2VideoProcessor(**kwargs: *<class 'transformers.models.molmo2.video_processing_molmo2.Molmo2VideoProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/molmo2/video_processing_molmo2.py#L148)

**Parameters:**

do_convert_rgb (`bool`, *kwargs*, *optional*) : Whether to convert the image to RGB.

do_resize (`bool`, *kwargs*, *optional*) : Whether to resize the image.

size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*, defaults to `{'height' : 378, 'width': 378}`): Describes the maximum input dimensions to the model.

default_to_square (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to default to a square image when resizing, if size is an int.

resample (`Annotated[Union[int, PILImageResampling, NoneType], None]`, *kwargs*, defaults to `2`) : Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.

do_rescale (`bool`, *kwargs*, *optional*) : Whether to rescale the image.

rescale_factor (`float`, *kwargs*, *optional*, defaults to `0.00392156862745098`) : Rescale factor to rescale the image by if `do_rescale` is set to `True`.

do_normalize (`bool`, *kwargs*, *optional*) : Whether to normalize the image.

image_mean (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*, defaults to `[0.5, 0.5, 0.5]`) : Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.

image_std (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*, defaults to `[0.5, 0.5, 0.5]`) : Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to `True`.

do_center_crop (`bool`, *kwargs*, *optional*) : Whether to center crop the image.

do_pad (`bool`, *kwargs*, *optional*) : Whether to pad the image. Padding is done either to the largest size in the batch or to a fixed square size per image. The exact padding strategy depends on the model.

crop_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : Size of the output image after applying `center_crop`.

data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.

input_data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format. - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

device (`Annotated[Union[str, torch.device, NoneType], None]`, *kwargs*) : The device to process the videos on. If unset, the device is inferred from the input videos.

do_sample_frames (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to sample frames from the video before processing or to process the whole video.

video_metadata (`Annotated[~video_utils.VideoMetadata | dict | list[dict | ~video_utils.VideoMetadata] | list[list[dict | ~video_utils.VideoMetadata]] | None, None]`, *kwargs*) : Metadata of the video containing information about total duration, fps and total number of frames. It will be used to sample frames from video or compute timestamps. Don't pass any metadata unless you are trying to decode the video manually before processing

fps (`Annotated[int | float | None, None]`, *kwargs*) : Target frames to sample per second when `do_sample_frames=True`.

num_frames (`Annotated[int | None, None]`, *kwargs*, defaults to `64`) : Maximum number of frames to sample when `do_sample_frames=True`.

return_metadata (`bool`, *kwargs*, *optional*, defaults to `False`) : Whether to return video metadata or not. Video metadats is an object containing info about video duration, fps, decoding backend, etc.

return_tensors (`Annotated[str | ~utils.generic.TensorType | None, None]`, *kwargs*) : If set, will return tensors of a particular framework. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

patch_size (`int`, *kwargs*, *optional*) : Side length in pixels of each ViT patch for video frames.

pooling_size (`list[int]`, *kwargs*, *optional*) : `[pool_h, pool_w]` pooling window applied to video patch features.

max_fps (`int`, *kwargs*, *optional*) : Maximum sampling rate in frames per second for short videos.

- ****kwargs** ([ProcessingKwargs](/docs/transformers/v5.20.1/en/main_classes/processors#transformers.ProcessingKwargs), *optional*) : Additional processing options for each modality (text, images, videos, audio). Model-specific parameters are listed above; see the TypedDict class for the complete list of supported arguments.

Constructs a Molmo2VideoProcessor video processor.

#### __call__[[transformers.Molmo2VideoProcessor.__call__]]

```python
__call__(videos, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/video_processing_utils.py#L86)

## Molmo2Model[[transformers.Molmo2Model]]

#### transformers.Molmo2Model[[transformers.Molmo2Model]]

```python
transformers.Molmo2Model(config: Molmo2Config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/molmo2/modeling_molmo2.py#L735)

**Parameters:**

config ([Molmo2Config](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2Config)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.20.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Molmo2 model which consists of a vision backbone, a pooling adapter and a language model, without a language
modeling head.

This model inherits from [PreTrainedModel](/docs/transformers/v5.20.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Molmo2Model.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, pixel_values: typing.Optional[torch.FloatTensor] = None, image_token_pooling: typing.Optional[torch.Tensor] = None, image_grids: typing.Optional[torch.Tensor] = None, image_num_crops: typing.Optional[torch.Tensor] = None, pixel_values_videos: typing.Optional[torch.Tensor] = None, video_token_pooling: typing.Optional[torch.Tensor] = None, video_grids: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.Tensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, mm_token_type_ids: typing.Optional[torch.LongTensor] = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/molmo2/modeling_molmo2.py#L803)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.20.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.20.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.20.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [Molmo2ImageProcessor](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2ImageProcessor). See [Molmo2ImageProcessor.__call__()](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([Molmo2Processor](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2Processor) uses [Molmo2ImageProcessor](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2ImageProcessor) for processing images).

image_token_pooling (`torch.Tensor` of shape `(num_image_tokens, pool_h * pool_w)`, *optional*) : Indices into the flattened image patch sequence gathered by each pooled image token, with `-1` marking padding slots. Can be obtained using [AutoProcessor](/docs/transformers/v5.20.1/en/model_doc/auto#transformers.AutoProcessor). See [ProcessorMixin.__call__()](/docs/transformers/v5.20.1/en/main_classes/processors#transformers.ProcessorMixin.__call__) for details.

image_grids (`torch.Tensor` of shape `(num_images, 4)`, *optional*) : Per-image `[low_res_h, low_res_w, high_res_h, high_res_w]` pooled token grid. Can be obtained using [AutoProcessor](/docs/transformers/v5.20.1/en/model_doc/auto#transformers.AutoProcessor). See [ProcessorMixin.__call__()](/docs/transformers/v5.20.1/en/main_classes/processors#transformers.ProcessorMixin.__call__) for details.

image_num_crops (`torch.Tensor` of shape `(num_images,)`, *optional*) : Number of crops per image, low-resolution view included. Can be obtained using [AutoProcessor](/docs/transformers/v5.20.1/en/model_doc/auto#transformers.AutoProcessor). See [ProcessorMixin.__call__()](/docs/transformers/v5.20.1/en/main_classes/processors#transformers.ProcessorMixin.__call__) for details.

pixel_values_videos (`torch.Tensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) : The tensors corresponding to the input video. Pixel values for videos can be obtained using [Molmo2VideoProcessor](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2VideoProcessor). See [Molmo2VideoProcessor.__call__()](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2VideoProcessor.__call__) for details ([Molmo2Processor](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2Processor) uses [Molmo2VideoProcessor](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2VideoProcessor) for processing videos).

video_token_pooling (`torch.Tensor` of shape `(num_video_tokens, pool_h * pool_w)`, *optional*) : Indices into the flattened video frame-patch sequence gathered by each pooled video token, with `-1` marking padding slots. Can be obtained using [AutoProcessor](/docs/transformers/v5.20.1/en/model_doc/auto#transformers.AutoProcessor). See [ProcessorMixin.__call__()](/docs/transformers/v5.20.1/en/main_classes/processors#transformers.ProcessorMixin.__call__) for details.

video_grids (`torch.Tensor` of shape `(num_videos, 3)`, *optional*) : Per-video `[num_frames, pooled_h, pooled_w]` token grid. Can be obtained using [AutoProcessor](/docs/transformers/v5.20.1/en/model_doc/auto#transformers.AutoProcessor). See [ProcessorMixin.__call__()](/docs/transformers/v5.20.1/en/main_classes/processors#transformers.ProcessorMixin.__call__) for details.

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.20.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.20.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

mm_token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens matching each modality. For example text (0), image (1), video (2). Multimodal token type ids can be obtained using [AutoProcessor](/docs/transformers/v5.20.1/en/model_doc/auto#transformers.AutoProcessor). See [ProcessorMixin.__call__()](/docs/transformers/v5.20.1/en/main_classes/processors#transformers.ProcessorMixin.__call__) for details. 

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** `Molmo2ModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `Molmo2ModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Molmo2Config](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2Config)) and inputs.

The [Molmo2Model](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.20.1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_hidden_states** (`torch.FloatTensor`, *optional*) -- A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.

## Molmo2TextModel[[transformers.Molmo2TextModel]]

#### transformers.Molmo2TextModel[[transformers.Molmo2TextModel]]

```python
transformers.Molmo2TextModel(config: Molmo2TextConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/molmo2/modeling_molmo2.py#L645)

**Parameters:**

config ([Molmo2TextConfig](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2TextConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.20.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Molmo2 Text Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.20.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Molmo2TextModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/molmo2/modeling_molmo2.py#L670)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.20.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.20.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.20.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.20.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.20.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

**Returns:** [BaseModelOutputWithPast](/docs/transformers/v5.20.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPast](/docs/transformers/v5.20.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Molmo2Config](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2Config)) and inputs.

The [Molmo2TextModel](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2TextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.20.1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

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

## Molmo2Adapter[[transformers.Molmo2Adapter]]

#### transformers.Molmo2Adapter[[transformers.Molmo2Adapter]]

```python
transformers.Molmo2Adapter(config: Molmo2AdapterConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/molmo2/modeling_molmo2.py#L595)

**Parameters:**

config ([Molmo2AdapterConfig](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2AdapterConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.20.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Molmo2 vision adapter: pools ViT patch features with a cross-attention layer and projects them into the
language model's embedding space.

This model inherits from [PreTrainedModel](/docs/transformers/v5.20.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Molmo2Adapter.forward]]

```python
forward(image_features: Tensor, pooled_patches_idx: Tensor, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/molmo2/modeling_molmo2.py#L608)

**Parameters:**

image_features (`torch.Tensor` of shape `(num_crops, num_patches, hidden_size * len(vision_feature_layer))`) : Concatenated intermediate ViT features of every crop.

pooled_patches_idx (`torch.Tensor` of shape `(num_tokens, pool_h * pool_w)`) : Indices into the flattened patch sequence pooled by each output token; `-1` marks padding slots.

**Returns:** [BaseModelOutput](/docs/transformers/v5.20.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

A [BaseModelOutput](/docs/transformers/v5.20.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Molmo2Config](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2Config)) and inputs.

The [Molmo2Adapter](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2Adapter) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## Molmo2VisionModel[[transformers.Molmo2VisionModel]]

#### transformers.Molmo2VisionModel[[transformers.Molmo2VisionModel]]

```python
transformers.Molmo2VisionModel(config: Molmo2VisionConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/molmo2/modeling_molmo2.py#L536)

**Parameters:**

config ([Molmo2VisionConfig](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2VisionConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.20.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Molmo2 Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.20.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Molmo2VisionModel.forward]]

```python
forward(pixel_values: Tensor, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/molmo2/modeling_molmo2.py#L556)

**Parameters:**

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input images. Pixel values can be obtained using [Molmo2ImageProcessor](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2ImageProcessor). See [Molmo2ImageProcessor.__call__()](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([Molmo2Processor](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2Processor) uses [Molmo2ImageProcessor](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2ImageProcessor) for processing images).

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.20.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.20.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Molmo2Config](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2Config)) and inputs.

The [Molmo2VisionModel](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2VisionModel) forward method, overrides the `__call__` special method.

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

## Molmo2ForConditionalGeneration[[transformers.Molmo2ForConditionalGeneration]]

#### transformers.Molmo2ForConditionalGeneration[[transformers.Molmo2ForConditionalGeneration]]

```python
transformers.Molmo2ForConditionalGeneration(config: Molmo2Config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/molmo2/modeling_molmo2.py#L918)

**Parameters:**

config ([Molmo2Config](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2Config)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.20.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Molmo2 model which consists of a vision backbone, a pooling adapter and a language model.

This model inherits from [PreTrainedModel](/docs/transformers/v5.20.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Molmo2ForConditionalGeneration.forward]]

```python
forward(input_ids: LongTensor = None, pixel_values: typing.Optional[torch.Tensor] = None, image_token_pooling: typing.Optional[torch.Tensor] = None, image_grids: typing.Optional[torch.Tensor] = None, image_num_crops: typing.Optional[torch.Tensor] = None, pixel_values_videos: typing.Optional[torch.Tensor] = None, video_token_pooling: typing.Optional[torch.Tensor] = None, video_grids: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: list[torch.FloatTensor] | None = None, mm_token_type_ids: typing.Optional[torch.LongTensor] = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/molmo2/modeling_molmo2.py#L953)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.20.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.20.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.20.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [Molmo2ImageProcessor](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2ImageProcessor). See [Molmo2ImageProcessor.__call__()](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([Molmo2Processor](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2Processor) uses [Molmo2ImageProcessor](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2ImageProcessor) for processing images).

image_token_pooling (`torch.Tensor` of shape `(num_image_tokens, pool_h * pool_w)`, *optional*) : Indices into the flattened image patch sequence gathered by each pooled image token, with `-1` marking padding slots. Can be obtained using [AutoProcessor](/docs/transformers/v5.20.1/en/model_doc/auto#transformers.AutoProcessor). See [ProcessorMixin.__call__()](/docs/transformers/v5.20.1/en/main_classes/processors#transformers.ProcessorMixin.__call__) for details.

image_grids (`torch.Tensor` of shape `(num_images, 4)`, *optional*) : Per-image `[low_res_h, low_res_w, high_res_h, high_res_w]` pooled token grid. Can be obtained using [AutoProcessor](/docs/transformers/v5.20.1/en/model_doc/auto#transformers.AutoProcessor). See [ProcessorMixin.__call__()](/docs/transformers/v5.20.1/en/main_classes/processors#transformers.ProcessorMixin.__call__) for details.

image_num_crops (`torch.Tensor` of shape `(num_images,)`, *optional*) : Number of crops per image, low-resolution view included. Can be obtained using [AutoProcessor](/docs/transformers/v5.20.1/en/model_doc/auto#transformers.AutoProcessor). See [ProcessorMixin.__call__()](/docs/transformers/v5.20.1/en/main_classes/processors#transformers.ProcessorMixin.__call__) for details.

pixel_values_videos (`torch.Tensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) : The tensors corresponding to the input video. Pixel values for videos can be obtained using [Molmo2VideoProcessor](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2VideoProcessor). See [Molmo2VideoProcessor.__call__()](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2VideoProcessor.__call__) for details ([Molmo2Processor](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2Processor) uses [Molmo2VideoProcessor](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2VideoProcessor) for processing videos).

video_token_pooling (`torch.Tensor` of shape `(num_video_tokens, pool_h * pool_w)`, *optional*) : Indices into the flattened video frame-patch sequence gathered by each pooled video token, with `-1` marking padding slots. Can be obtained using [AutoProcessor](/docs/transformers/v5.20.1/en/model_doc/auto#transformers.AutoProcessor). See [ProcessorMixin.__call__()](/docs/transformers/v5.20.1/en/main_classes/processors#transformers.ProcessorMixin.__call__) for details.

video_grids (`torch.Tensor` of shape `(num_videos, 3)`, *optional*) : Per-video `[num_frames, pooled_h, pooled_w]` token grid. Can be obtained using [AutoProcessor](/docs/transformers/v5.20.1/en/model_doc/auto#transformers.AutoProcessor). See [ProcessorMixin.__call__()](/docs/transformers/v5.20.1/en/main_classes/processors#transformers.ProcessorMixin.__call__) for details.

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`list[torch.FloatTensor]`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.20.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.20.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

mm_token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens matching each modality. For example text (0), image (1), video (2). Multimodal token type ids can be obtained using [AutoProcessor](/docs/transformers/v5.20.1/en/model_doc/auto#transformers.AutoProcessor). See [ProcessorMixin.__call__()](/docs/transformers/v5.20.1/en/main_classes/processors#transformers.ProcessorMixin.__call__) for details. 

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** `Molmo2CausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `Molmo2CausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Molmo2Config](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2Config)) and inputs.

The [Molmo2ForConditionalGeneration](/docs/transformers/v5.20.1/en/model_doc/molmo2#transformers.Molmo2ForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.20.1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_hidden_states** (`torch.FloatTensor`, *optional*) -- A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.

Example:

```python
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Molmo2ForConditionalGeneration

>>> model = Molmo2ForConditionalGeneration.from_pretrained("allenai/Molmo2-8B")
>>> processor = AutoProcessor.from_pretrained("allenai/Molmo2-8B")

>>> prompt = "What's the content of the image?"
>>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image}]}]

>>> inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)

>>> # Generate
>>> generated_ids = model.generate(**inputs, max_new_tokens=15)
>>> generated_tokens = generated_ids[:, inputs['input_ids'].size(1):]
>>> processor.post_process_image_text_to_text(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"The image shows a bustling street scene in what appears to be a Chinatown area. There's ..."
```

### HyperCLOVA X
https://huggingface.co/docs/transformers/v5.20.1/model_doc/hyperclovax.md
