# HYV4VL

## Overview

HYV4VL combines the [HYV4](./hy_v4) mixture-of-experts text backbone with a vision encoder for
image-text and video-text generation. It supports dynamic-resolution visual inputs and uses
multimodal rotary position embeddings (mRoPE) to represent temporal and spatial positions.

## Usage

The examples use the tentative model ID `tencent/Hy4`, which may change before release.
The full model requires multiple GPUs. The examples shard the dense layers with tensor parallelism and
distribute the experts with expert parallelism, which is more efficient for this mixture-of-experts model
than tensor parallelism alone.

Save either example as `generate.py` and launch it with one process per GPU. For a single-node
allocation, set `NUM_GPUS` to the number of GPUs to use and run:

```bash
torchrun --standalone --nproc-per-node="$NUM_GPUS" generate.py
```

For multi-node allocations, configure your distributed launcher's node count, node rank, and
rendezvous address. Model loading initializes distributed execution and selects each rank's device.

### Image inputs

```python
import os

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, DistributedConfig

model_id = "tencent/Hy4"
processor = AutoProcessor.from_pretrained(model_id)

distributed_config = DistributedConfig(
    tp_size=int(os.environ["WORLD_SIZE"]),  # Number of devices across which the dense layers are sharded
    ep_size=int(os.environ["WORLD_SIZE"]),  # Number of devices across which the experts are distributed
)
model = AutoModelForImageTextToText.from_pretrained(model_id, distributed_config=distributed_config)

messages = [{
    "role": "user",
    "content": [
        {
            "type": "image",
            "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
        },
        {"type": "text", "text": "Describe this image in one sentence."},
    ],
}]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    reasoning_effort="no_think",
    return_dict=True,
    return_tensors="pt",
    processor_kwargs={"images_kwargs": {"device": model.device}},
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)

if torch.distributed.get_rank() == 0:
    completion = outputs[0, inputs["input_ids"].shape[1]:]

    print(processor.decode(completion, skip_special_tokens=True))
```

For text-only generation, use a message containing only a `"text"` item.

### Video inputs

Video-file decoding requires TorchCodec and its FFmpeg runtime dependencies. This example
samples four frames; set `fps=None` when providing `num_frames` to disable the default frame rate.

```python
import os

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, DistributedConfig

model_id = "tencent/Hy4"
processor = AutoProcessor.from_pretrained(model_id)

distributed_config = DistributedConfig(
    tp_size=int(os.environ["WORLD_SIZE"]),  # Number of devices across which the dense layers are sharded
    ep_size=int(os.environ["WORLD_SIZE"]),  # Number of devices across which the experts are distributed
)
model = AutoModelForImageTextToText.from_pretrained(model_id, distributed_config=distributed_config)

messages = [{
    "role": "user",
    "content": [
        {
            "type": "video",
            "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4",
        },
        {"type": "text", "text": "Summarize the action in this video."},
    ],
}]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    reasoning_effort="no_think",
    return_dict=True,
    return_tensors="pt",
    processor_kwargs={
        "videos_kwargs": {
            "do_sample_frames": True,
            "num_frames": 4,
            "fps": None,
            "device": model.device,
        },
    },
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)

if torch.distributed.get_rank() == 0:
    completion = outputs[0, inputs["input_ids"].shape[1]:]

    print(processor.decode(completion, skip_special_tokens=True))
```

## Usage Tips

- The chat template defaults to `reasoning_effort="high"`. Use `reasoning_effort="no_think"` for
  a direct answer. For reasoning, allow enough generated tokens for both reasoning and the final answer.
- Load the processor from the checkpoint to use its image and video settings. Images use
  [HunYuanVLImageProcessor](/docs/transformers/v5.24.1/en/model_doc/hunyuan_vl#transformers.HunYuanVLImageProcessor); video preprocessing uses [HYV4VLVideoProcessor](/docs/transformers/v5.24.1/en/model_doc/hy_v4_vl#transformers.HYV4VLVideoProcessor).
- Pass image and video overrides separately through `processor_kwargs["images_kwargs"]` and
  `processor_kwargs["videos_kwargs"]`. Image `size["longest_edge"]` is a per-image pixel budget;
  video `size["longest_edge"]` is a whole-clip pixel budget, so sampling more frames can reduce
  each frame's resolution. Video `max_image_size["longest_edge"]` additionally caps each frame's
  pixel area, not its edge length.
- Video sampling is disabled by default. Enable `do_sample_frames` and pass either `num_frames`
  or `fps`; frame-rate sampling requires source frame-rate metadata.
- For batches with different text lengths, pass `padding=True` in `processor_kwargs["text_kwargs"]`.

## HYV4VLConfig[[transformers.HYV4VLConfig]]

#### transformers.HYV4VLConfig[[transformers.HYV4VLConfig]]

```python
transformers.HYV4VLConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, text_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, vision_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, image_token_id: int = 120120, video_token_id: int = 120124, tie_word_embeddings: bool = False)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/configuration_hy_v4_vl.py#L216)

**Parameters:**

text_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the text backbone.

vision_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the vision backbone.

image_token_id (`int`, *optional*, defaults to `120120`) : The image token index used as a placeholder for input images.

video_token_id (`int`, *optional*, defaults to `120124`) : The video token index used as a placeholder for input videos.

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

This is the configuration class to store the configuration of a HYV4VLModel. It is used to instantiate a Hy V4 Vl
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [tencent/Hy4](https://huggingface.co/tencent/Hy4)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.24.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.24.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## HYV4VLTextConfig[[transformers.HYV4VLTextConfig]]

#### transformers.HYV4VLTextConfig[[transformers.HYV4VLTextConfig]]

```python
transformers.HYV4VLTextConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vocab_size: int = 120832, hidden_size: int = 6144, intermediate_size: int = 18432, moe_intermediate_size: int = 2048, num_hidden_layers: int = 78, num_attention_heads: int = 64, num_key_value_heads: int = 32, head_dim: int = 256, hidden_act: str = 'silu', max_position_embeddings: int = 262144, initializer_range: float = 0.006, rms_norm_eps: float = 1e-05, use_cache: bool = True, pad_token_id: int | None = 120002, bos_token_id: int | None = 120000, eos_token_id: int | list[int] | None = 120025, tie_word_embeddings: bool = False, attention_bias: bool = False, attention_dropout: float = 0.0, n_routed_experts: int = 256, output_router_logits: bool = False, n_shared_experts: int = 1, num_experts_per_tok: int = 8, routed_scaling_factor: float = 2.827, norm_topk_prob: bool = True, n_group: int = 1, topk_group: int = 1, q_lora_rank: int = 2048, kv_lora_rank: int = 512, qk_nope_head_dim: int = 192, qk_rope_head_dim: int = 64, v_head_dim: int = 256, mlp_layer_types: list[str] | None = None, layer_types: list[str] | None = None, index_topk: int = 2048, index_head_dim: int = 128, index_n_heads: int = 32, indexer_types: list[str] | None = None, hc_mult: int = 4, hc_magnitude: float = 2.0, hc_eps: float = 1e-06, learnable_sink_init: float = 0.0, swiglu_limit: float = 10.0, rope_parameters: transformers.modeling_rope_utils.RopeParameters | dict | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/configuration_hy_v4_vl.py#L67)

**Parameters:**

vocab_size (`int`, *optional*, defaults to `120832`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

hidden_size (`int`, *optional*, defaults to `6144`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `18432`) : Dimension of the MLP representations.

moe_intermediate_size (`int`, *optional*, defaults to `2048`) : Intermediate size of the routed expert MLPs.

num_hidden_layers (`int`, *optional*, defaults to `78`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `64`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*, defaults to `32`) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

head_dim (`int`, *optional*, defaults to `256`) : The attention head dimension. If None, it will default to hidden_size // num_attention_heads

hidden_act (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

max_position_embeddings (`int`, *optional*, defaults to `262144`) : The maximum sequence length that this model might ever be used with.

initializer_range (`float`, *optional*, defaults to `0.006`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rms_norm_eps (`float`, *optional*, defaults to `1e-05`) : The epsilon used by the rms normalization layers.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

pad_token_id (`int`, *optional*, defaults to `120002`) : Token id used for padding in the vocabulary.

bos_token_id (`int`, *optional*, defaults to `120000`) : Token id used for beginning-of-stream in the vocabulary.

eos_token_id (`Union[int, list[int]]`, *optional*, defaults to `120025`) : Token id used for end-of-stream in the vocabulary.

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

attention_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

attention_dropout (`float`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

n_routed_experts (`int`, *optional*, defaults to `256`) : Number of routed experts.

output_router_logits (`bool`, *optional*, defaults to `False`) : Whether or not the router logits should be returned by the model. Enabling this will also allow the model to output the auxiliary loss, including load balancing loss and router z-loss.

n_shared_experts (`int`, *optional*, defaults to `1`) : Number of shared experts.

num_experts_per_tok (`int`, *optional*, defaults to `8`) : Number of experts to route each token to. This is the top-k value for the token-choice routing.

routed_scaling_factor (`float`, *optional*, defaults to `2.827`) : Scaling factor or routed experts.

norm_topk_prob (`bool`, *optional*, defaults to `True`) : Whether to normalize the weights of the routed experts. 

n_group (`int`, *optional*, defaults to 1) : Number of expert groups for routing.

topk_group (`int`, *optional*, defaults to 1) : Number of expert groups kept during routing.

q_lora_rank (`int`, *optional*, defaults to `2048`) : Rank of the LoRA matrices for query projections.

kv_lora_rank (`int`, *optional*, defaults to `512`) : Rank of the LoRA matrices for key and value projections.

qk_nope_head_dim (`int`, *optional*, defaults to `192`) : Dimension of the query/key heads that don't use rotary position embeddings.

qk_rope_head_dim (`int`, *optional*, defaults to `64`) : Dimension of the query/key heads that use rotary position embeddings.

v_head_dim (`int`, *optional*, defaults to `256`) : Dimension of the value heads.

mlp_layer_types (`list[str]`, *optional*) : Per-layer MLP type, either `"dense"` or `"sparse"`. Defaults to one dense layer followed by sparse layers.

layer_types (`list[str]`, *optional*) : A list that explicitly maps each layer index with its layer type. If not provided, it will be automatically generated based on config values.

index_topk (`int`, *optional*, defaults to 2048) : Maximum number of key positions selected by each DSA query.

index_head_dim (`int`, *optional*, defaults to 128) : Hidden dimension of each DSA indexer head.

index_n_heads (`int`, *optional*, defaults to 32) : Number of DSA indexer heads.

indexer_types (`list[str]`, *optional*) : Per-layer DSA indexer type, either `"full"` or `"shared"` (reuses the previous full layer's selection).

hc_mult (`int`, *optional*, defaults to 4) : Number of hidden-state channels maintained by iHC.

hc_magnitude (`float`, *optional*, defaults to 2.0) : Scale applied to the iHC post-gating branch.

hc_eps (`float`, *optional*, defaults to 1e-6) : Numerical epsilon added to iHC sigmoid gates.

learnable_sink_init (`float`, *optional*, defaults to 0.0) : Initial value of each learned attention-sink logit.

swiglu_limit (`float`, *optional*, defaults to 10.0) : Magnitude of the routed-expert SwiGLU clamp. Values at or below zero disable the clamp.

rope_parameters (`Union[~modeling_rope_utils.RopeParameters, dict]`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

This is the configuration class to store the configuration of a HYV4VLModel. It is used to instantiate a Hy V4 Vl
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [tencent/Hy4](https://huggingface.co/tencent/Hy4)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.24.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.24.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## HYV4VLVisionConfig[[transformers.HYV4VLVisionConfig]]

#### transformers.HYV4VLVisionConfig[[transformers.HYV4VLVisionConfig]]

```python
transformers.HYV4VLVisionConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, hidden_act: str = 'gelu', hidden_size: int = 1536, intermediate_size: int = 6144, interpolate_mode: str = 'bilinear', rms_norm_eps: float = 1e-05, attention_dropout: float = 0.0, num_attention_heads: int = 16, num_key_value_heads: int | None = None, num_channels: int = 3, num_hidden_layers: int = 40, out_hidden_size: int = 6144, patch_size: int = 16, spatial_merge_size: int = 2, max_image_size: int = 2048)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/configuration_hy_v4_vl.py#L29)

**Parameters:**

hidden_act (`str`, *optional*, defaults to `gelu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

hidden_size (`int`, *optional*, defaults to `1536`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `6144`) : Dimension of the MLP representations.

interpolate_mode (`str`, *optional*, defaults to `"bilinear"`) : Interpolation mode for the learned patch positional embeddings.

rms_norm_eps (`float`, *optional*, defaults to `1e-05`) : The epsilon used by the rms normalization layers.

attention_dropout (`float`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

num_attention_heads (`int`, *optional*, defaults to `16`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

num_channels (`int`, *optional*, defaults to `3`) : The number of input channels.

num_hidden_layers (`int`, *optional*, defaults to `40`) : Number of hidden layers in the Transformer decoder.

out_hidden_size (`int`, *optional*, defaults to 6144) : Width of the visual features passed to the text backbone.

patch_size (`int`, *optional*, defaults to `16`) : The size (resolution) of each patch.

spatial_merge_size (`int`, *optional*, defaults to `2`) : The size of the spatial merge window used to reduce the number of visual tokens by merging neighboring patches.

max_image_size (`int`, *optional*, defaults to 2048) : Image size used to define the learned positional embedding grid.

This is the configuration class to store the configuration of a HYV4VLModel. It is used to instantiate a Hy V4 Vl
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [tencent/Hy4](https://huggingface.co/tencent/Hy4)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.24.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.24.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## HYV4VLProcessor[[transformers.HYV4VLProcessor]]

#### transformers.HYV4VLProcessor[[transformers.HYV4VLProcessor]]

```python
transformers.HYV4VLProcessor(image_processor = None, tokenizer = None, video_processor = None, chat_template = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/processing_hy_v4_vl.py#L25)

**Parameters:**

image_processor (`HYV4VLImageProcessor`) : The image processor is a required input.

tokenizer (`tokenizer_class`) : The tokenizer is a required input.

video_processor (`HYV4VLVideoProcessor`) : The video processor is a required input.

chat_template (`str`) : A Jinja template to convert lists of messages in a chat into a tokenizable string.

Constructs a HYV4VLProcessor which wraps a image processor, a tokenizer, and a video processor into a single processor.

[HYV4VLProcessor](/docs/transformers/v5.24.1/en/model_doc/hy_v4_vl#transformers.HYV4VLProcessor) offers all the functionalities of [HYV4VLImageProcessor](/docs/transformers/v5.24.1/en/model_doc/hy_v4_vl#transformers.HYV4VLImageProcessor), `tokenizer_class`, and [HYV4VLVideoProcessor](/docs/transformers/v5.24.1/en/model_doc/hy_v4_vl#transformers.HYV4VLVideoProcessor). See the
[~HYV4VLImageProcessor](/docs/transformers/v5.24.1/en/model_doc/hy_v4_vl#transformers.HYV4VLImageProcessor), `~tokenizer_class`, and [~HYV4VLVideoProcessor](/docs/transformers/v5.24.1/en/model_doc/hy_v4_vl#transformers.HYV4VLVideoProcessor) for more information.

#### __call__[[transformers.HYV4VLProcessor.__call__]]

```python
__call__(images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None, text: str | list[str] | list[list[str]] | None = None, videos: typing.Union[list['PIL.Image.Image'], numpy.ndarray, ForwardRef('torch.Tensor'), list[numpy.ndarray], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list[numpy.ndarray]], list[list['torch.Tensor']], transformers.video_utils.URL, list[transformers.video_utils.URL], list[list[transformers.video_utils.URL]], transformers.video_utils.Path, list[transformers.video_utils.Path], list[list[transformers.video_utils.Path]], NoneType] = None, audio: typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence['torch.Tensor'], NoneType] = None, **kwargs: *<class 'transformers.processing_utils.ProcessingKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/processing_utils.py#L671)

**Parameters:**

images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]`, *optional*) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

text (`Union[str, list[str], list[list[str]]]`, *optional*) : The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings (pretokenized string). If you pass a pretokenized input, set `is_split_into_words=True` to avoid ambiguity with batched inputs.

videos (`Union[list[PIL.Image.Image], numpy.ndarray, torch.Tensor, list[numpy.ndarray], list[torch.Tensor], list[list[PIL.Image.Image]], list[list[numpy.ndarray]], list[list[torch.Tensor]], ~video_utils.URL, list[~video_utils.URL], list[list[~video_utils.URL]], ~video_utils.Path, list[~video_utils.Path], list[list[~video_utils.Path]]]`, *optional*) : Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If passing in videos with pixel values between 0 and 1, set `do_rescale=False`.

audio (`Union[numpy.ndarray, torch.Tensor, collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence[torch.Tensor]]`, *optional*) : The audio or batch of audios to be prepared. Each audio can be a NumPy array or PyTorch tensor. In case of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels, and T is the sample length of the audio.

return_mm_token_type_ids (`bool`, *kwargs*) : Whether to return multimodal token type ids indicating multimodal placeholder token positions.

return_text_replacement_offsets (`bool`, *kwargs*) : Whether to return character offsets for each multimodal placeholder and its replacement.

return_tensors (`str` or [TensorType](/docs/transformers/v5.24.1/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

- ****kwargs** ([ProcessingKwargs](/docs/transformers/v5.24.1/en/main_classes/processors#transformers.ProcessingKwargs), *optional*) : Additional processing options for each modality (text, images, videos, audio). Model-specific parameters are listed above; see the TypedDict class for the complete list of supported arguments.

## HYV4VLImageProcessor[[transformers.HYV4VLImageProcessor]]

#### transformers.HYV4VLImageProcessor[[transformers.HYV4VLImageProcessor]]

```python
transformers.HYV4VLImageProcessor(**kwargs: *<class 'transformers.models.hy_v4_vl.image_processing_hy_v4_vl.HYV4VLImageProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/image_processing_hy_v4_vl.py#L88)

**Parameters:**

do_convert_rgb (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to convert the image to RGB.

do_resize (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to resize the image.

size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*, defaults to `{'shortest_edge' : 262144, 'longest_edge': 4194304}`): Describes the maximum input dimensions to the model.

default_to_square (`bool`, *kwargs*, *optional*, defaults to `False`) : Whether to default to a square image when resizing, if size is an int.

crop_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : Size of the output image after applying `center_crop`.

resample (`Annotated[Union[int, PILImageResampling, NoneType], None]`, *kwargs*, defaults to `3`) : Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.

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

min_pixels (`int`, *kwargs*, *optional*, defaults to `512 * 512`) : The min pixels of the image to resize the image.

max_pixels (`int`, *kwargs*, *optional*, defaults to `2048 * 2048`) : The max pixels of the image to resize the image.

patch_size (`int`, *kwargs*, *optional*, defaults to 16) : The spatial patch size of the vision encoder.

temporal_patch_size (`int`, *kwargs*, *optional*, defaults to 1) : The temporal patch size of the vision encoder.

merge_size (`int`, *kwargs*, *optional*, defaults to 2) : The merge size of the vision encoder to llm encoder.

Constructs a HYV4VLImageProcessor image processor.

#### get_number_of_image_patches[[transformers.HYV4VLImageProcessor.get_number_of_image_patches]]

```python
get_number_of_image_patches(height: int, width: int, images_kwargs: dict | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/image_processing_hy_v4_vl.py#L268)

**Parameters:**

height (`int`) : Height of the input image.

width (`int`) : Width of the input image.

images_kwargs (`dict`, *optional*) : Any kwargs to override defaults of the image processor.

**Returns:** `tuple[int, int]`

Number of image patches per image, as a `(height, width)` grid.

A utility that returns number of image patches for a given image size.

Note: Do not remove this method! It is used by vLLM to infer the number of patches and placeholders
without an image input.

#### patchify[[transformers.HYV4VLImageProcessor.patchify]]

```python
patchify(images: torch.Tensor, patch_size: int, merge_size: int, temporal_patch_size: int)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/image_processing_hy_v4_vl.py#L176)

Patchifies each image into flat layout of shape (`seq_len`, `patch_dim`) so we can concat dynamically shaped pixels.

#### preprocess[[transformers.HYV4VLImageProcessor.preprocess]]

```python
preprocess(images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']], **kwargs: *<class 'transformers.models.hy_v4_vl.image_processing_hy_v4_vl.HYV4VLImageProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/image_processing_hy_v4_vl.py#L139)

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

min_pixels (`int`, *kwargs*, *optional*, defaults to `512 * 512`) : The min pixels of the image to resize the image.

max_pixels (`int`, *kwargs*, *optional*, defaults to `2048 * 2048`) : The max pixels of the image to resize the image.

patch_size (`int`, *kwargs*, *optional*, defaults to 16) : The spatial patch size of the vision encoder.

temporal_patch_size (`int`, *kwargs*, *optional*, defaults to 1) : The temporal patch size of the vision encoder.

merge_size (`int`, *kwargs*, *optional*, defaults to 2) : The merge size of the vision encoder to llm encoder.

**Returns:** `~image_processing_base.BatchFeature`

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

#### resize[[transformers.HYV4VLImageProcessor.resize]]

```python
resize(images: torch.Tensor, size: SizeDict, resample: PILImageResampling | tvF.InterpolationMode | int | None, factor: int, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/image_processing_hy_v4_vl.py#L147)

Resize dynamically based on input image aspect ratio.

## HYV4VLImageProcessorPil[[transformers.HYV4VLImageProcessorPil]]

#### transformers.HYV4VLImageProcessorPil[[transformers.HYV4VLImageProcessorPil]]

```python
transformers.HYV4VLImageProcessorPil(**kwargs: *<class 'transformers.models.hy_v4_vl.image_processing_pil_hy_v4_vl.HYV4VLImageProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/image_processing_pil_hy_v4_vl.py#L85)

**Parameters:**

do_convert_rgb (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to convert the image to RGB.

do_resize (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to resize the image.

size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*, defaults to `{'shortest_edge' : 262144, 'longest_edge': 4194304}`): Describes the maximum input dimensions to the model.

default_to_square (`bool`, *kwargs*, *optional*, defaults to `False`) : Whether to default to a square image when resizing, if size is an int.

crop_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : Size of the output image after applying `center_crop`.

resample (`Annotated[Union[int, PILImageResampling, NoneType], None]`, *kwargs*, defaults to `3`) : Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.

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

min_pixels (`int`, *kwargs*, *optional*, defaults to `512 * 512`) : The min pixels of the image to resize the image.

max_pixels (`int`, *kwargs*, *optional*, defaults to `2048 * 2048`) : The max pixels of the image to resize the image.

patch_size (`int`, *kwargs*, *optional*, defaults to 16) : The spatial patch size of the vision encoder.

temporal_patch_size (`int`, *kwargs*, *optional*, defaults to 1) : The temporal patch size of the vision encoder.

merge_size (`int`, *kwargs*, *optional*, defaults to 2) : The merge size of the vision encoder to llm encoder.

Constructs a HYV4VLImageProcessor image processor.

#### get_number_of_image_patches[[transformers.HYV4VLImageProcessorPil.get_number_of_image_patches]]

```python
get_number_of_image_patches(height: int, width: int, images_kwargs: dict | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/image_processing_pil_hy_v4_vl.py#L261)

**Parameters:**

height (`int`) : Height of the input image.

width (`int`) : Width of the input image.

images_kwargs (`dict`, *optional*) : Any kwargs to override defaults of the image processor.

**Returns:** `tuple[int, int]`

Number of image patches per image, as a `(height, width)` grid.

A utility that returns number of image patches for a given image size.

Note: Do not remove this method! It is used by vLLM to infer the number of patches and placeholders
without an image input.

#### patchify[[transformers.HYV4VLImageProcessorPil.patchify]]

```python
patchify(image: ndarray, patch_size: int, merge_size: int, temporal_patch_size: int)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/image_processing_pil_hy_v4_vl.py#L165)

Patchifies each image into flat layout of shape (`seq_len`, `patch_dim`) so we can concat dynamically shaped pixels.

#### preprocess[[transformers.HYV4VLImageProcessorPil.preprocess]]

```python
preprocess(images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']], **kwargs: *<class 'transformers.models.hy_v4_vl.image_processing_pil_hy_v4_vl.HYV4VLImageProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/image_processing_pil_hy_v4_vl.py#L202)

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

min_pixels (`int`, *kwargs*, *optional*, defaults to `512 * 512`) : The min pixels of the image to resize the image.

max_pixels (`int`, *kwargs*, *optional*, defaults to `2048 * 2048`) : The max pixels of the image to resize the image.

patch_size (`int`, *kwargs*, *optional*, defaults to 16) : The spatial patch size of the vision encoder.

temporal_patch_size (`int`, *kwargs*, *optional*, defaults to 1) : The temporal patch size of the vision encoder.

merge_size (`int`, *kwargs*, *optional*, defaults to 2) : The merge size of the vision encoder to llm encoder.

**Returns:** `~image_processing_base.BatchFeature`

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

#### resize[[transformers.HYV4VLImageProcessorPil.resize]]

```python
resize(image: ndarray, size: SizeDict, resample: PILImageResampling | int | None, factor: int, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/image_processing_pil_hy_v4_vl.py#L136)

Resize dynamically based on input image aspect ratio.

## HYV4VLVideoProcessor[[transformers.HYV4VLVideoProcessor]]

#### transformers.HYV4VLVideoProcessor[[transformers.HYV4VLVideoProcessor]]

```python
transformers.HYV4VLVideoProcessor(**kwargs: *<class 'transformers.models.hy_v4_vl.video_processing_hy_v4_vl.HYV4VLVideoProcessorInitKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/video_processing_hy_v4_vl.py#L102)

**Parameters:**

- ****kwargs** (`HYV4VLVideoProcessorInitKwargs`, *optional*) : Additional processing options for each modality (text, images, videos, audio). Model-specific parameters are listed above; see the TypedDict class for the complete list of supported arguments.

Constructs a HYV4VLVideoProcessor video processor.

#### preprocess[[transformers.HYV4VLVideoProcessor.preprocess]]

```python
preprocess(videos: typing.Union[list['PIL.Image.Image'], numpy.ndarray, ForwardRef('torch.Tensor'), list[numpy.ndarray], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list[numpy.ndarray]], list[list['torch.Tensor']], transformers.video_utils.URL, list[transformers.video_utils.URL], list[list[transformers.video_utils.URL]], transformers.video_utils.Path, list[transformers.video_utils.Path], list[list[transformers.video_utils.Path]]], **kwargs: *<class 'transformers.processing_utils.VideosKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/video_processing_utils.py#L98)

**Parameters:**

videos (`Union[list[PIL.Image.Image], numpy.ndarray, torch.Tensor, list[numpy.ndarray], list[torch.Tensor], list[list[PIL.Image.Image]], list[list[numpy.ndarray]], list[list[torch.Tensor]], ~video_utils.URL, list[~video_utils.URL], list[list[~video_utils.URL]], ~video_utils.Path, list[~video_utils.Path], list[list[~video_utils.Path]]]`) : Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If passing in videos with pixel values between 0 and 1, set `do_rescale=False`.

do_convert_rgb (`bool`, *kwargs*, *optional*) : Whether to convert the image to RGB.

do_resize (`bool`, *kwargs*, *optional*) : Whether to resize the image.

size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : Describes the maximum input dimensions to the model.

default_to_square (`bool`, *kwargs*, *optional*) : Whether to default to a square image when resizing, if size is an int.

resample (`Annotated[Union[int, PILImageResampling, NoneType], None]`, *kwargs*) : Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.

do_rescale (`bool`, *kwargs*, *optional*) : Whether to rescale the image.

rescale_factor (`float`, *kwargs*, *optional*) : Rescale factor to rescale the image by if `do_rescale` is set to `True`.

do_normalize (`bool`, *kwargs*, *optional*) : Whether to normalize the image.

image_mean (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*) : Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.

image_std (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*) : Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to `True`.

do_center_crop (`bool`, *kwargs*, *optional*) : Whether to center crop the image.

do_pad (`bool`, *kwargs*, *optional*) : Whether to pad the image. Padding is done either to the largest size in the batch or to a fixed square size per image. The exact padding strategy depends on the model.

crop_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : Size of the output image after applying `center_crop`.

data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.

input_data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format. - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

device (`Annotated[Union[str, torch.device, NoneType], None]`, *kwargs*) : The device to process the videos on. If unset, the device is inferred from the input videos.

do_sample_frames (`bool`, *kwargs*, *optional*) : Whether to sample frames from the video before processing or to process the whole video.

video_metadata (`Annotated[~video_utils.VideoMetadata | dict | list[dict | ~video_utils.VideoMetadata] | list[list[dict | ~video_utils.VideoMetadata]] | None, None]`, *kwargs*) : Metadata of the video containing information about total duration, fps and total number of frames. It will be used to sample frames from video or compute timestamps. Don't pass any metadata unless you are trying to decode the video manually before processing

fps (`Annotated[int | float | None, None]`, *kwargs*) : Target frames to sample per second when `do_sample_frames=True`.

num_frames (`Annotated[int | None, None]`, *kwargs*) : Maximum number of frames to sample when `do_sample_frames=True`.

return_metadata (`bool`, *kwargs*, *optional*) : Whether to return video metadata or not. Video metadats is an object containing info about video duration, fps, decoding backend, etc.

return_tensors (`Annotated[str | ~utils.generic.TensorType | None, None]`, *kwargs*) : Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors.

**Returns:** `~image_processing_base.BatchFeature`

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

## HYV4VLVisionModel[[transformers.HYV4VLVisionModel]]

#### transformers.HYV4VLVisionModel[[transformers.HYV4VLVisionModel]]

```python
transformers.HYV4VLVisionModel(config: HYV4VLVisionConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py#L1003)

**Parameters:**

config ([HYV4VLVisionConfig](/docs/transformers/v5.24.1/en/model_doc/hy_v4_vl#transformers.HYV4VLVisionConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Hy V4 Vl Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.HYV4VLVisionModel.forward]]

```python
forward(pixel_values: Tensor, grid_thw: LongTensor, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py#L1032)

**Parameters:**

pixel_values (`torch.Tensor` of shape `(num_patches, num_channels * patch_size * patch_size)`) : Flat per-patch pixel features produced by the image processor.

grid_thw (`torch.LongTensor` of shape `(num_images, 3)`) : The temporal, height and width dimensions for each image. Each row contains `[t, h, w]` patch counts.

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.24.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.Tensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.24.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([HYV4VLConfig](/docs/transformers/v5.24.1/en/model_doc/hy_v4_vl#transformers.HYV4VLConfig)) and inputs.

The [HYV4VLVisionModel](/docs/transformers/v5.24.1/en/model_doc/hy_v4_vl#transformers.HYV4VLVisionModel) forward method, overrides the `__call__` special method.

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

## HYV4VLTextModel[[transformers.HYV4VLTextModel]]

#### transformers.HYV4VLTextModel[[transformers.HYV4VLTextModel]]

```python
transformers.HYV4VLTextModel(config: HYV4VLTextConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py#L1137)

**Parameters:**

config ([HYV4VLTextConfig](/docs/transformers/v5.24.1/en/model_doc/hy_v4_vl#transformers.HYV4VLTextConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Hy V4 Vl Text Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.HYV4VLTextModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, cache: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.Tensor] = None, use_cache: bool | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py#L1157)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.24.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.24.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.24.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

cache (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `cache` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.24.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `cache` are passed, [DynamicCache](/docs/transformers/v5.24.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `cache` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `cache` key value states are returned and can be used to speed up decoding (see `cache`).

**Returns:** `MoeModelOutputWithPast` or `tuple(torch.Tensor)`

A `MoeModelOutputWithPast` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

The [HYV4VLTextModel](/docs/transformers/v5.24.1/en/model_doc/hy_v4_vl#transformers.HYV4VLTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **cache** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.24.1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

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
- **router_logits** (`tuple(torch.Tensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

  Raw router logits (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
  loss for Mixture of Experts models.

## HYV4VLModel[[transformers.HYV4VLModel]]

#### transformers.HYV4VLModel[[transformers.HYV4VLModel]]

```python
transformers.HYV4VLModel(config: HYV4VLConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py#L1230)

**Parameters:**

config ([HYV4VLConfig](/docs/transformers/v5.24.1/en/model_doc/hy_v4_vl#transformers.HYV4VLConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Hy V4 Vl Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.HYV4VLModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, cache: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.Tensor] = None, use_cache: bool | None = None, pixel_values: typing.Optional[torch.Tensor] = None, pixel_values_videos: typing.Optional[torch.Tensor] = None, image_grid_thw: typing.Optional[torch.LongTensor] = None, video_grid_thw: typing.Optional[torch.LongTensor] = None, mm_token_type_ids: typing.Optional[torch.IntTensor] = None, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py#L1520)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.24.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.24.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.24.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

cache (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `cache` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.24.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `cache` are passed, [DynamicCache](/docs/transformers/v5.24.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `cache` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `cache` key value states are returned and can be used to speed up decoding (see `cache`).

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses `image_processor_class` for processing images).

pixel_values_videos (`torch.Tensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) : The tensors corresponding to the input video. Pixel values for videos can be obtained using `video_processor_class`. See `video_processor_class.__call__` for details (`processor_class` uses `video_processor_class` for processing videos).

image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) : The temporal, height and width of feature shape of each image in LLM.

video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) : The temporal, height and width of feature shape of each video in LLM.

mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens matching each modality. For example text (0), image (1), video (2). Multimodal token type ids can be obtained using [AutoProcessor](/docs/transformers/v5.24.1/en/model_doc/auto#transformers.AutoProcessor). See [ProcessorMixin.__call__()](/docs/transformers/v5.24.1/en/main_classes/processors#transformers.ProcessorMixin.__call__) for details. 

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** `MultimodalMoeBaseModelOutputWithPast` or `tuple(torch.Tensor)`

A `MultimodalMoeBaseModelOutputWithPast` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

The [HYV4VLModel](/docs/transformers/v5.24.1/en/model_doc/hy_v4_vl#transformers.HYV4VLModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the language model backbone.
  If `cache` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **cache** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.24.1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `cache`
  input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
  Hidden-states of the language model backbone at the output of each layer plus the optional initial
  embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attention weights of the language model backbone, after the attention softmax.
- **router_logits** (`tuple(torch.Tensor)`, *optional*, returned when `output_router_logits=True` is passed or when `config.output_router_logits=True`) -- Tuple of `torch.Tensor` (one for each sparse layer) of shape `(batch_size * sequence_length, num_experts)`.
  Raw router logits computed by the MoE routers. They can be used to compute a load-balancing loss or to
  inspect how tokens are dispatched to experts.
- **image_outputs** (`BaseModelOutputWithPooling`, *optional*) -- Output of the image encoder: `last_hidden_state`, `pooler_output`, `hidden_states` and `attentions`.
  `None` if no images were processed.
- **video_outputs** (`BaseModelOutputWithPooling`, *optional*) -- Output of the video encoder: `last_hidden_state`, `pooler_output`, `hidden_states` and `attentions`.
  `None` if no videos were processed.
- **audio_outputs** (`BaseModelOutputWithPooling`, *optional*) -- Output of the audio encoder: `last_hidden_state`, `pooler_output`, `hidden_states` and `attentions`.
  `None` if no audio was processed.

#### get_image_features[[transformers.HYV4VLModel.get_image_features]]

```python
get_image_features(pixel_values: Tensor, image_grid_thw: typing.Optional[torch.LongTensor] = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py#L1413)

**Parameters:**

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input images. Pixel values can be obtained using [HYV4VLImageProcessor](/docs/transformers/v5.24.1/en/model_doc/hy_v4_vl#transformers.HYV4VLImageProcessor). See [HYV4VLImageProcessor.__call__()](/docs/transformers/v5.24.1/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([HYV4VLProcessor](/docs/transformers/v5.24.1/en/model_doc/hy_v4_vl#transformers.HYV4VLProcessor) uses [HYV4VLImageProcessor](/docs/transformers/v5.24.1/en/model_doc/hy_v4_vl#transformers.HYV4VLImageProcessor) for processing images).

image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) : The temporal, height and width of feature shape of each image in LLM.

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.24.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.Tensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.24.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([HYV4VLConfig](/docs/transformers/v5.24.1/en/model_doc/hy_v4_vl#transformers.HYV4VLConfig)) and inputs.

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

#### get_video_features[[transformers.HYV4VLModel.get_video_features]]

```python
get_video_features(pixel_values_videos: Tensor, video_grid_thw: typing.Optional[torch.LongTensor] = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py#L1401)

**Parameters:**

pixel_values_videos (`torch.Tensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`) : The tensors corresponding to the input video. Pixel values for videos can be obtained using [HYV4VLVideoProcessor](/docs/transformers/v5.24.1/en/model_doc/hy_v4_vl#transformers.HYV4VLVideoProcessor). See [HYV4VLVideoProcessor.__call__()](/docs/transformers/v5.24.1/en/model_doc/molmo2#transformers.Molmo2VideoProcessor.__call__) for details ([HYV4VLProcessor](/docs/transformers/v5.24.1/en/model_doc/hy_v4_vl#transformers.HYV4VLProcessor) uses [HYV4VLVideoProcessor](/docs/transformers/v5.24.1/en/model_doc/hy_v4_vl#transformers.HYV4VLVideoProcessor) for processing videos).

video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) : The temporal, height and width of feature shape of each video in LLM.

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.24.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.Tensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.24.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([HYV4VLConfig](/docs/transformers/v5.24.1/en/model_doc/hy_v4_vl#transformers.HYV4VLConfig)) and inputs.

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

## HYV4VLForConditionalGeneration[[transformers.HYV4VLForConditionalGeneration]]

#### transformers.HYV4VLForConditionalGeneration[[transformers.HYV4VLForConditionalGeneration]]

```python
transformers.HYV4VLForConditionalGeneration(config: HYV4VLConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py#L1602)

**Parameters:**

config ([HYV4VLConfig](/docs/transformers/v5.24.1/en/model_doc/hy_v4_vl#transformers.HYV4VLConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Hy V4 Vl Model for token generation conditioned on other modalities (e.g. image-text-to-text generation).

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.HYV4VLForConditionalGeneration.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, cache: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.Tensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, pixel_values: typing.Optional[torch.Tensor] = None, pixel_values_videos: typing.Optional[torch.Tensor] = None, image_grid_thw: typing.Optional[torch.LongTensor] = None, video_grid_thw: typing.Optional[torch.LongTensor] = None, mm_token_type_ids: typing.Optional[torch.IntTensor] = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/hy_v4_vl/modeling_hy_v4_vl.py#L1643)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.24.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.24.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.24.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

cache (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `cache` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.24.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `cache` are passed, [DynamicCache](/docs/transformers/v5.24.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `cache` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

use_cache (`bool`, *optional*) : If set to `True`, `cache` key value states are returned and can be used to speed up decoding (see `cache`).

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses `image_processor_class` for processing images).

pixel_values_videos (`torch.Tensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) : The tensors corresponding to the input video. Pixel values for videos can be obtained using `video_processor_class`. See `video_processor_class.__call__` for details (`processor_class` uses `video_processor_class` for processing videos).

image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) : The temporal, height and width of feature shape of each image in LLM.

video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) : The temporal, height and width of feature shape of each video in LLM.

mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens matching each modality. For example text (0), image (1), video (2). Multimodal token type ids can be obtained using [AutoProcessor](/docs/transformers/v5.24.1/en/model_doc/auto#transformers.AutoProcessor). See [ProcessorMixin.__call__()](/docs/transformers/v5.24.1/en/main_classes/processors#transformers.ProcessorMixin.__call__) for details. 

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** `MoeCausalLMOutputWithPast` or `tuple(torch.Tensor)`

A `MoeCausalLMOutputWithPast` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

The [HYV4VLForConditionalGeneration](/docs/transformers/v5.24.1/en/model_doc/hy_v4_vl#transformers.HYV4VLForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **aux_loss** (`torch.Tensor`, *optional*, returned when `output_router_logits=True` is passed or when `config.output_router_logits=True`, and the model trains its router with a load-balancing loss) -- Load-balancing auxiliary loss for the sparse modules. Models that balance their experts with a router bias
  instead (DeepSeek-V3 and the architectures derived from it) return `None` here while still returning
  `router_logits`.
- **router_logits** (`tuple(torch.Tensor)`, *optional*, returned when `output_router_logits=True` is passed or when `config.output_router_logits=True`) -- Tuple of `torch.Tensor` (one for each sparse layer) of shape `(batch_size * sequence_length, num_experts)`.

  Raw router logits computed by the MoE routers. They can be used to compute a load-balancing loss or to
  inspect how tokens are dispatched to experts.
- **cache** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.24.1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `cache` input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Example:

```python
>>> import os
>>> from transformers import AutoProcessor, DistributedConfig, HYV4VLForConditionalGeneration

>>> model_path = "tencent/Hy4"  # Tentative release identifier.
>>> world_size = int(os.environ["WORLD_SIZE"])
>>> processor = AutoProcessor.from_pretrained(model_path)
>>> model = HYV4VLForConditionalGeneration.from_pretrained(
...     model_path, distributed_config=DistributedConfig(tp_size=world_size, ep_size=world_size),
... )

>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {"type": "image", "image": "path/to/your/image.jpg"},
...             {"type": "text", "text": "Describe this image."},
...         ],
...     }
... ]
>>> inputs = processor.apply_chat_template(
...     messages, tokenize=True, add_generation_prompt=True, reasoning_effort="no_think",
...     return_tensors="pt", return_dict=True,
... ).to(model.device)

>>> generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
>>> generated_trimmed = generated_ids[0][inputs["input_ids"].shape[-1]:]
>>> print(processor.decode(generated_trimmed, skip_special_tokens=True))
```

### TVP
https://huggingface.co/docs/transformers/v5.24.1/model_doc/tvp.md
