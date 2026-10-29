# Muse Spark Open

## Overview

Muse Spark Open is a multimodal mixture-of-experts (MoE) model developed by Meta that accepts text, images, audio, and video as input and generates text. It has 816 billion total parameters, with 42 billion active per token, and supports a context window of up to 1,048,576 tokens shared across all input modalities. The model is trained for agentic coding tasks, including multi-file code editing, debugging, and tool use.

The language backbone consists of 62 transformer layers, each routing tokens to 8 of 256 experts. It uses grouped-query attention with query-key normalization, attention output gating and normalization, and learned residual gates. A shared perception encoder processes images and video frames, while a convolutional audio encoder produces audio representations at 12.5 tokens per second. Muse Spark Open shares its tokenizer and vocabulary with [Muse Glimmer](./muse_glimmer), with extensions to the chat template for multimodal inputs. Checkpoints are available in BF16 and [NVFP4 formats](#nvfp4-weights).

   

## Resources

- [Hugging Face Collection](https://huggingface.co/collections/meta-models/muse-spark-open)
- [Hugging Face Blog](http://hf.co/blog/meta-muse-spark-open)
- [Post-train with TRL](https://github.com/huggingface/trl/tree/main/examples/sft_muse_spark_open)

```python
import os

from transformers import AutoModelForMultimodalLM, AutoProcessor, DistributedConfig

distributed_config = DistributedConfig(
    tp_size=int(os.environ["WORLD_SIZE"]), # Number of devices across which the dense layers are sharded
    ep_size=int(os.environ["WORLD_SIZE"]), # Number of devices across which the experts are distributed
)

processor = AutoProcessor.from_pretrained("meta-models/Muse-Spark-Open-816B-A42B")
model = AutoModelForMultimodalLM.from_pretrained(
    "meta-models/Muse-Spark-Open-816B-A42B",
    distributed_config=distributed_config,
)

messages = [
    {"role": "user", "content": "Explain in one sentence why the sky is usually blue."},
]
inputs = processor.apply_chat_template(
    messages,
    reasoning_effort="low",  # default: xhigh, options: low/medium/high/xhigh
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device).to(model.dtype)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))
#  to=self<|message|>Explain in one sentence why the sky is usually blue.
# ...
# <|eom|><|start|>assistant to=user<|message|>The sky is usually blue because molecules in Earth's atmosphere scatter shorter-wavelength blue sunlight in all directions much more strongly than longer-wavelength red light.<|eot|>
```

```python
import os

from transformers import AutoModelForMultimodalLM, AutoProcessor, DistributedConfig

distributed_config = DistributedConfig(
    tp_size=int(os.environ["WORLD_SIZE"]),
    ep_size=int(os.environ["WORLD_SIZE"]),
)

processor = AutoProcessor.from_pretrained("meta-models/Muse-Spark-Open-816B-A42B")
model = AutoModelForMultimodalLM.from_pretrained(
    "meta-models/Muse-Spark-Open-816B-A42B",
    distributed_config=distributed_config,
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]
inputs = processor.apply_chat_template(
    messages,
    reasoning_effort="medium", # default: xhigh, options: low/medium/high/xhigh
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device).to(model.dtype)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
print(response)
#  to=self<|message|>What is shown in this image?
# ...
# <|eom|><|start|>assistant to=user<|message|>The image shows a **Pallas's cat — also known as a manul, *Otocolobus manul*** — walking through snow.
#
# It is a small, stocky wild cat recognizable by:
#
# * **Its extremely dense, long, greyish-brown winter coat** with faint dark stripes and a dusting of snow on its back
# * **Its flat, broad face** with low-set ears, a short muzzle, and dark stripes on the cheeks
# * **Its short legs and thick, plump body** adapted for cold climates
#
# The cat is captured mid-stride on a snow-covered surface, with one front paw lifted. In the background are white birch tree trunks with black markings and a chain-link fence on the left, indicating the animal is in a zoo or wildlife enclosure in winter.<|eot|>
```

```python
import os

from transformers import AutoModelForMultimodalLM, AutoProcessor, DistributedConfig

distributed_config = DistributedConfig(
    tp_size=int(os.environ["WORLD_SIZE"]),
    ep_size=int(os.environ["WORLD_SIZE"]),
)

processor = AutoProcessor.from_pretrained("meta-models/Muse-Spark-Open-816B-A42B")
model = AutoModelForMultimodalLM.from_pretrained(
    "meta-models/Muse-Spark-Open-816B-A42B",
    distributed_config=distributed_config,
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4"},
            {"type": "text", "text": "What is happening in this video?"},
        ],
    },
]
inputs = processor.apply_chat_template(
    messages,
    reasoning_effort="medium",  # default: xhigh, options: low/medium/high/xhigh
    processor_kwargs={
        "fps": 2,   # default: 2, number of frames sampled per second
        "num_frames": 8, # default: 94, maximum number of frames sampled from the video
    },
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device).to(model.dtype)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))
#  to=self<|message|>What is happening in this video?
# ...
# <|eom|><|start|>assistant to=user<|message|>In this video two people are playing tennis on an indoor court.
#
# The shot is taken from behind one of the players - a man in a white t-shirt, black shorts and white shoes and socks. He is standing near the baseline on an orange hard court with white lines, in a ready position with his knees bent and his racket held in front of him.
#
# Across the net, on the far side of the court, is his opponent, also dressed in light-colored clothing.
#
# The sequence shows a short rally:
#
# *   The opponent hits the ball over the net.
# *   The man in the foreground shuffles to his left, turns his body, and prepares for a shot.
# *   He executes a two-handed backhand, swinging the racket across his body and following through as he hits the ball back over the net.
#
# The setting is a large indoor tennis hall with a high, arched ceiling with exposed wooden beams and overhead fluorescent lights. The court is surrounded by a grey run-off area, green windscreening on the walls, and netting that separates it from adjacent courts. On the right side you can see an umpire's chair and a few yellow and black benches. An "Adobe Stock" watermark is visible in the center of the frame.<|eot|>
```

```python
import os

from transformers import AutoModelForMultimodalLM, AutoProcessor, DistributedConfig

distributed_config = DistributedConfig(
    tp_size=int(os.environ["WORLD_SIZE"]),
    ep_size=int(os.environ["WORLD_SIZE"]),
)

processor = AutoProcessor.from_pretrained("meta-models/Muse-Spark-Open-816B-A42B")
model = AutoModelForMultimodalLM.from_pretrained(
    "meta-models/Muse-Spark-Open-816B-A42B",
    distributed_config=distributed_config,
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "url": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/glass_breaking.mp3"},
            {"type": "text", "text": "What is said in this recording?"},
        ],
    },
]
inputs = processor.apply_chat_template(
    messages,
    reasoning_effort="low", # default: xhigh, options: low/medium/high/xhigh
    processor_kwargs={"sampling_rate": 24000},  # sampling rate the model was trained with
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device).to(model.dtype)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))
#  to=self<|message|>We need to respond to user request: "What is said in this recording?"
# ...
# <|eom|><|start|>assistant to=user<|message|>In the recording the person says:
#
# > "Just picked up this glass and the people at the store told me that if I dropped it it wouldn't break."
#
# Then the glass is dropped and shatters.<|eot|>
```

## NVFP4 Weights

NVFP4 [quantized weights](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4) with lower memory requirements are available on the hub. The model can be loaded on any NVFP4 compatible device without changes to the code. The quantization configuration is automatically applied at load time.

> [!IMPORTANT]
> For NVFP4 support, transformers must be installed from [this branch](https://github.com/huggingface/transformers/pull/49015).
>
> ```bash
> pip install git+https://github.com/huggingface/transformers.git@finegrained-per-scheme-demo
> ```

```python
import os

from transformers import AutoModelForMultimodalLM, AutoProcessor, DistributedConfig

distributed_config = DistributedConfig(
    tp_size=int(os.environ["WORLD_SIZE"]),
    ep_size=int(os.environ["WORLD_SIZE"]),
)

processor = AutoProcessor.from_pretrained("meta-models/Muse-Spark-Open-816B-A42B-NVFP4")
model = AutoModelForMultimodalLM.from_pretrained(
    "meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
    distributed_config=distributed_config,
)

# ...
```

## MuseSparkConfig[[transformers.MuseSparkConfig]]

#### transformers.MuseSparkConfig[[transformers.MuseSparkConfig]]

```python
transformers.MuseSparkConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, text_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, vision_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, image_token_id: int = 200092, video_token_id: int = 200091, out_hidden_size: int = 6144, projector_hidden_size: int = 4096, projector_hidden_act: str = 'gelu', audio_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, audio_token_id: int = 200038)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/muse_spark/configuration_muse_spark.py#L251)

**Parameters:**

text_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the text backbone.

vision_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the vision backbone.

image_token_id (`int`, *optional*, defaults to `200092`) : The image token index used as a placeholder for input images.

video_token_id (`int`, *optional*, defaults to `200091`) : The video token index used as a placeholder for input videos.

out_hidden_size (`int`, *optional*, defaults to 6144) : Output dimension of the vision encoder after patch merging (input width of the multimodal projection).

projector_hidden_size (`int`, *optional*, defaults to 4096) : Intermediate dimension of the multimodal projection.

projector_hidden_act (`str`, *optional*, defaults to `gelu`) : The activation function used by the multimodal projector.

audio_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the audio backbone.

audio_token_id (`int`, *optional*, defaults to `200038`) : The audio token index used as a placeholder for input audio.

This is the configuration class to store the configuration of a MuseSparkModel. It is used to instantiate a Muse Spark
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [meta-models/Muse-Spark-Open-816B-A42B](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import MuseSparkForConditionalGeneration, MuseSparkConfig

>>> # Initializing an MuseSpark style configuration
>>> configuration = MuseSparkConfig()

>>> # Initializing a model from the configuration
>>> model = MuseSparkForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## MuseSparkTextConfig[[transformers.MuseSparkTextConfig]]

#### transformers.MuseSparkTextConfig[[transformers.MuseSparkTextConfig]]

```python
transformers.MuseSparkTextConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vocab_size: int = 202048, hidden_size: int = 8192, intermediate_size: int = 16384, num_hidden_layers: int = 62, num_attention_heads: int = 128, num_key_value_heads: int = 16, head_dim: int = 64, max_position_embeddings: int = 1048576, initializer_range: float = 0.02, rms_norm_eps: float = 1e-05, use_cache: bool = True, pad_token_id: int | None = None, eos_token_id: int | list[int] | None = 200001, bos_token_id: int | None = 200000, tie_word_embeddings: bool = False, rope_parameters: transformers.modeling_rope_utils.RopeParameters | dict | None = None, attention_bias: bool = False, attention_dropout: int | float | None = 0.0, sliding_window: int | None = 2048, layer_types: list[str] | None = None, final_logit_softcapping: float = 20.0, qk_scale_factor: float = 5.4730064863875, output_multiplier: float = 0.1767766952966369, post_norm_eps: float = 1e-08, layer_rope_theta: list[float | int] | None = None, hidden_act: str = 'silu', moe_intermediate_size: int = 4096, num_local_experts: int = 256, num_experts_per_tok: int = 8, routed_scaling_factor: float = 1.0, moe_hidden_size: int = 4096, mlp_layer_types: list[str] | None = None, residual_gate_temperature: float = 0.3, output_router_logits: bool = False)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/muse_spark/configuration_muse_spark.py#L75)

**Parameters:**

vocab_size (`int`, *optional*, defaults to `202048`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

hidden_size (`int`, *optional*, defaults to `8192`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `16384`) : Dimension of the MLP representations.

num_hidden_layers (`int`, *optional*, defaults to `62`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `128`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*, defaults to `16`) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

head_dim (`int`, *optional*, defaults to `64`) : The attention head dimension. If None, it will default to hidden_size // num_attention_heads

max_position_embeddings (`int`, *optional*, defaults to `1048576`) : The maximum sequence length that this model might ever be used with.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rms_norm_eps (`float`, *optional*, defaults to `1e-05`) : The epsilon used by the rms normalization layers.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

pad_token_id (`int`, *optional*) : Token id used for padding in the vocabulary.

eos_token_id (`Union[int, list[int]]`, *optional*, defaults to `200001`) : Token id used for end-of-stream in the vocabulary.

bos_token_id (`int`, *optional*, defaults to `200000`) : Token id used for beginning-of-stream in the vocabulary.

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

rope_parameters (`Union[~modeling_rope_utils.RopeParameters, dict]`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

attention_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

attention_dropout (`Union[int, float]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

sliding_window (`int`, *optional*, defaults to `2048`) : Sliding window attention window size. If `None`, no sliding window is applied.

layer_types (`list[str]`, *optional*) : A list that explicitly maps each layer index with its layer type. If not provided, it will be automatically generated based on config values.

final_logit_softcapping (`float`, *optional*, defaults to `20.0`) : Soft-capping value applied to the final logits before computing the probability distribution. Logits are scaled by `tanh(logit / cap) * cap`.

qk_scale_factor (`float`, *optional*, defaults to 5.4730064863875) : Multiplier applied to Q after the scaleless QK-norm, on top of the standard `1/sqrt(head_dim)` attention scaling.

output_multiplier (`float`, *optional*, defaults to 0.1767766952966369) : Scale applied to logits before the final tanh softcap. Equal to `1/sqrt(hidden_size / 256)` for the released checkpoint.

post_norm_eps (`float`, *optional*, defaults to 1e-8) : Epsilon used for the post-attention and post-FFN norms (which sit between the sub-layer output and the residual).

layer_rope_theta (`list[float]`, *optional*) : Per-layer RoPE base theta; `0` disables rotary (NoPE) for that layer. Overrides the global `rope_parameters["rope_theta"]`. Defaults to the global theta everywhere except every 4th layer counted backward from the last, which is NoPE.

hidden_act (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

moe_intermediate_size (`int`, *optional*, defaults to `4096`) : Intermediate size of the routed expert MLPs.

num_local_experts (`int`, *optional*, defaults to `256`) : Number of local experts on each device. `num_experts` should be divisible by `num_local_experts`.

num_experts_per_tok (`int`, *optional*, defaults to `8`) : Number of experts to route each token to. This is the top-k value for the token-choice routing.

routed_scaling_factor (`float`, *optional*, defaults to `1.0`) : Scaling factor or routed experts.

moe_hidden_size (`int`, *optional*, defaults to 4096) : Width the MoE block pools `hidden_size` down to before the routed experts, and unpools from afterwards. The experts run entirely at this width.

mlp_layer_types (`list[str]`, *optional*) : Per-layer MLP type — `"dense"` or `"sparse"`. Length must equal `num_hidden_layers`. Defaults to all sparse.

residual_gate_temperature (`float`, *optional*, defaults to 0.3) : Temperature applied to the residual gate parameter before the sigmoid.

output_router_logits (`bool`, *optional*, defaults to `False`) : Whether or not the router logits should be returned by the model. Enabling this will also allow the model to output the auxiliary loss, including load balancing loss and router z-loss.

This is the configuration class to store the configuration of a MuseSparkModel. It is used to instantiate a Muse Spark
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [meta-models/Muse-Spark-Open-816B-A42B](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## MuseSparkVisionConfig[[transformers.MuseSparkVisionConfig]]

#### transformers.MuseSparkVisionConfig[[transformers.MuseSparkVisionConfig]]

```python
transformers.MuseSparkVisionConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, patch_size: int = 14, pos_emb_height: int = 32, pos_emb_width: int = 32, num_attention_heads: int = 16, num_hidden_layers: int = 50, hidden_size: int = 1536, intermediate_size: int = 8960, hidden_act: str = 'gelu', rope_parameters: dict | None = None, patch_temporal: int = 2, merge_size: int = 2, max_position_embeddings: int = 1024, layer_norm_eps: float = 1e-05, layer_types: list[str] | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/muse_spark/configuration_muse_spark.py#L31)

**Parameters:**

patch_size (`int`, *optional*, defaults to `14`) : The size (resolution) of each patch.

pos_emb_height (`int`, *optional*) : Initial position embedding height.

pos_emb_width (`int`, *optional*) : Initial position embedding width.

num_attention_heads (`int`, *optional*, defaults to `16`) : Number of attention heads for each attention layer in the Transformer decoder.

num_hidden_layers (`int`, *optional*, defaults to `50`) : Number of hidden layers in the Transformer decoder.

hidden_size (`int`, *optional*, defaults to `1536`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `8960`) : Dimension of the MLP representations.

hidden_act (`str`, *optional*, defaults to `gelu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

rope_parameters (`dict`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

patch_temporal (`int`, *optional*) : The temporal patch size used to embed inputs.

merge_size (`tuple[int] | list[int]`, *optional*) : Kernel size for patch merging.

max_position_embeddings (`int`, *optional*, defaults to `1024`) : The maximum sequence length that this model might ever be used with.

layer_norm_eps (`float`, *optional*, defaults to `1e-05`) : The epsilon used by the layer normalization layers.

layer_types (`list[str]`, *optional*) : A list that explicitly maps each layer index with its layer type. If not provided, it will be automatically generated based on config values.

This is the configuration class to store the configuration of a MuseSparkModel. It is used to instantiate a Muse Spark
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [meta-models/Muse-Spark-Open-816B-A42B](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## MuseSparkAudioConfig[[transformers.MuseSparkAudioConfig]]

#### transformers.MuseSparkAudioConfig[[transformers.MuseSparkAudioConfig]]

```python
transformers.MuseSparkAudioConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, patch_sizes: list[int] | tuple[int, ...] = (16, 6, 5, 4), hidden_sizes: list[int] | tuple[int, ...] = (256, 512, 1024, 2048), depths: list[int] | tuple[int, ...] = (2, 2, 2, 2), num_hidden_layers: int = 4, mlp_ratio: float = 4.0, conv_kernel_size: int = 7, hidden_act: str = 'gelu', rms_norm_eps: float = 1e-06, out_hidden_size: int = 8192, initializer_range: float = 0.02)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/muse_spark/configuration_muse_spark.py#L192)

**Parameters:**

patch_sizes (`list[int] | tuple[int, ...]`, *optional*, defaults to `(16, 6, 5, 4)`) : Temporal downsampling factor applied by each stage's patch embedding. Their product is the number of waveform samples collapsed into one output token.

hidden_sizes (`list[int] | tuple[int, ...]`, *optional*, defaults to `(256, 512, 1024, 2048)`) : Hidden width of each stage.

depths (`list[int] | tuple[int, ...]`, *optional*, defaults to `(2, 2, 2, 2)`) : Number of conv blocks in each layer.

num_hidden_layers (`int`, *optional*, defaults to 4) : Number of `MuseSparkAudioLayer`s, i.e. resolutions in the pyramid. Redundant with `len(hidden_sizes)` and validated against it, so a custom pyramid must set both.

mlp_ratio (`float`, *optional*, defaults to 4.0) : Ratio of a layer's MLP intermediate width to its hidden width.

conv_kernel_size (`int`, *optional*, defaults to 7) : Kernel size of the depthwise causal convolutions.

hidden_act (`str`, *optional*, defaults to `gelu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

rms_norm_eps (`float`, *optional*, defaults to `1e-06`) : The epsilon used by the rms normalization layers.

out_hidden_size (`int`, *optional*, defaults to 8192) : Width the final projection maps to, i.e. the text model's hidden size.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a MuseSparkModel. It is used to instantiate a Muse Spark
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [meta-models/Muse-Spark-Open-816B-A42B](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import MuseSparkAudioConfig, MuseSparkAudioModel

>>> configuration = MuseSparkAudioConfig()
>>> model = MuseSparkAudioModel(configuration)
>>> configuration = model.config
```

## MuseSparkFeatureExtractor[[transformers.MuseSparkFeatureExtractor]]

#### transformers.MuseSparkFeatureExtractor[[transformers.MuseSparkFeatureExtractor]]

```python
transformers.MuseSparkFeatureExtractor(feature_size = 1, sampling_rate = 24000, padding_value = 0.0, samples_per_token = 1920, return_attention_mask = True, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/muse_spark/feature_extraction_muse_spark.py#L28)

**Parameters:**

feature_size (`int`, *optional*, defaults to 1) : The feature dimension of the extracted features. Muse Spark Open takes mono audio.

sampling_rate (`int`, *optional*, defaults to 24000) : The sampling rate, in hertz, the audio is expected at. Audio at any other rate must be resampled before it reaches this class.

padding_value (`float`, *optional*, defaults to 0.0) : Value used to pad, on the right.

samples_per_token (`int`, *optional*, defaults to 1920) : Waveform samples the encoder collapses into one embedding — the product of the audio config's `patch_sizes` (16 * 6 * 5 * 4). Batches are padded up to a multiple of this.

return_attention_mask (`bool`, *optional*, defaults to `True`) : Whether `__call__()` should return `input_values_mask`. The audio tower is fully convolutional and does not consume one; it is off by default.

Constructs a Muse Spark Open feature extractor.

Muse Spark Opens's audio tower consumes the raw waveform — there is no spectrogram front end — so this
only batches, pads and (optionally) truncates. The one non-obvious job it does is pad every batch
up to a multiple of `samples_per_token`: the encoder collapses exactly that many samples into one
embedding and raises otherwise.

This feature extractor inherits from [SequenceFeatureExtractor](/docs/transformers/v5.21.1/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor)
which contains most of the main methods. Users should refer to this superclass for more information
regarding those methods.

## MuseSparkProcessor[[transformers.MuseSparkProcessor]]

#### transformers.MuseSparkProcessor[[transformers.MuseSparkProcessor]]

```python
transformers.MuseSparkProcessor(image_processor = None, video_processor = None, tokenizer = None, feature_extractor = None, chat_template = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/muse_spark/processing_muse_spark.py#L28)

**Parameters:**

image_processor (`MuseGlimmerImageProcessor`) : The image processor is a required input.

video_processor (`MuseGlimmerVideoProcessor`) : The video processor is a required input.

tokenizer (`tokenizer_class`) : The tokenizer is a required input.

feature_extractor (`MuseSparkFeatureExtractor`) : The feature extractor is a required input.

chat_template (`str`) : A Jinja template to convert lists of messages in a chat into a tokenizable string.

Constructs a MuseSparkProcessor which wraps a image processor, a video processor, a tokenizer, and a feature extractor into a single processor.

[MuseSparkProcessor](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkProcessor) offers all the functionalities of [MuseGlimmerImageProcessor](/docs/transformers/v5.21.1/en/model_doc/muse_glimmer#transformers.MuseGlimmerImageProcessor), [MuseGlimmerVideoProcessor](/docs/transformers/v5.21.1/en/model_doc/muse_glimmer#transformers.MuseGlimmerVideoProcessor), `tokenizer_class`, and [MuseSparkFeatureExtractor](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkFeatureExtractor). See the
[~MuseGlimmerImageProcessor](/docs/transformers/v5.21.1/en/model_doc/muse_glimmer#transformers.MuseGlimmerImageProcessor), [~MuseGlimmerVideoProcessor](/docs/transformers/v5.21.1/en/model_doc/muse_glimmer#transformers.MuseGlimmerVideoProcessor), `~tokenizer_class`, and [~MuseSparkFeatureExtractor](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkFeatureExtractor) for more information.

## MuseSparkPreTrainedModel[[transformers.MuseSparkPreTrainedModel]]

#### transformers.MuseSparkPreTrainedModel[[transformers.MuseSparkPreTrainedModel]]

```python
transformers.MuseSparkPreTrainedModel(config: PreTrainedConfig, *inputs, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/muse_spark/modeling_muse_spark.py#L494)

**Parameters:**

config ([PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

This model inherits from [PreTrainedModel](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

## MuseSparkTextModel[[transformers.MuseSparkTextModel]]

#### transformers.MuseSparkTextModel[[transformers.MuseSparkTextModel]]

```python
transformers.MuseSparkTextModel(config: MuseSparkTextConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/muse_spark/modeling_muse_spark.py#L590)

**Parameters:**

config ([MuseSparkTextConfig](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkTextConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Muse Spark Text Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.MuseSparkTextModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/muse_spark/modeling_muse_spark.py#L615)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.21.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.21.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.21.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

**Returns:** `MoeModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `MoeModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MuseSparkConfig](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkConfig)) and inputs.

The [MuseSparkTextModel](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.21.1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

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

  Raw router logits (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
  loss for Mixture of Experts models.

## MuseSparkVisionModel[[transformers.MuseSparkVisionModel]]

#### transformers.MuseSparkVisionModel[[transformers.MuseSparkVisionModel]]

```python
transformers.MuseSparkVisionModel(config: MuseSparkVisionConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/muse_spark/modeling_muse_spark.py#L948)

**Parameters:**

config ([MuseSparkVisionConfig](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkVisionConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Muse Spark Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.MuseSparkVisionModel.forward]]

```python
forward(pixel_values: FloatTensor, grid_thw: LongTensor, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/muse_spark/modeling_muse_spark.py#L979)

**Parameters:**

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input images. Pixel values can be obtained using [MuseGlimmerImageProcessor](/docs/transformers/v5.21.1/en/model_doc/muse_glimmer#transformers.MuseGlimmerImageProcessor). See [MuseGlimmerImageProcessor.__call__()](/docs/transformers/v5.21.1/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([MuseSparkProcessor](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkProcessor) uses [MuseGlimmerImageProcessor](/docs/transformers/v5.21.1/en/model_doc/muse_glimmer#transformers.MuseGlimmerImageProcessor) for processing images).

grid_thw (`torch.LongTensor` of shape `(num_images_or_videos, 3)`) : The temporal, height and width patch-grid dimensions for each packed image or video.

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MuseSparkConfig](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkConfig)) and inputs.

The [MuseSparkVisionModel](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkVisionModel) forward method, overrides the `__call__` special method.

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

## MuseSparkAudioModel[[transformers.MuseSparkAudioModel]]

#### transformers.MuseSparkAudioModel[[transformers.MuseSparkAudioModel]]

```python
transformers.MuseSparkAudioModel(config: MuseSparkAudioConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/muse_spark/modeling_muse_spark.py#L1213)

**Parameters:**

config ([MuseSparkAudioConfig](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkAudioConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Muse Spark Open waveform audio encoder: a fully causal convolutional pyramid that maps raw mono
24 kHz audio to one embedding per 1920 samples (80 ms, i.e. 12.5 Hz), already in the text model's
hidden size. It uses no attention and no spectrogram front-end.

This model inherits from [PreTrainedModel](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.MuseSparkAudioModel.forward]]

```python
forward(input_values: <module 'torch.Tensor' (<doc_builder.mock_imports.MockFinder object at 0x7fef3bad2690>)>, padding_cache: transformers.models.muse_spark.modeling_muse_spark.MuseSparkAudioConv1dPaddingCache | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/muse_spark/modeling_muse_spark.py#L1228)

**Parameters:**

input_values (`doc_builder.mock_imports.torch.Tensor` of shape `(batch_size, sequence_length)`) : Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library (`pip install torchcodec`) or the soundfile library (`pip install soundfile`). To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v5.21.1/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion into a tensor of type `torch.FloatTensor`. See [MuseSparkProcessor.__call__()](/docs/transformers/v5.21.1/en/main_classes/processors#transformers.ProcessorMixin.__call__) for details.

padding_cache (`MuseSparkAudioConv1dPaddingCache`, *optional*) : Cache for padding in convolutional layers to maintain state across streaming chunks.

**Returns:** [BaseModelOutput](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

A [BaseModelOutput](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MuseSparkConfig](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkConfig)) and inputs.

The [MuseSparkAudioModel](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkAudioModel) forward method, overrides the `__call__` special method.

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

## MuseSparkModel[[transformers.MuseSparkModel]]

#### transformers.MuseSparkModel[[transformers.MuseSparkModel]]

```python
transformers.MuseSparkModel(config: MuseSparkConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/muse_spark/modeling_muse_spark.py#L1269)

**Parameters:**

config ([MuseSparkConfig](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Muse Spark Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.MuseSparkModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, pixel_values: typing.Optional[torch.Tensor] = None, image_grid_thw: typing.Optional[torch.LongTensor] = None, pixel_values_videos: typing.Optional[torch.Tensor] = None, video_grid_thw: typing.Optional[torch.LongTensor] = None, input_values: typing.Optional[torch.Tensor] = None, input_values_mask: typing.Optional[torch.Tensor] = None, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/muse_spark/modeling_muse_spark.py#L1362)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.21.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.21.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.21.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [MuseGlimmerImageProcessor](/docs/transformers/v5.21.1/en/model_doc/muse_glimmer#transformers.MuseGlimmerImageProcessor). See [MuseGlimmerImageProcessor.__call__()](/docs/transformers/v5.21.1/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([MuseSparkProcessor](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkProcessor) uses [MuseGlimmerImageProcessor](/docs/transformers/v5.21.1/en/model_doc/muse_glimmer#transformers.MuseGlimmerImageProcessor) for processing images).

image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) : The temporal, height and width of feature shape of each image in LLM.

pixel_values_videos (`torch.Tensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) : The tensors corresponding to the input video. Pixel values for videos can be obtained using [MuseGlimmerVideoProcessor](/docs/transformers/v5.21.1/en/model_doc/muse_glimmer#transformers.MuseGlimmerVideoProcessor). See [MuseGlimmerVideoProcessor.__call__()](/docs/transformers/v5.21.1/en/model_doc/molmo2#transformers.Molmo2VideoProcessor.__call__) for details ([MuseSparkProcessor](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkProcessor) uses [MuseGlimmerVideoProcessor](/docs/transformers/v5.21.1/en/model_doc/muse_glimmer#transformers.MuseGlimmerVideoProcessor) for processing videos).

video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) : The temporal, height and width of feature shape of each video in LLM.

input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library (`pip install torchcodec`) or the soundfile library (`pip install soundfile`). To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v5.21.1/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion into a tensor of type `torch.FloatTensor`. See [MuseSparkProcessor.__call__()](/docs/transformers/v5.21.1/en/main_classes/processors#transformers.ProcessorMixin.__call__) for details.

input_values_mask (`torch.Tensor` of shape `(batch_size, num_samples)`, *optional*) : Which entries of `input_values` are real audio rather than padding, as produced by [MuseSparkFeatureExtractor](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkFeatureExtractor). Required when batching clips of different lengths.

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** `MuseSparkModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `MuseSparkModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MuseSparkConfig](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkConfig)) and inputs.

The [MuseSparkModel](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.21.1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

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
- **image_hidden_states** (`torch.FloatTensor`, *optional*) -- A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
- **router_logits** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

  Router logits of the model, useful to compute the auxiliary loss for Mixture of Experts models.

Example:

```python
>>> import os

>>> from transformers import AutoProcessor, DistributedConfig, MuseSparkModel

>>> distributed_config = DistributedConfig(
...     tp_size=int(os.environ["WORLD_SIZE"]),
...     ep_size=int(os.environ["WORLD_SIZE"]),
... )

>>> processor = AutoProcessor.from_pretrained("meta-models/Muse-Spark-Open-816B-A42B")
>>> model = MuseSparkModel.from_pretrained(
...     "meta-models/Muse-Spark-Open-816B-A42B",
...     dtype="auto",
...     distributed_config=distributed_config,
... )

>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
...             {"type": "text", "text": "What is shown in this image?"},
...         ],
...     },
... ]

>>> inputs = processor.apply_chat_template(
...     messages,
...     reasoning_effort="medium",
...     add_generation_prompt=True,
...     tokenize=True,
...     return_dict=True,
...     return_tensors="pt",
... ).to(model.device).to(model.dtype)

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
```

## MuseSparkForConditionalGeneration[[transformers.MuseSparkForConditionalGeneration]]

#### transformers.MuseSparkForConditionalGeneration[[transformers.MuseSparkForConditionalGeneration]]

```python
transformers.MuseSparkForConditionalGeneration(config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/muse_spark/modeling_muse_spark.py#L1552)

**Parameters:**

config ([MuseSparkForConditionalGeneration](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkForConditionalGeneration)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Muse Spark Model for token generation conditioned on other modalities (e.g. image-text-to-text generation).

This model inherits from [PreTrainedModel](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.MuseSparkForConditionalGeneration.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, pixel_values: typing.Optional[torch.FloatTensor] = None, image_grid_thw: typing.Optional[torch.LongTensor] = None, pixel_values_videos: typing.Optional[torch.FloatTensor] = None, video_grid_thw: typing.Optional[torch.LongTensor] = None, input_values: typing.Optional[torch.Tensor] = None, input_values_mask: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/muse_spark/modeling_muse_spark.py#L1590)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.21.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [MuseGlimmerImageProcessor](/docs/transformers/v5.21.1/en/model_doc/muse_glimmer#transformers.MuseGlimmerImageProcessor). See [MuseGlimmerImageProcessor.__call__()](/docs/transformers/v5.21.1/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([MuseSparkProcessor](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkProcessor) uses [MuseGlimmerImageProcessor](/docs/transformers/v5.21.1/en/model_doc/muse_glimmer#transformers.MuseGlimmerImageProcessor) for processing images).

image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) : The temporal, height and width of feature shape of each image in LLM.

pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) : The tensors corresponding to the input video. Pixel values for videos can be obtained using [MuseGlimmerVideoProcessor](/docs/transformers/v5.21.1/en/model_doc/muse_glimmer#transformers.MuseGlimmerVideoProcessor). See [MuseGlimmerVideoProcessor.__call__()](/docs/transformers/v5.21.1/en/model_doc/molmo2#transformers.Molmo2VideoProcessor.__call__) for details ([MuseSparkProcessor](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkProcessor) uses [MuseGlimmerVideoProcessor](/docs/transformers/v5.21.1/en/model_doc/muse_glimmer#transformers.MuseGlimmerVideoProcessor) for processing videos).

video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) : The temporal, height and width of feature shape of each video in LLM.

input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library (`pip install torchcodec`) or the soundfile library (`pip install soundfile`). To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v5.21.1/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion into a tensor of type `torch.FloatTensor`. See [MuseSparkProcessor.__call__()](/docs/transformers/v5.21.1/en/main_classes/processors#transformers.ProcessorMixin.__call__) for details.

input_values_mask (`torch.Tensor` of shape `(batch_size, num_samples)`, *optional*) : Which entries of `input_values` are real audio rather than padding, as produced by [MuseSparkFeatureExtractor](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkFeatureExtractor). Required when batching clips of different lengths.

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.21.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.21.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** `MuseSparkCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `MuseSparkCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MuseSparkConfig](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkConfig)) and inputs.

The [MuseSparkForConditionalGeneration](/docs/transformers/v5.21.1/en/model_doc/muse_spark#transformers.MuseSparkForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.text_config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.21.1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

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
  image_hidden_states of the model produced by the vision encoder after projecting last hidden state.
- **router_logits** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

  Router logits of the model, useful to compute the auxiliary loss for Mixture of Experts models.

Example:

```python
>>> import os

>>> from transformers import AutoProcessor, DistributedConfig, MuseSparkForConditionalGeneration

>>> distributed_config = DistributedConfig(
...     tp_size=int(os.environ["WORLD_SIZE"]),
...     ep_size=int(os.environ["WORLD_SIZE"]),
... )

>>> processor = AutoProcessor.from_pretrained("meta-models/Muse-Spark-Open-816B-A42B")
>>> model = MuseSparkForConditionalGeneration.from_pretrained(
...     "meta-models/Muse-Spark-Open-816B-A42B",
...     dtype="auto",
...     distributed_config=distributed_config,
... )

>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
...             {"type": "text", "text": "What is shown in this image?"},
...         ],
...     },
... ]

>>> inputs = processor.apply_chat_template(
...     messages,
...     reasoning_effort="medium",
...     add_generation_prompt=True,
...     tokenize=True,
...     return_dict=True,
...     return_tensors="pt",
... ).to(model.device).to(model.dtype)
>>> input_len = inputs["input_ids"].shape[-1]

>>> outputs = model.generate(**inputs, max_new_tokens=512)

>>> response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
>>> print(response)
The image shows a **Pallas's cat — also known as a manul, *Otocolobus manul*** — walking through snow.
...
```

### ColModernVBert
https://huggingface.co/docs/transformers/v5.21.1/model_doc/colmodernvbert.md
