# UnlimitedOcr

## Overview

The UnlimitedOcr model was proposed in [Unlimited OCR Works](https://huggingface.co/papers/2606.23050) by Youyang Yin, Huanhuan Liu, Qunyi Xie, Chaorun Liu, Shiqi Yang, Shaohua Wang, Zhanlong Liu, Hao Zou, Jinyue Chen, Shu Wei, Jingjing Wu, Mingxin Huang, Zhen Wu, Guibin Wang, Tengyu Du, and Lei Jia from Baidu Inc.

It is a single 3B parameter model with a standard context length of 32,768 tokens.

The abstract from the paper is the following:

*Recently, end-to-end OCR models, exemplified by DeepSeek OCR, have once again thrust OCR into the spotlight. A widely held view is that employing a large language model (LLM) as the decoder allows the model to leverage the prior distribution of language, leading to improved OCR performance. However, the downside is equally evident: as the output sequence lengthens, the accumulated KV cache drives up memory consumption and progressively slows down generation. This stands in stark contrast to humans, who exhibit no such decline in efficiency during long-horizon copying tasks. In this technical report, we propose Unlimited OCR, a model designed to emulate human parsing working memory. Taking DeepSeek OCR as the baseline, we replace all attention layers in the decoder with our proposed Reference Sliding Window Attention (R-SWA), which reduces attention computation costs while maintaining a constant KV cache throughout the entire decoding process. By combining the high compression rate of DeepSeek OCR's encoder with our constant KV cache design, Unlimited OCR can transcribe dozens of pages of documents in a single forward pass under a standard maximum length of 32K. More importantly, R-SWA is a general-purpose parsing attention mechanism — beyond OCR, it is equally applicable to tasks such as ASR, translation, etc. Codes and model weights are publicly available at http://github.com/baidu/Unlimited-OCR*

Unlimited-OCR supports two inference configurations: the default "gundam" mode uses 640x640 tiles with dynamic cropping for high-resolution documents, and "base" mode uses a single 1024x1024 global view for standard-resolution inputs. To enable base mode set `crop_to_patches=False` in the processor, or pass it via `processor_kwargs` when using [apply_chat_template()](/docs/transformers/v5.20.2/en/main_classes/processors#transformers.ProcessorMixin.apply_chat_template).

The vision tower follows the two-stage approach from [DeepSeek-OCR-2](./deepseek_ocr2): a SAM ViT-B encoder feeds into a CLIP ViT encoder. Unlike DeepSeek-OCR-2, the CLIP features are additionally concatenated with the SAM features to yield the final image tokens. Unlimited-OCR also omits the learnable patch queries from DeepSeek-OCR-2.

The text model is identical to DeepSeek-OCR-2 with the additional Reference Sliding Window Attention (R-SWA). R-SWA applies only to generated tokens. All image and prompt tokens remain fully visible throughout decoding, so long documents do not lose context from earlier pages.

This model was contributed by [guarin](https://huggingface.co/guarin).
The original code can be found [here](https://github.com/baidu/Unlimited-OCR).

> [!TIP]
> The original implementation runs the model with [torch.autocast](https://pytorch.org/docs/stable/amp.html#torch.autocast) in bfloat16. Wrap the forward and generate calls in an autocast context to reproduce its outputs.
>
> ```python
> with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16):
>     output = model.generate(**inputs)
> ```

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("baidu/Unlimited-OCR", device_map="auto")
processor = AutoProcessor.from_pretrained("baidu/Unlimited-OCR")

image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_suggestion_form.jpg"
messages = [
    {
        "role": "user",
        "content": [{"type": "image", "url": image}, {"type": "text", "text": "document parsing."}],
    }
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device, model.dtype)

output = model.generate(**inputs, max_new_tokens=512)
processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
# image [383, 87, 497, 171]\ntext [333, 201, 558, 230]R&D QUALITY IMPROVEMENT\nSUGGESTION/SOLUTION FORM...

# All bounding boxes are in (x1, y1, x2, y2) format with coordinates normalized to [0, 999]
```

### Batch processing

For batch processing, pass a list of messages. Set `padding=True` for the processor if images have different sizes.

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("baidu/Unlimited-OCR", device_map="auto")
processor = AutoProcessor.from_pretrained("baidu/Unlimited-OCR")

image1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_suggestion_form.jpg"
image2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_receipt.jpeg"
messages = [
    [
        {
            "role": "user",
            "content": [{"type": "image", "url": image}, {"type": "text", "text": "document parsing."}],
        }
    ]
    for image in [image1, image2]
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    processor_kwargs={"padding": True},
).to(model.device, model.dtype)

output = model.generate(**inputs)
processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
# image [383, 87, 497, 171]\ntext [333, 201, 558, 230]R&D QUALITY IMPROVEMENT\nSUGGESTION/SOLUTION FORM...

# All bounding boxes are in (x1, y1, x2, y2) format with coordinates normalized to [0, 999]
```

### Region detections

Set `skip_special_tokens=False` to wrap all detections and region types in `<|det|>...<|/det|>` markers. This is useful for further post-processing of the output, for example to plot the detected bounding boxes on the image. Each detection is wrapped as `<|det|>region_type [x1, y1, x2, y2]<|/det|>text...` with coordinates normalized to a `[0, 999]` range. Set `return_detections=True` to get an additional list of dictionaries with all detections parsed as `{"region_type": region_type, "box": [x1, y1, x2, y2], "text": "..."}`.

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("baidu/Unlimited-OCR", device_map="auto")
processor = AutoProcessor.from_pretrained("baidu/Unlimited-OCR")

image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_suggestion_form.jpg"
messages = [
    {
        "role": "user",
        "content": [{"type": "image", "url": image}, {"type": "text", "text": "document parsing."}],
    }
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device, model.dtype)

output = model.generate(**inputs)
decoded, detections = processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False, return_detections=True)
# <|det|>image [383, 87, 497, 171]<|/det|>\n<|det|>text [333, 201, 558, 230]<|/det|>R&D QUALITY IMPROVEMENT\nSUGGESTION/SOLUTION FORM...

# Visualization
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers.image_utils import load_image

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_suggestion_form.jpg")
width, height = image.size

figure, axis = plt.subplots(figsize=(10, 12))
axis.imshow(image)
for detection in detections:
    region_type = detection["region_type"]
    x1, y1, x2, y2 = detection["box"]
    x1, y1, x2, y2 = int(x1) / 999 * width, int(y1) / 999 * height, int(x2) / 999 * width, int(y2) / 999 * height
    color = (random.random(), random.random(), random.random())
    rectangle = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor=color, facecolor="none")
    axis.add_patch(rectangle)
    axis.text(x1, y1, region_type, color="white", fontsize=8, backgroundcolor=color, verticalalignment="top")
axis.axis("off")
plt.show()
```

Multi-page documents can be parsed jointly in a single forward pass by passing all page images together. Add one image block per page to the message so the model processes all pages as a continuous document.

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("baidu/Unlimited-OCR", device_map="auto")
processor = AutoProcessor.from_pretrained("baidu/Unlimited-OCR")

page1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_suggestion_form.jpg"
page2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_receipt.jpeg"

messages = [
    {
        "role": "user",
        "content": [{"type": "image", "url": page} for page in [page1, page2]]
        + [{"type": "text", "text": "Multi page parsing."}],
    }
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    processor_kwargs={"crop_to_patches": False},
).to(model.device, model.dtype)

output = model.generate(
    **inputs,
    no_repeat_ngram_window_size=1024, # default: 128, larger window size recommended for long documents
)
processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
# <PAGE>image [382, 87, 489, 174]\ntitle [333, 201, 556, 230]R&D QUALITY IMPROVEMENT\nSUGGESTION/SCLUTION FORM...

# All bounding boxes are in (x1, y1, x2, y2) format with coordinates normalized to [0, 999]
```

## UnlimitedOcrConfig[[transformers.UnlimitedOcrConfig]]

#### transformers.UnlimitedOcrConfig[[transformers.UnlimitedOcrConfig]]

```python
transformers.UnlimitedOcrConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vision_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, text_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, image_token_id: int = 128815, tie_word_embeddings: bool = False)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/unlimited_ocr/configuration_unlimited_ocr.py#L262)

**Parameters:**

vision_config (`dict` or `UnlimitedOcrVisionConfig`, *optional*) : Configuration for the vision encoders. Defaults to `UnlimitedOcrVisionConfig()`.

text_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the text backbone.

image_token_id (`int`, *optional*, defaults to `128815`) : The image token index used as a placeholder for input images.

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

This is the configuration class to store the configuration of a UnlimitedOcrModel. It is used to instantiate a Unlimited Ocr
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.20.2/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.20.2/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import UnlimitedOcrConfig, UnlimitedOcrModel

>>> configuration = UnlimitedOcrConfig()
>>> model = UnlimitedOcrModel(configuration)
>>> configuration = model.config
```

## UnlimitedOcrTextConfig[[transformers.UnlimitedOcrTextConfig]]

#### transformers.UnlimitedOcrTextConfig[[transformers.UnlimitedOcrTextConfig]]

```python
transformers.UnlimitedOcrTextConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vocab_size: int = 129280, hidden_size: int = 1280, intermediate_size: int = 6848, num_hidden_layers: int = 12, num_attention_heads: int = 10, num_key_value_heads: int | None = 10, hidden_act: str = 'silu', max_position_embeddings: int = 32768, initializer_range: float = 0.02, rms_norm_eps: float = 1e-06, use_cache: bool = True, pad_token_id: int | None = None, bos_token_id: int | None = 0, eos_token_id: int | list[int] | None = 1, pretraining_tp: int | None = 1, tie_word_embeddings: bool = False, rope_parameters: transformers.modeling_rope_utils.RopeParameters | dict | None = None, attention_bias: bool = False, attention_dropout: float | None = 0.0, mlp_bias: bool = False, head_dim: int | None = None, n_group: int | None = 1, n_routed_experts: int = 64, output_router_logits: bool = False, n_shared_experts: int = 2, routed_scaling_factor: float = 1.0, topk_group: int | None = 1, topk_method: str | None = 'greedy', num_experts_per_tok: int | None = 6, moe_intermediate_size: int = 896, mlp_layer_types: list[str] | None = None, layer_types: list[str] | None = None, sliding_window: int | None = 128)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/unlimited_ocr/configuration_unlimited_ocr.py#L141)

**Parameters:**

vocab_size (`int`, *optional*, defaults to `129280`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

hidden_size (`int`, *optional*, defaults to `1280`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `6848`) : Dimension of the MLP representations.

num_hidden_layers (`int`, *optional*, defaults to `12`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `10`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*, defaults to `10`) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

hidden_act (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

max_position_embeddings (`int`, *optional*, defaults to `32768`) : The maximum sequence length that this model might ever be used with.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rms_norm_eps (`float`, *optional*, defaults to `1e-06`) : The epsilon used by the rms normalization layers.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

pad_token_id (`int`, *optional*) : Token id used for padding in the vocabulary.

bos_token_id (`int`, *optional*, defaults to `0`) : Token id used for beginning-of-stream in the vocabulary.

eos_token_id (`Union[int, list[int]]`, *optional*, defaults to `1`) : Token id used for end-of-stream in the vocabulary.

pretraining_tp (`int`, *optional*, defaults to `1`) : Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this document](https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism) to understand more about it. This value is necessary to ensure exact reproducibility of the pretraining results. Please refer to [this issue](https://github.com/pytorch/pytorch/issues/76232).

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

rope_parameters (`Union[~modeling_rope_utils.RopeParameters, dict]`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

attention_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

attention_dropout (`float`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

mlp_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.

head_dim (`int`, *optional*) : The attention head dimension. If None, it will default to hidden_size // num_attention_heads

n_group (`int`, *optional*) : Number of groups for grouped top-k expert routing.

n_routed_experts (`int`, *optional*, defaults to `64`) : Number of routed experts.

output_router_logits (`bool`, *optional*, defaults to `False`) : Whether or not the router logits should be returned by the model. Enabling this will also allow the model to output the auxiliary loss, including load balancing loss and router z-loss.

n_shared_experts (`int`, *optional*, defaults to `2`) : Number of shared experts.

routed_scaling_factor (`float`, *optional*, defaults to `1.0`) : Scaling factor or routed experts.

topk_group (`int`, *optional*, defaults to `1`) : Number of selected groups for each token (for each token, ensuring the selected experts is only within `topk_group` groups).

topk_method (`str`, *optional*, defaults to `"greedy"`) : Method for selecting top-k experts in MoE layers.

num_experts_per_tok (`int`, *optional*, defaults to `6`) : Number of experts to route each token to. This is the top-k value for the token-choice routing.

moe_intermediate_size (`int`, *optional*, defaults to `896`) : Intermediate size of the routed expert MLPs.

mlp_layer_types (`list[str]`, *optional*) : MLP type (`"dense"` or `"sparse"`) for each decoder layer, e.g. `["dense", "sparse", "sparse", ...]`.

layer_types (`list[str]`, *optional*) : Attention type for each decoder layer. Defaults to `"reference_sliding_attention"` on every layer.

sliding_window (`int`, *optional*, defaults to `128`) : Sliding window size for reference sliding window attention. If set, every token attends to the last `sliding_window` and all image and prompt tokens.

This is the configuration class to store the configuration of a UnlimitedOcrModel. It is used to instantiate a Unlimited Ocr
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.20.2/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.20.2/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import UnlimitedOcrTextConfig, UnlimitedOcrTextModel

>>> configuration = UnlimitedOcrTextConfig()
>>> model = UnlimitedOcrTextModel(configuration)
>>> configuration = model.config
```

## UnlimitedOcrVisionConfig[[transformers.UnlimitedOcrVisionConfig]]

#### transformers.UnlimitedOcrVisionConfig[[transformers.UnlimitedOcrVisionConfig]]

```python
transformers.UnlimitedOcrVisionConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, sam_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, encoder_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/unlimited_ocr/configuration_unlimited_ocr.py#L112)

**Parameters:**

sam_config (`dict` or `UnlimitedOcrSamVisionConfig`, *optional*) : Configuration for the SAM vision encoder. Defaults to `UnlimitedOcrSamVisionConfig()`.

encoder_config (`dict` or `UnlimitedOcrVisionEncoderConfig`, *optional*) : Configuration for the vision encoder. Defaults to `UnlimitedOcrVisionEncoderConfig()`.

This is the configuration class to store the configuration of a UnlimitedOcrModel. It is used to instantiate a Unlimited Ocr
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.20.2/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.20.2/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import UnlimitedOcrVisionConfig, UnlimitedOcrVisionModel

>>> configuration = UnlimitedOcrVisionConfig()
>>> model = UnlimitedOcrVisionModel(configuration)
>>> configuration = model.config
```

## UnlimitedOcrVisionEncoderConfig[[transformers.UnlimitedOcrVisionEncoderConfig]]

#### transformers.UnlimitedOcrVisionEncoderConfig[[transformers.UnlimitedOcrVisionEncoderConfig]]

```python
transformers.UnlimitedOcrVisionEncoderConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, hidden_size: int = 1024, intermediate_size: int = 4096, num_hidden_layers: int = 24, num_attention_heads: int = 16, num_channels: int = 3, image_size: int | list[int] | tuple[int, int] | None = 224, patch_size: int | list[int] | tuple[int, int] | None = 14, hidden_act: str = 'quick_gelu', layer_norm_eps: float = 1e-05, attention_dropout: int | float | None = 0.0, initializer_range: float = 0.02, initializer_factor: float = 1.0)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/unlimited_ocr/configuration_unlimited_ocr.py#L74)

**Parameters:**

hidden_size (`int`, *optional*, defaults to `1024`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `4096`) : Dimension of the MLP representations.

num_hidden_layers (`int`, *optional*, defaults to `24`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `16`) : Number of attention heads for each attention layer in the Transformer decoder.

num_channels (`int`, *optional*, defaults to `3`) : The number of input channels.

image_size (`Union[int, list[int], tuple[int, int]]`, *optional*, defaults to `224`) : The size (resolution) of each image.

patch_size (`Union[int, list[int], tuple[int, int]]`, *optional*, defaults to `14`) : The size (resolution) of each patch.

hidden_act (`str`, *optional*, defaults to `quick_gelu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

layer_norm_eps (`float`, *optional*, defaults to `1e-05`) : The epsilon used by the layer normalization layers.

attention_dropout (`Union[int, float]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

initializer_factor (`float`, *optional*, defaults to `1.0`) : A factor for initializing all weight matrices (should be kept to 1, used internally for initialization testing).

This is the configuration class to store the configuration of a UnlimitedOcrModel. It is used to instantiate a Unlimited Ocr
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.20.2/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.20.2/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import UnlimitedOcrVisionEncoderConfig, UnlimitedOcrVisionEncoder

>>> configuration = UnlimitedOcrVisionEncoderConfig()
>>> model = UnlimitedOcrVisionEncoder(configuration)
>>> configuration = model.config
```

## UnlimitedOcrSamVisionConfig[[transformers.UnlimitedOcrSamVisionConfig]]

#### transformers.UnlimitedOcrSamVisionConfig[[transformers.UnlimitedOcrSamVisionConfig]]

```python
transformers.UnlimitedOcrSamVisionConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, hidden_size: int = 768, output_channels: int = 256, num_hidden_layers: int = 12, num_attention_heads: int = 12, num_channels: int = 3, image_size: int | list[int] | tuple[int, int] = 1024, patch_size: int | list[int] | tuple[int, int] = 16, hidden_act: str = 'gelu', layer_norm_eps: float = 1e-06, attention_dropout: float | int = 0.0, initializer_range: float = 1e-10, qkv_bias: bool = True, use_abs_pos: bool = True, use_rel_pos: bool = True, window_size: int = 14, global_attn_indexes: list[int] | tuple[int, ...] = (2, 5, 8, 11), mlp_dim: int = 3072, downsample_channels: list[int] | tuple[int, ...] = (512, 1024))
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/unlimited_ocr/configuration_unlimited_ocr.py#L29)

**Parameters:**

hidden_size (`int`, *optional*, defaults to `768`) : Dimension of the hidden representations.

output_channels (`int`, *optional*, defaults to 256) : Dimensionality of the output channels in the Patch Encoder.

num_hidden_layers (`int`, *optional*, defaults to `12`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `12`) : Number of attention heads for each attention layer in the Transformer decoder.

num_channels (`int`, *optional*, defaults to `3`) : The number of input channels.

image_size (`Union[int, list[int], tuple[int, int]]`, *optional*, defaults to `1024`) : The size (resolution) of each image.

patch_size (`Union[int, list[int], tuple[int, int]]`, *optional*, defaults to `16`) : The size (resolution) of each patch.

hidden_act (`str`, *optional*, defaults to `gelu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

layer_norm_eps (`float`, *optional*, defaults to `1e-06`) : The epsilon used by the layer normalization layers.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

initializer_range (`float`, *optional*, defaults to `1e-10`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

qkv_bias (`bool`, *optional*, defaults to `True`) : Whether to add a bias to the queries, keys and values.

use_abs_pos (`bool`, *optional*, defaults to `True`) : Whether to use absolute position embedding.

use_rel_pos (`bool`, *optional*, defaults to `True`) : Whether to use relative position embedding.

window_size (`int`, *optional*, defaults to 14) : Window size for relative position.

global_attn_indexes (`list[int]`, *optional*, defaults to `[2, 5, 8, 11]`) : The indexes of the global attention layers.

mlp_dim (`int`, *optional*, defaults to 3072) : The dimensionality of the MLP layer in the Transformer encoder.

downsample_channels (`list[int]`, *optional*, defaults to `(512, 1024)`) : The channel dimensions for the multi-scale downsampling neck layers.

This is the configuration class to store the configuration of a UnlimitedOcrModel. It is used to instantiate a Unlimited Ocr
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.20.2/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.20.2/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## UnlimitedOcrImageProcessor[[transformers.UnlimitedOcrImageProcessor]]

#### transformers.UnlimitedOcrImageProcessor[[transformers.UnlimitedOcrImageProcessor]]

```python
transformers.UnlimitedOcrImageProcessor(**kwargs: *<class 'transformers.models.unlimited_ocr.image_processing_unlimited_ocr.UnlimitedOcrImageProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/unlimited_ocr/image_processing_unlimited_ocr.py#L141)

**Parameters:**

do_convert_rgb (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to convert the image to RGB.

do_resize (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to resize the image.

size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*, defaults to `{'height' : 1024, 'width': 1024}`): Describes the maximum input dimensions to the model.

default_to_square (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to default to a square image when resizing, if size is an int.

crop_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : Size of the output image after applying `center_crop`.

resample (`Annotated[Union[int, PILImageResampling, NoneType], None]`, *kwargs*, defaults to `3`) : Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.

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

crop_to_patches (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to crop the image to patches. Can be overridden by the `crop_to_patches` parameter in the `preprocess` method.

min_patches (`int`, *kwargs*, *optional*, defaults to `2`) : The minimum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is set to `True`. Can be overridden by the `min_patches` parameter in the `preprocess` method.

max_patches (`int`, *kwargs*, *optional*, defaults to `32`) : The maximum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is set to `True`. Can be overridden by the `max_patches` parameter in the `preprocess` method.

tile_size (`int`, *kwargs*, *optional*, defaults to `640`) : The size of each local tile. Must match the model's query embedding size.

background_color (`list[int]`, *kwargs*, *optional*, defaults to `[127, 127, 127]`) : The background color for padding.

maximum_pad_value (`int`, *kwargs*, *optional*, defaults to `640`) : If `crop_to_patches` is `False` and `max(size.height, size.width)` is smaller than or equal to this value, the image is resized directly to a square of `max(size.height, size.width)` without preserving the aspect ratio. Otherwise, the image is resized while preserving the aspect ratio and then padded to a square with `background_color`.

Constructs a UnlimitedOcrImageProcessor image processor.

#### crop_image_to_patches[[transformers.UnlimitedOcrImageProcessor.crop_image_to_patches]]

```python
crop_image_to_patches(images: torch.Tensor, min_patches: int, max_patches: int, tile_size: int, resample: PIL.Image.Resampling | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/unlimited_ocr/image_processing_unlimited_ocr.py#L163)

**Parameters:**

images (`torch.Tensor`) : The images to crop, shape `(batch, channels, height, width)`.

min_patches (`int`) : Minimum number of patches.

max_patches (`int`) : Maximum number of patches.

tile_size (`int`) : The size of each tile.

resample (`PILImageResampling`, *optional*) : Resampling filter for resizing.

**Returns:** `tuple[torch.Tensor, int]`

Stacked patches `(batch, num_patches, channels, tile_size, tile_size)`
and number of patches per image.

Crop batched images to patches based on optimal tiling.

#### get_number_of_image_patches[[transformers.UnlimitedOcrImageProcessor.get_number_of_image_patches]]

```python
get_number_of_image_patches(height: int, width: int, images_kwargs: dict | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/unlimited_ocr/image_processing_unlimited_ocr.py#L309)

Returns the number of image patches for a given image size (1 global + local patches).

#### pad_to_square[[transformers.UnlimitedOcrImageProcessor.pad_to_square]]

```python
pad_to_square(images: torch.Tensor, background_color: int | tuple[int, int, int] = 0)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/unlimited_ocr/image_processing_unlimited_ocr.py#L328)

**Parameters:**

images (`torch.Tensor`) : The images to pad. Shape: (batch_size, num_channels, height, width) or (num_channels, height, width).

background_color (`int` or `tuple[int, int, int]`, *optional*, defaults to 0) : The color to use for the padding. Can be an integer for single channel or a tuple of integers representing for multi-channel images. If passed as integer in multi-channel mode, it will default to `0` in subsequent channels.

**Returns:** `torch.Tensor`

The padded images.

Pads an image to a square based on the longest edge.

## UnlimitedOcrProcessor[[transformers.UnlimitedOcrProcessor]]

#### transformers.UnlimitedOcrProcessor[[transformers.UnlimitedOcrProcessor]]

```python
transformers.UnlimitedOcrProcessor(image_processor = None, tokenizer = None, chat_template = None, patch_size = 16, downsample_ratio = 4, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/unlimited_ocr/processing_unlimited_ocr.py#L30)

**Parameters:**

image_processor (`UnlimitedOcrImageProcessor`) : The image processor is a required input.

tokenizer (`TokenizersBackend`) : The tokenizer is a required input.

chat_template (`str`) : A Jinja template to convert lists of messages in a chat into a tokenizable string.

patch_size (`int`, *optional*, defaults to `16`) : The patch size used by the vision encoder (SAM ViT patch embedding size).

downsample_ratio (`int`, *optional*, defaults to `4`) : The downsampling ratio applied after the vision encoder.

Constructs a UnlimitedOcrProcessor which wraps a image processor and a tokenizer into a single processor.

[UnlimitedOcrProcessor](/docs/transformers/v5.20.2/en/model_doc/unlimited_ocr#transformers.UnlimitedOcrProcessor) offers all the functionalities of [UnlimitedOcrImageProcessor](/docs/transformers/v5.20.2/en/model_doc/unlimited_ocr#transformers.UnlimitedOcrImageProcessor) and [TokenizersBackend](/docs/transformers/v5.20.2/en/main_classes/tokenizer#transformers.TokenizersBackend). See the
[~UnlimitedOcrImageProcessor](/docs/transformers/v5.20.2/en/model_doc/unlimited_ocr#transformers.UnlimitedOcrImageProcessor) and [~TokenizersBackend](/docs/transformers/v5.20.2/en/main_classes/tokenizer#transformers.TokenizersBackend) for more information.

#### batch_decode[[transformers.UnlimitedOcrProcessor.batch_decode]]

```python
batch_decode(*args, return_detections: bool = False, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/unlimited_ocr/processing_unlimited_ocr.py#L92)

**Parameters:**

return_detections (`bool`, *optional*, defaults to `False`) : Whether or not to also return the layout detections parsed from the decoded text.

**Returns:** `list[str]` or `tuple[list[str], list[list[dict]]]`

The decoded text. If `return_detections` is `True`,
a tuple of the decoded text and the detections of every sequence, where every detection is a dictionary
with the keys `region_type`, `box` and `text`. Boxes are in [x1, y1, x2, y2] format with coordinates
normalized to [0, 999].

This method forwards all its arguments to PreTrainedTokenizer's [batch_decode()](/docs/transformers/v5.20.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode). Please refer to
the docstring of this method for more information.

#### decode[[transformers.UnlimitedOcrProcessor.decode]]

```python
decode(*args, return_detections: bool = False, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/unlimited_ocr/processing_unlimited_ocr.py#L113)

**Parameters:**

return_detections (`bool`, *optional*, defaults to `False`) : Whether or not to also return the layout detections parsed from the decoded text.

**Returns:** `str` or `tuple[str, list[dict]]`

The decoded text. If `return_detections` is `True`, a tuple of the
decoded text and a list of detections, where every detection is a dictionary with the keys `region_type`,
`box` and `text`. Boxes are in [x1, y1, x2, y2] format with coordinates normalized to [0, 999].

This method forwards all its arguments to PreTrainedTokenizer's [decode()](/docs/transformers/v5.20.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.decode). Please refer to
the docstring of this method for more information.

## UnlimitedOcrPreTrainedModel[[transformers.UnlimitedOcrPreTrainedModel]]

#### transformers.UnlimitedOcrPreTrainedModel[[transformers.UnlimitedOcrPreTrainedModel]]

```python
transformers.UnlimitedOcrPreTrainedModel(config: PreTrainedConfig, *inputs, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/unlimited_ocr/modeling_unlimited_ocr.py#L281)

**Parameters:**

config ([PreTrainedConfig](/docs/transformers/v5.20.2/en/main_classes/configuration#transformers.PreTrainedConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.20.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

This model inherits from [PreTrainedModel](/docs/transformers/v5.20.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

## UnlimitedOcrTextPreTrainedModel[[transformers.UnlimitedOcrTextPreTrainedModel]]

#### transformers.UnlimitedOcrTextPreTrainedModel[[transformers.UnlimitedOcrTextPreTrainedModel]]

```python
transformers.UnlimitedOcrTextPreTrainedModel(config: PreTrainedConfig, *inputs, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/unlimited_ocr/modeling_unlimited_ocr.py#L1266)

**Parameters:**

config ([PreTrainedConfig](/docs/transformers/v5.20.2/en/main_classes/configuration#transformers.PreTrainedConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.20.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

This model inherits from [PreTrainedModel](/docs/transformers/v5.20.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

## UnlimitedOcrTextModel[[transformers.UnlimitedOcrTextModel]]

#### transformers.UnlimitedOcrTextModel[[transformers.UnlimitedOcrTextModel]]

```python
transformers.UnlimitedOcrTextModel(config: UnlimitedOcrTextConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/unlimited_ocr/modeling_unlimited_ocr.py#L1728)

**Parameters:**

config ([UnlimitedOcrTextConfig](/docs/transformers/v5.20.2/en/model_doc/unlimited_ocr#transformers.UnlimitedOcrTextConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.20.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Unlimited Ocr Text Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.20.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.UnlimitedOcrTextModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/unlimited_ocr/modeling_unlimited_ocr.py#L1745)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.20.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.20.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.20.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.20.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.20.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

**Returns:** `MoeModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `MoeModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([UnlimitedOcrConfig](/docs/transformers/v5.20.2/en/model_doc/unlimited_ocr#transformers.UnlimitedOcrConfig)) and inputs.

The [UnlimitedOcrTextModel](/docs/transformers/v5.20.2/en/model_doc/unlimited_ocr#transformers.UnlimitedOcrTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.20.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

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

## UnlimitedOcrVisionModel[[transformers.UnlimitedOcrVisionModel]]

#### transformers.UnlimitedOcrVisionModel[[transformers.UnlimitedOcrVisionModel]]

```python
transformers.UnlimitedOcrVisionModel(config: UnlimitedOcrVisionConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/unlimited_ocr/modeling_unlimited_ocr.py#L926)

**Parameters:**

config ([UnlimitedOcrVisionConfig](/docs/transformers/v5.20.2/en/model_doc/unlimited_ocr#transformers.UnlimitedOcrVisionConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.20.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Vision model encoding images first with SAM followed by an additional (CLIP) model.

This model inherits from [PreTrainedModel](/docs/transformers/v5.20.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.UnlimitedOcrVisionModel.forward]]

```python
forward(pixel_values: Tensor, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/unlimited_ocr/modeling_unlimited_ocr.py#L933)

**Parameters:**

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input images. Pixel values can be obtained using [UnlimitedOcrImageProcessor](/docs/transformers/v5.20.2/en/model_doc/unlimited_ocr#transformers.UnlimitedOcrImageProcessor). See [UnlimitedOcrImageProcessor.__call__()](/docs/transformers/v5.20.2/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([UnlimitedOcrProcessor](/docs/transformers/v5.20.2/en/model_doc/unlimited_ocr#transformers.UnlimitedOcrProcessor) uses [UnlimitedOcrImageProcessor](/docs/transformers/v5.20.2/en/model_doc/unlimited_ocr#transformers.UnlimitedOcrImageProcessor) for processing images).

**Returns:** [BaseModelOutput](/docs/transformers/v5.20.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

A [BaseModelOutput](/docs/transformers/v5.20.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([UnlimitedOcrConfig](/docs/transformers/v5.20.2/en/model_doc/unlimited_ocr#transformers.UnlimitedOcrConfig)) and inputs.

The [UnlimitedOcrVisionModel](/docs/transformers/v5.20.2/en/model_doc/unlimited_ocr#transformers.UnlimitedOcrVisionModel) forward method, overrides the `__call__` special method.

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

Example:

```python
>>> import torch
>>> from transformers import AutoConfig, AutoImageProcessor, AutoModel
>>> from transformers.image_utils import load_image

>>> model_id = "baidu/Unlimited-OCR"
>>> config = AutoConfig.from_pretrained(model_id)
>>> model = AutoModel.from_pretrained(model_id, config=config.vision_config, device_map="auto")
>>> image_processor = AutoImageProcessor.from_pretrained(model_id)

>>> image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_suggestion_form.jpg")
>>> inputs = image_processor(images=image, return_tensors="pt").to(model.device, model.dtype)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # Patch tokens without class tokens (batch_size, height * width, vision_hidden_size + sam_hidden_size)
>>> patch_tokens = outputs.last_hidden_state
```

## UnlimitedOcrModel[[transformers.UnlimitedOcrModel]]

#### transformers.UnlimitedOcrModel[[transformers.UnlimitedOcrModel]]

```python
transformers.UnlimitedOcrModel(config: UnlimitedOcrConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/unlimited_ocr/modeling_unlimited_ocr.py#L1813)

**Parameters:**

config ([UnlimitedOcrConfig](/docs/transformers/v5.20.2/en/model_doc/unlimited_ocr#transformers.UnlimitedOcrConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.20.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The UnlimitedOcr model which consists of a vision backbone and a language model.

This model inherits from [PreTrainedModel](/docs/transformers/v5.20.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.UnlimitedOcrModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, pixel_values: typing.Optional[torch.FloatTensor] = None, pixel_values_local: typing.Optional[torch.FloatTensor] = None, num_local_patches: list[int] | None = None, patches_grid: list[list[int]] | None = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/unlimited_ocr/modeling_unlimited_ocr.py#L1930)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.20.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.20.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.20.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [UnlimitedOcrImageProcessor](/docs/transformers/v5.20.2/en/model_doc/unlimited_ocr#transformers.UnlimitedOcrImageProcessor). See [UnlimitedOcrImageProcessor.__call__()](/docs/transformers/v5.20.2/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([UnlimitedOcrProcessor](/docs/transformers/v5.20.2/en/model_doc/unlimited_ocr#transformers.UnlimitedOcrProcessor) uses [UnlimitedOcrImageProcessor](/docs/transformers/v5.20.2/en/model_doc/unlimited_ocr#transformers.UnlimitedOcrImageProcessor) for processing images).

pixel_values_local (`torch.FloatTensor`, *optional*) : Local patch pixel values of shape `(total_patches, 3, H, W)`.

num_local_patches (`list[int]`, *optional*) : Number of local patches per image in the batch.

patches_grid (`list[list[int]]`, *optional*) : The patches grid `(num_columns, num_rows)` per image.

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.20.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.20.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** `UnlimitedOcrModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `UnlimitedOcrModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([UnlimitedOcrConfig](/docs/transformers/v5.20.2/en/model_doc/unlimited_ocr#transformers.UnlimitedOcrConfig)) and inputs.

The [UnlimitedOcrModel](/docs/transformers/v5.20.2/en/model_doc/unlimited_ocr#transformers.UnlimitedOcrModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.20.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

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
- **router_logits** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

  Router logits of the model, useful to compute the auxiliary loss for Mixture of Experts models.

## UnlimitedOcrForConditionalGeneration[[transformers.UnlimitedOcrForConditionalGeneration]]

#### transformers.UnlimitedOcrForConditionalGeneration[[transformers.UnlimitedOcrForConditionalGeneration]]

```python
transformers.UnlimitedOcrForConditionalGeneration(config: UnlimitedOcrConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/unlimited_ocr/modeling_unlimited_ocr.py#L1996)

**Parameters:**

config ([UnlimitedOcrConfig](/docs/transformers/v5.20.2/en/model_doc/unlimited_ocr#transformers.UnlimitedOcrConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.20.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Unlimited Ocr Model for token generation conditioned on other modalities (e.g. image-text-to-text generation).

This model inherits from [PreTrainedModel](/docs/transformers/v5.20.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.UnlimitedOcrForConditionalGeneration.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, pixel_values: typing.Optional[torch.FloatTensor] = None, pixel_values_local: typing.Optional[torch.FloatTensor] = None, num_local_patches: list[int] | None = None, patches_grid: list[list[int]] | None = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/unlimited_ocr/modeling_unlimited_ocr.py#L2034)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.20.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.20.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.20.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [UnlimitedOcrImageProcessor](/docs/transformers/v5.20.2/en/model_doc/unlimited_ocr#transformers.UnlimitedOcrImageProcessor). See [UnlimitedOcrImageProcessor.__call__()](/docs/transformers/v5.20.2/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([UnlimitedOcrProcessor](/docs/transformers/v5.20.2/en/model_doc/unlimited_ocr#transformers.UnlimitedOcrProcessor) uses [UnlimitedOcrImageProcessor](/docs/transformers/v5.20.2/en/model_doc/unlimited_ocr#transformers.UnlimitedOcrImageProcessor) for processing images).

pixel_values_local (`torch.FloatTensor`, *optional*) : Local patch pixel values of shape `(total_patches, 3, H, W)`.

num_local_patches (`list[int]`, *optional*) : Number of local patches per image in the batch.

patches_grid (`list[list[int]]`, *optional*) : The patches grid `(num_columns, num_rows)` per image.

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.20.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.20.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** `UnlimitedOcrCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `UnlimitedOcrCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([UnlimitedOcrConfig](/docs/transformers/v5.20.2/en/model_doc/unlimited_ocr#transformers.UnlimitedOcrConfig)) and inputs.

The [UnlimitedOcrForConditionalGeneration](/docs/transformers/v5.20.2/en/model_doc/unlimited_ocr#transformers.UnlimitedOcrForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.20.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_hidden_states** (`torch.FloatTensor`, *optional*) -- A `torch.FloatTensor` of size (batch_size * num_patches, num_images, sequence_length, hidden_size)`.
  image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
- **router_logits** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

  Router logits of the model, useful to compute the auxiliary loss for Mixture of Experts models.

Example single-page OCR:

```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText

>>> model = AutoModelForImageTextToText.from_pretrained("baidu/Unlimited-OCR", device_map="auto")
>>> processor = AutoProcessor.from_pretrained("baidu/Unlimited-OCR")

>>> image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_suggestion_form.jpg"
>>> messages = [
...     {
...         "role": "user",
...         "content": [{"type": "image", "url": image}, {"type": "text", "text": "document parsing."}],
...     }
... ]
>>> inputs = processor.apply_chat_template(
...     messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
... ).to(model.device, model.dtype)

>>> output = model.generate(**inputs)
>>> processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
>>> # image [383, 87, 497, 171]\ntext [333, 201, 558, 230]R&D QUALITY IMPROVEMENT\nSUGGESTION/SOLUTION FORM...
```

Example multi-page OCR:

```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText

>>> model = AutoModelForImageTextToText.from_pretrained("baidu/Unlimited-OCR", device_map="auto")
>>> processor = AutoProcessor.from_pretrained("baidu/Unlimited-OCR")

>>> page1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_suggestion_form.jpg"
>>> page2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_receipt.jpeg"

>>> messages = [
...     {
...         "role": "user",
...         "content": [{"type": "image", "url": page} for page in [page1, page2]]
...         + [{"type": "text", "text": "Multi page parsing."}],
...     }
... ]
>>> inputs = processor.apply_chat_template(
...     messages,
...     add_generation_prompt=True,
...     tokenize=True,
...     return_dict=True,
...     return_tensors="pt",
...     processor_kwargs={"crop_to_patches": False},
... ).to(model.device, model.dtype)

>>> output = model.generate(
...     **inputs,
...     no_repeat_ngram_window_size=1024,
... )
>>> processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
>>> # <PAGE>image [382, 87, 489, 174]\ntitle [333, 201, 556, 230]R&D QUALITY IMPROVEMENT\nSUGGESTION/SCLUTION FORM...
```

### NomicBERT
https://huggingface.co/docs/transformers/v5.20.2/model_doc/nomic_bert.md
