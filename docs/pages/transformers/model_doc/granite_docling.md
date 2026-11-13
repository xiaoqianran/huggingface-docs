# GraniteDocling

[GraniteDocling](https://huggingface.co/papers/2408.09869) is a vision-language model for document conversion. Given a page image, it generates [DocLang](https://doclang.ai/) (`<doclang>`): the layout elements in reading order with their bounding boxes, the text, tables, formulas and code they contain. DocLang is the markup; [Docling](https://github.com/docling-project/docling) (`docling-core`) parses a prediction into a document you can export to Markdown, HTML, JSON, and other formats.

The page is split into tiles of 512x512 pixels laid out on the grid that best matches its aspect ratio, plus a thumbnail of the whole page when there is more than one tile. A vision encoder embeds every tile, and a pixel-shuffle connector maps vision patches to image tokens. Intermediate vision encoder states are projected as well and added to the image tokens after the first decoder layers. A dense Granite-style text decoder then generates the DocLang.

GraniteDocling is released as two models. The 500M model uses 64 image tokens per tile and can switch a dense page to 256; a router chooses that per page. The 2B model uses a pixel-shuffle ratio of 2, which is 256 image tokens per tile, the same density as the 500M fine path.

The [Docling project](https://huggingface.co/docling-project) created GraniteDocling. Both models are published on the [IBM Granite](https://huggingface.co/ibm-granite) organization.

GraniteDocling is a different architecture from [granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M), which keeps loading through [Idefics3](./idefics3).

> [!TIP]
> This model was contributed by the [Docling team](https://huggingface.co/docling-project).
>
> Click on the GraniteDocling models in the right sidebar for more examples of how to apply GraniteDocling to different document conversion tasks.

## Usage tips

- Prompt the model through the chat template. `<doclang>` converts the full page. Append a bracket list to request a subset of elements, for example `<doclang> [<ocr>]` or `<doclang> [<ocr>, <layout>, <picture>]`.
- Decode with `skip_special_tokens=False` if you will parse the string as DocLang. Dedicated markup tokens are otherwise stripped.
- On the 500M model, pass `fine_route=True` for dense pages (small print, long tables, multi-column text). That path uses four times as many image tokens, so the prompt is longer and inference is slower. The 2B model already uses that token density and has no second path, so `fine_route=True` raises an error for it.
- The 500M model includes a density router, which decides that for you: [predict_fine_route()](/docs/transformers/v5.23.0/en/model_doc/granite_docling#transformers.GraniteDoclingModel.predict_fine_route) returns which pages of a batch need the fine path.

The examples below convert page 1 of the [Docling Technical Report](https://huggingface.co/papers/2408.09869) (`arXiv:2408.09869`). Change only the text prompt to switch task; the model returns DocLang filtered to the requested elements.

| Task | Prompt |
|---|---|
| Full page | `<doclang>` |
| OCR | `<doclang> [<ocr>]` |
| Layout | `<doclang> [<layout>]` |
| Tables | `<doclang> [<table>]` |
| Formulas | `<doclang> [<formula>]` |
| Code | `<doclang> [<code>]` |
| Charts | `<doclang> [<chart>]` |
| Pictures | `<doclang> [<picture>]` |
| Combined | `<doclang> [<ocr>, <layout>, <picture>]` |

### Full page

`<doclang>` converts the whole page: layout, text, tables, formulas, code, charts, and pictures.

```python
from transformers import pipeline

pipe = pipeline(
    task="image-text-to-text",
    model="ibm-granite/granite-for-docling-500m",
)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/ibm-granite/granite-for-docling-500m/resolve/main/docling_technical_report_p1.png"},
            {"type": "text", "text": "<doclang>"},
        ],
    }
]
pipe(text=messages, max_new_tokens=4096, return_full_text=False, skip_special_tokens=False)
```

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model_id = "ibm-granite/granite-for-docling-500m"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/ibm-granite/granite-for-docling-500m/resolve/main/docling_technical_report_p1.png"},
            {"type": "text", "text": "<doclang>"},
        ],
    },
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=4096)
# Keep the special tokens: they are the DocLang markup. Drop the end-of-text token that closes the generation.
doclang = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False)
doclang = doclang.removesuffix(processor.tokenizer.eos_token).strip()
print(doclang)
```

To convert only some elements, change the text prompt to one of the table above, for example `<doclang> [<ocr>, <layout>, <picture>]`. Everything else stays the same.

### Dense pages: the fine path

The fine connector path spends four times as many image tokens per tile, which helps on small print, long tables and multi-column text. Select it per call with `fine_route=True`:

```python
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    processor_kwargs={"fine_route": True},
).to(model.device)
output = model.generate(**inputs, max_new_tokens=4096)
```

The 500M model (`config.density_router_hidden_size` is set) predicts which pages need the fine path from its default image tokens:

```python
needs_fine = model.model.predict_fine_route(inputs["pixel_values"], inputs["pixel_attention_mask"])  # (batch_size,) bool
if needs_fine[0]:
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"fine_route": True},
    ).to(model.device)
output = model.generate(**inputs, max_new_tokens=4096)
```

### Batched inference

Pages get different tile grids, so `pixel_values` is padded to the largest tile count in the batch. `pixel_attention_mask` marks the real tiles and the model discards the padding.

```python
pages = [
    "https://huggingface.co/ibm-granite/granite-for-docling-500m/resolve/main/docling_technical_report_p1.png",
    "https://huggingface.co/ibm-granite/granite-for-docling-500m/resolve/main/docling_technical_report_p1.png",
]
conversations = [
    [{"role": "user", "content": [{"type": "image", "url": url}, {"type": "text", "text": "<doclang>"}]}]
    for url in pages
]
inputs = processor.apply_chat_template(
    conversations,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    processor_kwargs={"padding": True},
).to(model.device)

output = model.generate(**inputs, max_new_tokens=4096)
for row in output:
    doclang = processor.decode(row[inputs["input_ids"].shape[1] :], skip_special_tokens=False)
    print(doclang.removesuffix(processor.tokenizer.eos_token).strip())
```

Add `"fine_route": True` to `processor_kwargs` to route the whole batch through the fine path; the processor then returns `tile_fine_mask`. A batch takes one path, so run the pages the router sends to the fine path as their own batch.

### Export DocLang with Docling

`docling-core` parses the `doclang` string from the AutoModel example into a `DoclingDocument`, which you can serialize to whatever you need.

```python
from pathlib import Path
from docling_core.transforms.deserializer.doclang import DocLangDocDeserializer

document = DocLangDocDeserializer().deserialize_str(doclang)

print(document.export_to_markdown())
print(document.export_to_html())
print(document.export_to_dict())

document.save_as_markdown(Path("page.md"))
document.save_as_html(Path("page.html"))
document.save_as_json(Path("page.json"))
```

The [DocLang](https://doclang.ai/) spec and `doclang` toolkit (validate / pack) live at [doclang-project/doclang](https://github.com/doclang-project/doclang). The deserializer and the `DoclingDocument` exporters live in [`docling-core`](https://github.com/docling-project/docling-core).

## Notes

- A full page is long: budget `max_new_tokens` in the thousands, otherwise the DocLang is cut off silently.
- Tiles are 512x512, laid out on the grid that best matches the page's aspect ratio, capped by `max_patches` (default 32) and 16 tiles per side. A page that fits in one tile gets no thumbnail.
- A checkpoint can carry a spread rule in its image processor config: `dense_max_patches` tiles instead of `max_patches` for a wide landscape page (`dense_min_aspect` < width/height <= `dense_max_aspect`) that the `max_patches` grid would shrink below `dense_max_scale`. Newspaper spreads are the typical case. The rule is off by default (`dense_max_patches=0`) and must match what the checkpoint was trained with.
- On the 500M model the fine path quadruples the image tokens per tile, from 64 to 256. Use it for dense pages only, or let the router decide with [predict_fine_route()](/docs/transformers/v5.23.0/en/model_doc/granite_docling#transformers.GraniteDoclingModel.predict_fine_route). The 2B model uses 256 image tokens per tile throughout (pixel-shuffle ratio 2).

## GraniteDoclingConfig[[transformers.GraniteDoclingConfig]]

#### transformers.GraniteDoclingConfig[[transformers.GraniteDoclingConfig]]

```python
transformers.GraniteDoclingConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, text_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, vision_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, image_token_id: int = 100337, tie_word_embeddings: bool = True, use_cache: bool = True, pad_token_id: int | None = 100256, density_router_hidden_size: int | None = None, density_router_threshold: float = 0.4, density_router_loss_weight: float = 0.1)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/granite_docling/configuration_granite_docling.py#L160)

**Parameters:**

text_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the text backbone.

vision_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the vision backbone.

image_token_id (`int`, *optional*, defaults to `100337`) : The image token index used as a placeholder for input images.

tie_word_embeddings (`bool`, *optional*, defaults to `True`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

pad_token_id (`int`, *optional*, defaults to `100256`) : Token id used for padding in the vocabulary.

density_router_hidden_size (`int`, *optional*) : Hidden size of the density router, which predicts from the vision encoder features whether a page needs the fine connector path. No router is built when `None`.

density_router_threshold (`float`, *optional*, defaults to 0.4) : Probability above which the density router selects the fine connector path.

density_router_loss_weight (`float`, *optional*, defaults to 0.1) : Weight of the density router loss when `router_labels` are given.

This is the configuration class to store the configuration of a GraniteDoclingModel. It is used to instantiate a Granite Docling
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [ibm-granite/granite-for-docling-500m](https://huggingface.co/ibm-granite/granite-for-docling-500m)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## GraniteDoclingTextConfig[[transformers.GraniteDoclingTextConfig]]

#### transformers.GraniteDoclingTextConfig[[transformers.GraniteDoclingTextConfig]]

```python
transformers.GraniteDoclingTextConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vocab_size: int = 133800, hidden_size: int = 1024, intermediate_size: int = 2048, num_hidden_layers: int = 28, num_attention_heads: int = 16, num_key_value_heads: int | None = 4, hidden_act: str = 'silu', max_position_embeddings: int = 32768, initializer_range: float = 0.02, rms_norm_eps: float = 1e-05, use_cache: bool = True, pad_token_id: int | None = 100256, bos_token_id: int | None = 100257, eos_token_id: int | list[int] | None = 100257, tie_word_embeddings: bool = True, rope_parameters: transformers.modeling_rope_utils.RopeParameters | dict | None = None, attention_bias: bool = False, attention_dropout: float | int = 0.0, mlp_bias: bool = False, embedding_multiplier: float | int = 12.0, logits_scaling: float | int = 4.0, residual_multiplier: float | int = 0.263, attention_multiplier: float | int = 0.015625)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/granite_docling/configuration_granite_docling.py#L92)

**Parameters:**

vocab_size (`int`, *optional*, defaults to `133800`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

hidden_size (`int`, *optional*, defaults to `1024`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `2048`) : Dimension of the MLP representations.

num_hidden_layers (`int`, *optional*, defaults to `28`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `16`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*, defaults to `4`) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

hidden_act (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

max_position_embeddings (`int`, *optional*, defaults to `32768`) : The maximum sequence length that this model might ever be used with.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rms_norm_eps (`float`, *optional*, defaults to `1e-05`) : The epsilon used by the rms normalization layers.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

pad_token_id (`int`, *optional*, defaults to `100256`) : Token id used for padding in the vocabulary.

bos_token_id (`int`, *optional*, defaults to `100257`) : Token id used for beginning-of-stream in the vocabulary.

eos_token_id (`Union[int, list[int]]`, *optional*, defaults to `100257`) : Token id used for end-of-stream in the vocabulary.

tie_word_embeddings (`bool`, *optional*, defaults to `True`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

rope_parameters (`Union[~modeling_rope_utils.RopeParameters, dict]`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

attention_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

mlp_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.

embedding_multiplier (`Union[float, int]`, *optional*, defaults to `12.0`) : Scaling factor applied to the word embeddings. Used to scale the embeddings relative to the hidden size.

logits_scaling (`Union[float, int]`, *optional*, defaults to `4.0`) : Scaling factor applied to the output logits before computing the probability distribution.

residual_multiplier (`Union[float, int]`, *optional*, defaults to `0.263`) : Scaling factor applied to the residual connections.

attention_multiplier (`Union[float, int]`, *optional*, defaults to `0.015625`) : Scaling factor applied to the attention weights.

This is the configuration class to store the configuration of a GraniteDoclingModel. It is used to instantiate a Granite Docling
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [ibm-granite/granite-for-docling-500m](https://huggingface.co/ibm-granite/granite-for-docling-500m)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

```python
>>> from transformers import GraniteDoclingTextModel, GraniteDoclingTextConfig

>>> # Initializing a GraniteDoclingText granite_docling_text-3b style configuration
>>> configuration = GraniteDoclingTextConfig()

>>> # Initializing a model from the granite_docling_text-7b style configuration
>>> model = GraniteDoclingTextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## GraniteDoclingVisionConfig[[transformers.GraniteDoclingVisionConfig]]

#### transformers.GraniteDoclingVisionConfig[[transformers.GraniteDoclingVisionConfig]]

```python
transformers.GraniteDoclingVisionConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, hidden_size: int = 768, intermediate_size: int = 3072, num_hidden_layers: int = 12, num_attention_heads: int = 12, num_channels: int = 3, image_size: int | list[int] | tuple[int, int] = 512, patch_size: int | list[int] | tuple[int, int] = 16, hidden_act: str = 'gelu_pytorch_tanh', layer_norm_eps: float = 1e-06, attention_dropout: float | int = 0.0, initializer_range: float = 0.02, out_hidden_size: int = 1024, scale_factor: int = 4, deepstack_visual_indexes: list[int] | None = None, use_fine_route: bool = True)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/granite_docling/configuration_granite_docling.py#L36)

**Parameters:**

hidden_size (`int`, *optional*, defaults to `768`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `3072`) : Dimension of the MLP representations.

num_hidden_layers (`int`, *optional*, defaults to `12`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `12`) : Number of attention heads for each attention layer in the Transformer decoder.

num_channels (`int`, *optional*, defaults to `3`) : The number of input channels.

image_size (`Union[int, list[int], tuple[int, int]]`, *optional*, defaults to `512`) : The size (resolution) of each image.

patch_size (`Union[int, list[int], tuple[int, int]]`, *optional*, defaults to `16`) : The size (resolution) of each patch.

hidden_act (`str`, *optional*, defaults to `gelu_pytorch_tanh`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

layer_norm_eps (`float`, *optional*, defaults to `1e-06`) : The epsilon used by the layer normalization layers.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

out_hidden_size (`int`, *optional*, defaults to 1024) : Hidden size of the text decoder that the connector projects to.

scale_factor (`int`, *optional*, defaults to 4) : Pixel shuffle factor of the connector. Each tile of `(image_size // patch_size) ** 2` vision tokens is reduced by `scale_factor ** 2`.

deepstack_visual_indexes (`list[int]`, *optional*, defaults to `[3, 7, 10]`) : Indices of the vision encoder layers whose output is projected by the DeepStack mergers. Index 0 is the output of the first layer.

use_fine_route (`bool`, *optional*, defaults to `True`) : Whether to build the fine connector path, which shuffles pixels by half of `scale_factor` and yields four times as many image tokens per tile. The 500M checkpoint has it, the 2B checkpoint does not.

Configuration of the SigLIP-style vision encoder that embeds every 512x512 tile of a page, and of the connector
that turns its outputs into image tokens.

Example:

```python
>>> from transformers import GraniteDoclingVisionConfig, GraniteDoclingVisionModel

>>> # Initializing a GraniteDoclingVisionConfig with ibm-granite/granite-for-docling-500m style configuration
>>> configuration = GraniteDoclingVisionConfig()

>>> # Initializing a GraniteDoclingVisionModel (with random weights) from the ibm-granite/granite-for-docling-500m style configuration
>>> model = GraniteDoclingVisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## GraniteDoclingImageProcessor[[transformers.GraniteDoclingImageProcessor]]

#### transformers.GraniteDoclingImageProcessor[[transformers.GraniteDoclingImageProcessor]]

```python
transformers.GraniteDoclingImageProcessor(**kwargs: *<class 'transformers.models.granite_docling.image_processing_granite_docling.GraniteDoclingImageProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/granite_docling/image_processing_granite_docling.py#L241)

**Parameters:**

do_convert_rgb (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to convert the image to RGB.

do_resize (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to resize the image.

size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*, defaults to `{'height' : 512, 'width': 512}`): Describes the maximum input dimensions to the model.

default_to_square (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to default to a square image when resizing, if size is an int.

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

crop_to_patches (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to split the page into 512x512 tiles laid out on the grid that best matches its aspect ratio (at most 16 tiles per side). Can be overridden by the `crop_to_patches` parameter in the `preprocess` method.

min_patches (`int`, *kwargs*, *optional*, defaults to 1) : The minimum number of tiles per page. Only has an effect if `crop_to_patches` is set to `True`. Can be overridden by the `min_patches` parameter in the `preprocess` method.

max_patches (`int`, *kwargs*, *optional*, defaults to 32) : The maximum number of tiles per page. Only has an effect if `crop_to_patches` is set to `True`. Can be overridden by the `max_patches` parameter in the `preprocess` method.

fine_route (`bool`, *kwargs*, *optional*, defaults to `False`) : Whether the pages take the fine connector path, which yields four times as many image tokens per tile.

dense_max_patches (`int`, *kwargs*, *optional*, defaults to 0) : The maximum number of tiles for the pages the spread rule selects (0 disables the rule). A page is tiled with up to `dense_max_patches` tiles instead of `max_patches` when the `max_patches` grid shrinks it below `dense_max_scale` and its width/height ratio is above `dense_min_aspect` (and at most `dense_max_aspect` when that is set). The checkpoint must have been trained with the same rule.

dense_max_scale (`float`, *kwargs*, *optional*, defaults to 0.0) : Ratio of the tiled canvas width to the page width at `max_patches` below which a page counts as under-resolved. 0 skips this test.

dense_min_aspect (`float`, *kwargs*, *optional*, defaults to 0.0) : Width/height ratio a page must exceed to get `dense_max_patches`. 0 skips this test.

dense_max_aspect (`float`, *kwargs*, *optional*, defaults to 0.0) : Width/height ratio a page may not exceed to get `dense_max_patches`, which keeps strips such as single formula lines on `max_patches`. 0 means no upper bound.

Splits a page into 512x512 tiles on the grid that best matches its aspect ratio (`crop_to_patches`, at most
`max_patches` tiles and 16 per side), adds a thumbnail of the whole page when there is more than one tile, and
pads the samples of a batch to the same number of tiles. `pixel_attention_mask` marks the real tiles.
`fine_route=True` marks the samples for the fine connector path (`tile_fine_mask`, one value per sample).

#### preprocess[[transformers.GraniteDoclingImageProcessor.preprocess]]

```python
preprocess(images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']], *args, image_like_kwargs: dict[str, typing.Any] | None = None, **kwargs: *<class 'transformers.processing_utils.ImagesKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/image_processing_utils.py#L382)

**Parameters:**

images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]`) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

image_like_kwargs (`dict[str, Any]`, *optional*) : Developer flag for additional image like inputs that will also be preprocessed. Only use this if passing the inputs as kwarg doesn't work. For example, `preprocess(images, masks=masks)` is the preferred option but results in argument priority issues for some models. In those cases `preprocess(images, image_like_inputs={"masks": masks})` can be used instead.

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

**Returns:** `~image_processing_base.BatchFeature`

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

Preprocess an image or a batch of images.

## GraniteDoclingImageProcessorPil[[transformers.GraniteDoclingImageProcessorPil]]

#### transformers.GraniteDoclingImageProcessorPil[[transformers.GraniteDoclingImageProcessorPil]]

```python
transformers.GraniteDoclingImageProcessorPil(**kwargs: *<class 'transformers.models.granite_docling.image_processing_pil_granite_docling.GraniteDoclingImageProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/granite_docling/image_processing_pil_granite_docling.py#L239)

**Parameters:**

do_convert_rgb (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to convert the image to RGB.

do_resize (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to resize the image.

size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*, defaults to `{'height' : 512, 'width': 512}`): Describes the maximum input dimensions to the model.

default_to_square (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to default to a square image when resizing, if size is an int.

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

crop_to_patches (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to split the page into 512x512 tiles laid out on the grid that best matches its aspect ratio (at most 16 tiles per side). Can be overridden by the `crop_to_patches` parameter in the `preprocess` method.

min_patches (`int`, *kwargs*, *optional*, defaults to 1) : The minimum number of tiles per page. Only has an effect if `crop_to_patches` is set to `True`. Can be overridden by the `min_patches` parameter in the `preprocess` method.

max_patches (`int`, *kwargs*, *optional*, defaults to 32) : The maximum number of tiles per page. Only has an effect if `crop_to_patches` is set to `True`. Can be overridden by the `max_patches` parameter in the `preprocess` method.

fine_route (`bool`, *kwargs*, *optional*, defaults to `False`) : Whether the pages take the fine connector path, which yields four times as many image tokens per tile.

dense_max_patches (`int`, *kwargs*, *optional*, defaults to 0) : The maximum number of tiles for the pages the spread rule selects (0 disables the rule). A page is tiled with up to `dense_max_patches` tiles instead of `max_patches` when the `max_patches` grid shrinks it below `dense_max_scale` and its width/height ratio is above `dense_min_aspect` (and at most `dense_max_aspect` when that is set). The checkpoint must have been trained with the same rule.

dense_max_scale (`float`, *kwargs*, *optional*, defaults to 0.0) : Ratio of the tiled canvas width to the page width at `max_patches` below which a page counts as under-resolved. 0 skips this test.

dense_min_aspect (`float`, *kwargs*, *optional*, defaults to 0.0) : Width/height ratio a page must exceed to get `dense_max_patches`. 0 skips this test.

dense_max_aspect (`float`, *kwargs*, *optional*, defaults to 0.0) : Width/height ratio a page may not exceed to get `dense_max_patches`, which keeps strips such as single formula lines on `max_patches`. 0 means no upper bound.

PIL backend of [GraniteDoclingImageProcessor](/docs/transformers/v5.23.0/en/model_doc/granite_docling#transformers.GraniteDoclingImageProcessor): the same 512x512 tiling, thumbnail and tile padding.

#### preprocess[[transformers.GraniteDoclingImageProcessorPil.preprocess]]

```python
preprocess(images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']], *args, image_like_kwargs: dict[str, typing.Any] | None = None, **kwargs: *<class 'transformers.processing_utils.ImagesKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/image_processing_utils.py#L382)

**Parameters:**

images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]`) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

image_like_kwargs (`dict[str, Any]`, *optional*) : Developer flag for additional image like inputs that will also be preprocessed. Only use this if passing the inputs as kwarg doesn't work. For example, `preprocess(images, masks=masks)` is the preferred option but results in argument priority issues for some models. In those cases `preprocess(images, image_like_inputs={"masks": masks})` can be used instead.

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

**Returns:** `~image_processing_base.BatchFeature`

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

Preprocess an image or a batch of images.

## GraniteDoclingProcessor[[transformers.GraniteDoclingProcessor]]

#### transformers.GraniteDoclingProcessor[[transformers.GraniteDoclingProcessor]]

```python
transformers.GraniteDoclingProcessor(image_processor, tokenizer = None, image_seq_len: int = 64, chat_template: str | None = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/granite_docling/processing_granite_docling.py#L34)

**Parameters:**

image_processor (`GraniteDoclingImageProcessor`) : The image processor is a required input.

tokenizer (`tokenizer_class`) : The tokenizer is a required input.

image_seq_len (`int`, *optional*, defaults to 64) : The number of `<image>` tokens per tile on the coarse path. It is computed as `image_seq_len = int(((image_size // patch_size) ** 2) / (scale_factor**2))`.

chat_template (`str`, *optional*) : A Jinja template to convert lists of messages in a chat into a tokenizable string.

Constructs a GraniteDoclingProcessor which wraps a image processor and a tokenizer into a single processor.

[GraniteDoclingProcessor](/docs/transformers/v5.23.0/en/model_doc/granite_docling#transformers.GraniteDoclingProcessor) offers all the functionalities of [GraniteDoclingImageProcessor](/docs/transformers/v5.23.0/en/model_doc/granite_docling#transformers.GraniteDoclingImageProcessor) and `tokenizer_class`. See the
[~GraniteDoclingImageProcessor](/docs/transformers/v5.23.0/en/model_doc/granite_docling#transformers.GraniteDoclingImageProcessor) and `~tokenizer_class` for more information.

#### __call__[[transformers.GraniteDoclingProcessor.__call__]]

```python
__call__(images = None, text = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/granite_docling/processing_granite_docling.py#L55)

**Parameters:**

images (``) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

text (``) : The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings (pretokenized string). If you pass a pretokenized input, set `is_split_into_words=True` to avoid ambiguity with batched inputs.

return_tensors (`str` or [TensorType](/docs/transformers/v5.23.0/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

**Returns:** [BatchEncoding](/docs/transformers/v5.23.0/en/main_classes/tokenizer#transformers.BatchEncoding)

The tokenized prompts with `pixel_values` and `pixel_attention_mask`, and
`tile_fine_mask` when `fine_route=True`. The tile grids only build the prompt and are not returned.

## GraniteDoclingVisionModel[[transformers.GraniteDoclingVisionModel]]

#### transformers.GraniteDoclingVisionModel[[transformers.GraniteDoclingVisionModel]]

```python
transformers.GraniteDoclingVisionModel(config: GraniteDoclingVisionConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/granite_docling/modeling_granite_docling.py#L955)

**Parameters:**

config ([GraniteDoclingVisionConfig](/docs/transformers/v5.23.0/en/model_doc/granite_docling#transformers.GraniteDoclingVisionConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The vision encoder of GraniteDocling with its connector on top: embeds the tiles of a page and returns the
image tokens of every tile together with the projected DeepStack features.

This model inherits from [PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.GraniteDoclingVisionModel.forward]]

```python
forward(pixel_values: FloatTensor, fine: bool = False, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/granite_docling/modeling_granite_docling.py#L981)

**Parameters:**

pixel_values (`torch.FloatTensor` of shape `(num_tiles, num_channels, image_size, image_size)`) : Tiles flattened across the batch, without padding tiles.

fine (`bool`, *optional*, defaults to `False`) : Whether the tiles take the fine connector path, which yields four times as many image tokens per tile.

**Returns:**

Image tokens of every tile in tile order (`pooler_output`) and one projected DeepStack feature per
tapped vision layer (`deepstack_features`), with the same tile order.

The [GraniteDoclingVisionModel](/docs/transformers/v5.23.0/en/model_doc/granite_docling#transformers.GraniteDoclingVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## GraniteDoclingModel[[transformers.GraniteDoclingModel]]

#### transformers.GraniteDoclingModel[[transformers.GraniteDoclingModel]]

```python
transformers.GraniteDoclingModel(config: GraniteDoclingConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/granite_docling/modeling_granite_docling.py#L1025)

**Parameters:**

config ([GraniteDoclingConfig](/docs/transformers/v5.23.0/en/model_doc/granite_docling#transformers.GraniteDoclingConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

GraniteDocling model consisting of a vision encoder, a connector and a text decoder, outputting raw hidden
states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.GraniteDoclingModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, pixel_values: typing.Optional[torch.FloatTensor] = None, tile_fine_mask: typing.Optional[torch.BoolTensor] = None, pixel_attention_mask: typing.Optional[torch.BoolTensor] = None, mm_encoder_outputs: dict | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/granite_docling/modeling_granite_docling.py#L1091)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.23.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [GraniteDoclingImageProcessor](/docs/transformers/v5.23.0/en/model_doc/granite_docling#transformers.GraniteDoclingImageProcessor). See [GraniteDoclingImageProcessor.__call__()](/docs/transformers/v5.23.0/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([GraniteDoclingProcessor](/docs/transformers/v5.23.0/en/model_doc/granite_docling#transformers.GraniteDoclingProcessor) uses [GraniteDoclingImageProcessor](/docs/transformers/v5.23.0/en/model_doc/granite_docling#transformers.GraniteDoclingImageProcessor) for processing images).

tile_fine_mask (`torch.BoolTensor` of shape `(batch_size,)`, *optional*) : All True when the processor is called with `fine_route=True`: the batch takes the fine connector path, which yields four times as many image tokens per tile.

pixel_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_tiles)`, *optional*) : Real tiles, as returned by the processor. Padding tiles are false. Required with `pixel_values`.

mm_encoder_outputs (`dict`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** `GraniteDoclingBaseModelOutputWithPast` or `tuple(torch.Tensor)`

A `GraniteDoclingBaseModelOutputWithPast` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([GraniteDoclingConfig](/docs/transformers/v5.23.0/en/model_doc/granite_docling#transformers.GraniteDoclingConfig)) and inputs.

Inputs fed to the model can have an arbitrary number of images. To account for this, pixel_values fed to
the model have image padding -> (batch_size, max_num_tiles, 3, image_size, image_size) where max_num_tiles
is the maximum number of tiles among the batch_size samples in the batch. Padding tiles are false in
`pixel_attention_mask` and are discarded before the vision encoder.

- **last_hidden_state** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) -- Sequence of hidden-states at the output of the last layer of the model.
- **past_key_values** (`~cache_utils.Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
- **hidden_states** (`tuple[torch.Tensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.Tensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_hidden_states** (`tuple[torch.Tensor]` of shape `(batch_size, num_images, sequence_length, hidden_size)`, *optional*) -- Image hidden states of the model produced by the vision encoder and after projecting the last hidden state.
- **router_logits** (`torch.FloatTensor` of shape `(batch_size,)`, *optional*) -- Logits of the density router, one per sample. Returned when the model has a router and `pixel_values` are
  given.

#### get_image_features[[transformers.GraniteDoclingModel.get_image_features]]

```python
get_image_features(pixel_values: FloatTensor, pixel_attention_mask: BoolTensor, tile_fine_mask: typing.Optional[torch.BoolTensor] = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/granite_docling/modeling_granite_docling.py#L1052)

**Parameters:**

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_tiles, num_channels, image_size, image_size)`) : The tensors corresponding to the input image tiles.

pixel_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_tiles)`) : Real tiles, as returned by the processor. Padding tiles are false.

tile_fine_mask (`torch.BoolTensor` of shape `(batch_size,)`, *optional*) : Requires processor called with `fine_route=True`: the batch takes the fine connector path, which yields four times as many image tokens per tile.

**Returns:** `GraniteDoclingImageFeaturesOutput` or `tuple(torch.Tensor)`

A `GraniteDoclingImageFeaturesOutput` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([GraniteDoclingConfig](/docs/transformers/v5.23.0/en/model_doc/granite_docling#transformers.GraniteDoclingConfig)) and inputs.

- **last_hidden_state** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`tuple[torch.FloatTensor]` of length `batch_size`) -- Image tokens of each sample, in tile order, projected to the text hidden size. Each tensor has shape
  `(num_image_tokens, hidden_size)`. A tile yields `image_seq_len` tokens on the coarse connector path and four
  times as many on the fine path. Generation repeats these per-sample tensors across beams.
- **hidden_states** (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **deepstack_features** (`list[tuple[torch.FloatTensor]]`, *optional*) -- One projected intermediate vision encoder hidden state per entry of `config.deepstack_visual_indexes`. Each
  entry is split per sample, with the same layout as `pooler_output`.
- **router_logits** (`torch.FloatTensor` of shape `(batch_size,)`, *optional*) -- Logits of the density router, one per sample. Returned when the model has a router.

#### predict_fine_route[[transformers.GraniteDoclingModel.predict_fine_route]]

```python
predict_fine_route(pixel_values: FloatTensor, pixel_attention_mask: BoolTensor)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/granite_docling/modeling_granite_docling.py#L1198)

**Parameters:**

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_tiles, num_channels, image_size, image_size)`) : The tensors corresponding to the input image tiles.

pixel_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_tiles)`) : Real tiles, as returned by the processor.

**Returns:** `torch.BoolTensor` of shape `(batch_size,)`

`True` for the samples that need the fine connector path.

Predicts with the density router which samples need the fine connector path. Route them by calling the
processor again with `fine_route=True`.

## GraniteDoclingTextModel[[transformers.GraniteDoclingTextModel]]

#### transformers.GraniteDoclingTextModel[[transformers.GraniteDoclingTextModel]]

```python
transformers.GraniteDoclingTextModel(config: GraniteDoclingTextConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/granite_docling/modeling_granite_docling.py#L423)

**Parameters:**

config ([GraniteDoclingTextConfig](/docs/transformers/v5.23.0/en/model_doc/granite_docling#transformers.GraniteDoclingTextConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Granite Docling Text Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.GraniteDoclingTextModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, visual_pos_masks: typing.Optional[torch.BoolTensor] = None, deepstack_visual_embeds: list[torch.Tensor] | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/granite_docling/modeling_granite_docling.py#L441)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.23.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

visual_pos_masks (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*) : Positions of the image tokens that receive the DeepStack features.

deepstack_visual_embeds (`list[torch.Tensor]`, *optional*) : DeepStack features, each of shape `(num_image_tokens, hidden_size)`. Entry `i` is added to the hidden states at `visual_pos_masks` after decoder layer `i`.

**Returns:** [BaseModelOutputWithPast](/docs/transformers/v5.23.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.Tensor)`

A [BaseModelOutputWithPast](/docs/transformers/v5.23.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([GraniteDoclingConfig](/docs/transformers/v5.23.0/en/model_doc/granite_docling#transformers.GraniteDoclingConfig)) and inputs.

The [GraniteDoclingTextModel](/docs/transformers/v5.23.0/en/model_doc/granite_docling#transformers.GraniteDoclingTextModel) forward method, overrides the `__call__` special method.

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

## GraniteDoclingForConditionalGeneration[[transformers.GraniteDoclingForConditionalGeneration]]

#### transformers.GraniteDoclingForConditionalGeneration[[transformers.GraniteDoclingForConditionalGeneration]]

```python
transformers.GraniteDoclingForConditionalGeneration(config: GraniteDoclingConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/granite_docling/modeling_granite_docling.py#L1231)

**Parameters:**

config ([GraniteDoclingConfig](/docs/transformers/v5.23.0/en/model_doc/granite_docling#transformers.GraniteDoclingConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The GraniteDocling Model with a language modeling head. It is made up of a vision encoder, a connector and a
text decoder, with a language modeling head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.GraniteDoclingForConditionalGeneration.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, pixel_values: typing.Optional[torch.FloatTensor] = None, tile_fine_mask: typing.Optional[torch.BoolTensor] = None, pixel_attention_mask: typing.Optional[torch.BoolTensor] = None, mm_encoder_outputs: dict | None = None, labels: typing.Optional[torch.LongTensor] = None, router_labels: typing.Optional[torch.Tensor] = None, use_cache: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/granite_docling/modeling_granite_docling.py#L1269)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.23.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [GraniteDoclingImageProcessor](/docs/transformers/v5.23.0/en/model_doc/granite_docling#transformers.GraniteDoclingImageProcessor). See [GraniteDoclingImageProcessor.__call__()](/docs/transformers/v5.23.0/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([GraniteDoclingProcessor](/docs/transformers/v5.23.0/en/model_doc/granite_docling#transformers.GraniteDoclingProcessor) uses [GraniteDoclingImageProcessor](/docs/transformers/v5.23.0/en/model_doc/granite_docling#transformers.GraniteDoclingImageProcessor) for processing images).

tile_fine_mask (`torch.BoolTensor` of shape `(batch_size,)`, *optional*) : All True when the processor is called with `fine_route=True`: the batch takes the fine connector path, which yields four times as many image tokens per tile.

pixel_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_tiles)`, *optional*) : Real tiles, as returned by the processor. Padding tiles are false. Required with `pixel_values`.

mm_encoder_outputs (`dict`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or `model.image_token_id` (where `model` is your instance of `GraniteDoclingForConditionalGeneration`). Tokens with indices set to `model.image_token_id` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

router_labels (`torch.Tensor` of shape `(batch_size,)`, *optional*) : Targets of the density router, `1` (or `True`) for the samples that need the fine connector path and `0` otherwise. Given as float, bool or int; the router loss is a binary cross-entropy on the logits.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

**Returns:** `GraniteDoclingCausalLMOutputWithPast` or `tuple(torch.Tensor)`

A `GraniteDoclingCausalLMOutputWithPast` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([GraniteDoclingConfig](/docs/transformers/v5.23.0/en/model_doc/granite_docling#transformers.GraniteDoclingConfig)) and inputs.

The [GraniteDoclingForConditionalGeneration](/docs/transformers/v5.23.0/en/model_doc/granite_docling#transformers.GraniteDoclingForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`~cache_utils.Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
- **hidden_states** (`tuple[torch.Tensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.Tensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_hidden_states** (`tuple[torch.Tensor]` of shape `(batch_size, num_images, sequence_length, hidden_size)`, *optional*) -- Image hidden states of the model produced by the vision encoder and after projecting the last hidden state.
- **router_logits** (`torch.FloatTensor` of shape `(batch_size,)`, *optional*) -- Logits of the density router, one per sample. Returned when the model has a router and `pixel_values` are
  given.

Example:

```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText
>>> from transformers.image_utils import load_image

>>> image = load_image("https://huggingface.co/ibm-granite/granite-for-docling-500m/resolve/main/docling_technical_report_p1.png")

>>> processor = AutoProcessor.from_pretrained("ibm-granite/granite-for-docling-500m")
>>> model = AutoModelForImageTextToText.from_pretrained("ibm-granite/granite-for-docling-500m", device_map="auto")

>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {"type": "image"},
...             {"type": "text", "text": "<doclang>"},
...         ],
...     }
... ]
>>> prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
>>> inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

>>> generated_ids = model.generate(**inputs, max_new_tokens=4096)
>>> print(processor.decode(generated_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False))
```

#### get_image_features[[transformers.GraniteDoclingForConditionalGeneration.get_image_features]]

```python
get_image_features(pixel_values: Tensor, pixel_attention_mask: typing.Optional[torch.LongTensor] = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/granite_docling/modeling_granite_docling.py#L1252)

**Parameters:**

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input images.

pixel_attention_mask (`torch.LongTensor`, *optional*) : The attention mask indicating padded regions in the image.

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.23.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.Tensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.23.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([GraniteDoclingConfig](/docs/transformers/v5.23.0/en/model_doc/granite_docling#transformers.GraniteDoclingConfig)) and inputs.

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

Example:

```python
>>> from PIL import Image
>>> from transformers import AutoProcessor, GraniteDoclingForConditionalGeneration

>>> model = GraniteDoclingForConditionalGeneration.from_pretrained("ibm-granite/granite-for-docling-500m")
>>> processor = AutoProcessor.from_pretrained("ibm-granite/granite-for-docling-500m")

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

### DINOv2 with Registers
https://huggingface.co/docs/transformers/v5.23.0/model_doc/dinov2_with_registers.md
