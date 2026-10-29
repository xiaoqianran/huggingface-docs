# MiniCPM-OCR

MiniCPM-OCR is a document-understanding model from the [MiniCPM-V](https://huggingface.co/papers/2509.18154) family developed by [OpenBMB](https://github.com/OpenBMB). It keeps the [MiniCPM-V 4.6](./minicpmv4_6) architecture -- a [SigLIP](./siglip) vision encoder with a window-attention merger in front of a [Qwen3.5](./qwen3_5) language model -- and specializes it for reading text-dense images such as scanned pages, screenshots, receipts and tables.

This model was contributed by [OpenBMB](https://huggingface.co/openbmb).
The original code can be found [here](https://github.com/OpenBMB/MiniCPM-V).

## Usage example

```python
from transformers import pipeline

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_receipt.jpeg",
            },
            {"type": "text", "text": "Transcribe all the text in this document."},
        ],
    },
]

pipe = pipeline("image-text-to-text", model="openbmb/MiniCPM-OCR-1.0")
outputs = pipe(text=messages, max_new_tokens=256, return_full_text=False)
print(outputs[0]["generated_text"])
# PSPS
# Document No : TD01167104
# Date : 25/12/2018 8:13:39 PM
# Cashier : MANIS
# Member :
#
# CASH BILL
#
# CODE/DESC PRICE Disc AMOUNT
# QTY RM
# 9556939040118 KF MODELLING CLAY KIDDY FISH
# 1 PC * 9.00 0.00 9.00
# Total : 9.00
# ...
```

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "openbmb/MiniCPM-OCR-1.0"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_suggestion_form.jpg"},
            {"type": "text", "text": "Convert this page to markdown."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device, dtype=model.dtype)

output = model.generate(**inputs, max_new_tokens=512)
decoded_output = processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(decoded_output)
# But, let's break this down and convert the page to markdown.
#
# ## R&D QUALITY IMPROVEMENT
# ### SUGGESTION/SOLUTION FORM
#
# #### Name/Phone Ext.: M. Hamann, P. Harper, P. Martinez  Date: 9/3/92
#
# #### Supervisor/Manager: J. S. Wigand  R&D Group: Licensee
# ...
```

The chat template opens a thinking block unless `enable_thinking=False` is passed to
`apply_chat_template`, which is why the decoded text above starts with the model reasoning about
the page before transcribing it.

MiniCPM-OCR supports both visual downsampling modes of the MiniCPM-V family:

- **4x** (default): Less downsampling, more visual tokens, which is what document and screenshot transcription needs.
- **16x**: More aggressive downsampling, fewer visual tokens, faster inference on natural images.

The mode has to be passed to the processor and to `model.generate` together, because the processor counts the visual placeholders the vision tower is expected to produce:

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "openbmb/MiniCPM-OCR-1.0"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_suggestion_form.jpg"},
            {"type": "text", "text": "Convert this page to markdown."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
    processor_kwargs={"downsample_mode": "4x"},
).to(model.device, dtype=model.dtype)

output = model.generate(**inputs, max_new_tokens=512, downsample_mode="4x")
print((inputs["input_ids"] == model.config.image_token_id).sum().item())
# 1260 visual tokens; the same page in `"16x"` mode is 315 instead
```

A per-call `downsample_mode` applies to that call only and leaves the processor unchanged, so the
same processor can serve both modes concurrently. When calling the processor directly instead of
going through `apply_chat_template`, pass it as a plain keyword argument:

```python
inputs = processor(images=[image], text=[text], return_tensors="pt", downsample_mode="4x")
```

To make a mode the default instead, set it when loading the processor:

```python
processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-OCR-1.0", downsample_mode="4x")
```

Images too small to be sliced are upscaled to `scale_resolution` by default, which is the setting
the released checkpoint was trained with. Screenshots and crops are often better kept at their
native resolution, where upscaling only blurs the glyphs, so `small_image_upsize` selects between
the two:

- `"always"` (default): upscale them to `scale_resolution`.
- `"never"`: keep them at their native resolution, aligned to the patch grid only.
- `"adaptive"`: decide per image, and per slice, from glyph size and density.

```python
from transformers import AutoProcessor, load_image

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_receipt.jpeg"
image = load_image(url).resize((224, 224))

processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-OCR-1.0")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Read the text."}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

print(processor(images=[image], text=[text], return_tensors="pt")["target_sizes"].tolist())
# [[32, 32]] -- upscaled to `scale_resolution`

processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-OCR-1.0", small_image_upsize="never")
print(processor(images=[image], text=[text], return_tensors="pt")["target_sizes"].tolist())
# [[16, 16]] -- 224 / 14 patches per side, no upscaling
```

`"adaptive"` upscales small dense text to `scale_resolution`, images below 6272 pixels to 84, and
keeps the rest native; on a slice it never spends more tokens than the baseline tile would. Glyph
height and density are measured with an Otsu threshold followed by connected component analysis, so
it requires [OpenCV](https://pypi.org/project/opencv-python/) and raises without it, rather than
silently falling back to a different rule.

```python
processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-OCR-1.0", small_image_upsize="adaptive")
print(processor(images=[image], text=[text], return_tensors="pt")["target_sizes"].tolist())
# [[32, 32]] -- dense text detected, so this one is upscaled anyway
```

Like `downsample_mode`, it can be overridden per call as well, which keeps the 224x224 receipt at
its native resolution without reloading the processor:

```python
inputs = processor(images=[image], text=[text], return_tensors="pt", small_image_upsize="never")
print(inputs["target_sizes"].tolist())
# [[16, 16]] -- 224 / 14 patches per side, no upscaling
```

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "openbmb/MiniCPM-OCR-1.0"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")

messages = [
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_receipt.jpeg"},
                {"type": "text", "text": "Transcribe all the text in this document."},
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/ocr_suggestion_form.jpg"},
                {"type": "text", "text": "Transcribe all the text in this document."},
            ],
        }
    ],
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
    processor_kwargs={"padding": True},
).to(model.device, dtype=model.dtype)

output = model.generate(**inputs, max_new_tokens=256)
decoded_outputs = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(decoded_outputs[0])
# PSPS
# Document No: TD01167104
# Date: 25/12/2018
# Cashier: MANIS
# ...
print(decoded_outputs[1])
# ... thinking preamble ...
# **R&D QUALITY IMPROVEMENT
# SUGGESTION/SOLUTION FORM**
#
# Name/Phone Ext.: M. Hamann, P. Harper, P. Martinez  Date: 9/3/92
# ...
```

The two pages are resized independently, so their prompts have different lengths. `padding` has to
be requested through `processor_kwargs` for `apply_chat_template` to forward it to the processor,
and the processor pads on the left so that `generate` keeps producing from the end of each prompt.

## MiniCPMOcrConfig[[transformers.MiniCPMOcrConfig]]

#### transformers.MiniCPMOcrConfig[[transformers.MiniCPMOcrConfig]]

```python
transformers.MiniCPMOcrConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, text_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, vision_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, insert_layer_id: int = 6, image_token_id: int | None = None, video_token_id: int | None = None, tie_word_embeddings: bool = True, downsample_mode: str = '4x', merge_kernel_size: tuple[int, int] | list[int] = (2, 2), merger_times: int = 1)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/minicpm_ocr/configuration_minicpm_ocr.py#L66)

**Parameters:**

text_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the text backbone.

vision_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the vision backbone.

insert_layer_id (`int`, *optional*, defaults to 6) : Vision encoder layer index after which the window-attention merger is applied.

image_token_id (`int`, *optional*) : The image token index used as a placeholder for input images.

video_token_id (`int`, *optional*) : The video token index used as a placeholder for input videos.

tie_word_embeddings (`bool`, *optional*, defaults to `True`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

downsample_mode (`str`, *optional*, defaults to `"4x"`) : Visual token downsampling ratio. `"4x"` keeps 4× more tokens than `"16x"`, which is what text-dense document images are read with.

merge_kernel_size (`tuple[int, int]`, *optional*, defaults to `(2, 2)`) : Kernel size `(h, w)` for merging adjacent visual patches in the Merger.

merger_times (`int`, *optional*, defaults to 1) : Number of iterative merge rounds in the Merger.

This is the configuration class to store the configuration of a MiniCPMOcrModel. It is used to instantiate a Minicpm Ocr
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [openbmb/MiniCPM-OCR-1.0](https://huggingface.co/openbmb/MiniCPM-OCR-1.0)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## MiniCPMOcrVisionConfig[[transformers.MiniCPMOcrVisionConfig]]

#### transformers.MiniCPMOcrVisionConfig[[transformers.MiniCPMOcrVisionConfig]]

```python
transformers.MiniCPMOcrVisionConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, hidden_size: int = 768, intermediate_size: int = 3072, num_hidden_layers: int = 12, num_attention_heads: int = 12, num_channels: int = 3, image_size: int | list[int] | tuple[int, int] = 224, patch_size: int | list[int] | tuple[int, int] = 16, hidden_act: str = 'gelu_pytorch_tanh', layer_norm_eps: float = 1e-06, attention_dropout: float | int = 0.0, insert_layer_id: int = 6, window_kernel_size: tuple[int, int] | list[int] = (2, 2))
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/minicpm_ocr/configuration_minicpm_ocr.py#L31)

**Parameters:**

hidden_size (`int`, *optional*, defaults to `768`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `3072`) : Dimension of the MLP representations.

num_hidden_layers (`int`, *optional*, defaults to `12`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `12`) : Number of attention heads for each attention layer in the Transformer decoder.

num_channels (`int`, *optional*, defaults to `3`) : The number of input channels.

image_size (`Union[int, list[int], tuple[int, int]]`, *optional*, defaults to `224`) : The size (resolution) of each image.

patch_size (`Union[int, list[int], tuple[int, int]]`, *optional*, defaults to `16`) : The size (resolution) of each patch.

hidden_act (`str`, *optional*, defaults to `gelu_pytorch_tanh`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

layer_norm_eps (`float`, *optional*, defaults to `1e-06`) : The epsilon used by the layer normalization layers.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

insert_layer_id (`int`, *optional*, defaults to 6) : Vision encoder layer index after which the window-attention merger is applied.

window_kernel_size (`tuple[int, int]`, *optional*, defaults to `(2, 2)`) : Window size `(h, w)` for the intermediate window-attention merger.

This is the configuration class to store the configuration of a MiniCPMOcrModel. It is used to instantiate a Minicpm Ocr
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [openbmb/MiniCPM-OCR-1.0](https://huggingface.co/openbmb/MiniCPM-OCR-1.0)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## MiniCPMOcrVisionPreTrainedModel[[transformers.MiniCPMOcrVisionPreTrainedModel]]

#### transformers.MiniCPMOcrVisionPreTrainedModel[[transformers.MiniCPMOcrVisionPreTrainedModel]]

```python
transformers.MiniCPMOcrVisionPreTrainedModel(config: PreTrainedConfig, *inputs, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/minicpm_ocr/modeling_minicpm_ocr.py#L396)

#### forward[[transformers.MiniCPMOcrVisionPreTrainedModel.forward]]

```python
forward(*args, **kwargs)
```

A mock value for a dotted path (e.g. `torch.float32`): attribute access chains,
calls behave as pass-through decorators, `repr` is the dotted path, and using it
as a base class substitutes a plain-`type` base (PEP 560 `__mro_entries__`), so
real subclasses keep a normal metaclass and `inspect.signature` reads their real
`__init__` instead of a mock's.

## MiniCPMOcrVisionModel[[transformers.MiniCPMOcrVisionModel]]

#### transformers.MiniCPMOcrVisionModel[[transformers.MiniCPMOcrVisionModel]]

```python
transformers.MiniCPMOcrVisionModel(config: MiniCPMOcrVisionConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/minicpm_ocr/modeling_minicpm_ocr.py#L410)

#### forward[[transformers.MiniCPMOcrVisionModel.forward]]

```python
forward(pixel_values, target_sizes: typing.Optional[torch.IntTensor] = None, use_vit_merger: bool = True, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/minicpm_ocr/modeling_minicpm_ocr.py#L442)

**Parameters:**

pixel_values (`` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input images. Pixel values can be obtained using [MiniCPMOcrImageProcessor](/docs/transformers/v5.21.1/en/model_doc/minicpm_ocr#transformers.MiniCPMOcrImageProcessor). See [MiniCPMOcrImageProcessor.__call__()](/docs/transformers/v5.21.1/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([MiniCPMOcrProcessor](/docs/transformers/v5.21.1/en/model_doc/minicpm_ocr#transformers.MiniCPMOcrProcessor) uses [MiniCPMOcrImageProcessor](/docs/transformers/v5.21.1/en/model_doc/minicpm_ocr#transformers.MiniCPMOcrImageProcessor) for processing images).

target_sizes (`torch.IntTensor` of shape `(batch_size, 2)`, *optional*) : Patch grid sizes `(h, w)` for computing position embeddings.

use_vit_merger (`bool`, *optional*, defaults to `True`) : Whether to apply the ViT window-attention merger after the encoder.

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MiniCPMOcrConfig](/docs/transformers/v5.21.1/en/model_doc/minicpm_ocr#transformers.MiniCPMOcrConfig)) and inputs.

The [MiniCPMOcrVisionModel](/docs/transformers/v5.21.1/en/model_doc/minicpm_ocr#transformers.MiniCPMOcrVisionModel) forward method, overrides the `__call__` special method.

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

## MiniCPMOcrModel[[transformers.MiniCPMOcrModel]]

#### transformers.MiniCPMOcrModel[[transformers.MiniCPMOcrModel]]

```python
transformers.MiniCPMOcrModel(config: MiniCPMOcrConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/minicpm_ocr/modeling_minicpm_ocr.py#L593)

**Parameters:**

config ([MiniCPMOcrConfig](/docs/transformers/v5.21.1/en/model_doc/minicpm_ocr#transformers.MiniCPMOcrConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The MiniCPMOcr model which consists of a vision backbone and a language model, without a language modeling head.

This model inherits from [PreTrainedModel](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.MiniCPMOcrModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, pixel_values: typing.Optional[torch.FloatTensor] = None, target_sizes: typing.Optional[torch.IntTensor] = None, pixel_values_videos: typing.Optional[torch.FloatTensor] = None, target_sizes_videos: typing.Optional[torch.IntTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: list[torch.FloatTensor] | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, downsample_mode: str | None = None, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/minicpm_ocr/modeling_minicpm_ocr.py#L662)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.21.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

pixel_values (`torch.FloatTensor`, *optional*) : Pixel value patches for images, NaViT-packed.

target_sizes (`torch.IntTensor`, *optional*) : Height and width (in patches) for each image.

pixel_values_videos (`torch.FloatTensor`, *optional*) : Pixel value patches for video frames, NaViT-packed.

target_sizes_videos (`torch.IntTensor`, *optional*) : Height and width (in patches) for each video frame.

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`list[torch.FloatTensor]`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.21.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.21.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

downsample_mode (`str`, *optional*) : `"4x"` keeps 4x more visual tokens; default `"16x"` applies full merge.

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** [BaseModelOutputWithPast](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPast](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MiniCPMOcrConfig](/docs/transformers/v5.21.1/en/model_doc/minicpm_ocr#transformers.MiniCPMOcrConfig)) and inputs.

The [MiniCPMOcrModel](/docs/transformers/v5.21.1/en/model_doc/minicpm_ocr#transformers.MiniCPMOcrModel) forward method, overrides the `__call__` special method.

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

#### get_image_features[[transformers.MiniCPMOcrModel.get_image_features]]

```python
get_image_features(pixel_values: FloatTensor, target_sizes: IntTensor, downsample_mode: str | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/minicpm_ocr/modeling_minicpm_ocr.py#L602)

**Parameters:**

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input images. Pixel values can be obtained using [MiniCPMOcrImageProcessor](/docs/transformers/v5.21.1/en/model_doc/minicpm_ocr#transformers.MiniCPMOcrImageProcessor). See [MiniCPMOcrImageProcessor.__call__()](/docs/transformers/v5.21.1/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([MiniCPMOcrProcessor](/docs/transformers/v5.21.1/en/model_doc/minicpm_ocr#transformers.MiniCPMOcrProcessor) uses [MiniCPMOcrImageProcessor](/docs/transformers/v5.21.1/en/model_doc/minicpm_ocr#transformers.MiniCPMOcrImageProcessor) for processing images).

target_sizes (`torch.IntTensor` of shape `(num_images, 2)`) : Height and width (in patches) of each image.

downsample_mode (`str`, *optional*) : When set to `"4x"` the intermediate `vit_merger` is skipped so that each image keeps `4×` more visual tokens. Default `"16x"` mode applies the full merge pipeline.

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MiniCPMOcrConfig](/docs/transformers/v5.21.1/en/model_doc/minicpm_ocr#transformers.MiniCPMOcrConfig)) and inputs.

Extract image features: vision encoder, insert merger, then MLP merger.

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

#### get_video_features[[transformers.MiniCPMOcrModel.get_video_features]]

```python
get_video_features(pixel_values_videos: FloatTensor, target_sizes_videos: IntTensor, downsample_mode: str | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/minicpm_ocr/modeling_minicpm_ocr.py#L743)

**Parameters:**

pixel_values_videos (`torch.FloatTensor` of shape `(1, channels, patch_size, seq_len)`) : NaViT-packed pixel patches for all video frames. The video processor concatenates every frame's patches along the last dimension into a single sequence with dim-0 = 1, identical to the image packing format.

target_sizes_videos (`torch.IntTensor` of shape `(num_patches, 2)`) : Height and width (in patches) of each visual unit.

downsample_mode (`str`, *optional*) : When set to `"4x"` the intermediate `vit_merger` is skipped so that each frame keeps `4×` more visual tokens. Default `"16x"` mode applies the full merge pipeline.

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MiniCPMOcrConfig](/docs/transformers/v5.21.1/en/model_doc/minicpm_ocr#transformers.MiniCPMOcrConfig)) and inputs.

Extract video features: repack frames into NaViT format, then vision encoder + merger.

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

## MiniCPMOcrForConditionalGeneration[[transformers.MiniCPMOcrForConditionalGeneration]]

#### transformers.MiniCPMOcrForConditionalGeneration[[transformers.MiniCPMOcrForConditionalGeneration]]

```python
transformers.MiniCPMOcrForConditionalGeneration(config: MiniCPMOcrConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/minicpm_ocr/modeling_minicpm_ocr.py#L779)

**Parameters:**

config ([MiniCPMOcrConfig](/docs/transformers/v5.21.1/en/model_doc/minicpm_ocr#transformers.MiniCPMOcrConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Minicpm Ocr Model for token generation conditioned on other modalities (e.g. image-text-to-text generation).

This model inherits from [PreTrainedModel](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.MiniCPMOcrForConditionalGeneration.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, pixel_values: typing.Optional[torch.FloatTensor] = None, target_sizes: typing.Optional[torch.IntTensor] = None, pixel_values_videos: typing.Optional[torch.FloatTensor] = None, target_sizes_videos: typing.Optional[torch.IntTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: list[torch.FloatTensor] | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, downsample_mode: str | None = None, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/minicpm_ocr/modeling_minicpm_ocr.py#L790)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.21.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

pixel_values (`torch.FloatTensor`, *optional*) : Pixel value patches for images, NaViT-packed.

target_sizes (`torch.IntTensor`, *optional*) : Height and width (in patches) for each image.

pixel_values_videos (`torch.FloatTensor`, *optional*) : Pixel value patches for video frames, NaViT-packed.

target_sizes_videos (`torch.IntTensor`, *optional*) : Height and width (in patches) for each video frame.

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`list[torch.FloatTensor]`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.21.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.21.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

downsample_mode (`str`, *optional*) : `"4x"` keeps 4x more visual tokens; default `"16x"` applies full merge.

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** [CausalLMOutputWithPast](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

A [CausalLMOutputWithPast](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MiniCPMOcrConfig](/docs/transformers/v5.21.1/en/model_doc/minicpm_ocr#transformers.MiniCPMOcrConfig)) and inputs.

The [MiniCPMOcrForConditionalGeneration](/docs/transformers/v5.21.1/en/model_doc/minicpm_ocr#transformers.MiniCPMOcrForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.21.1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

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
>>> from transformers import AutoProcessor, MiniCPMOcrForConditionalGeneration

>>> model = MiniCPMOcrForConditionalGeneration.from_pretrained("openbmb/MiniCPM-OCR-1.0")
>>> processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-OCR-1.0")

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

#### get_image_features[[transformers.MiniCPMOcrForConditionalGeneration.get_image_features]]

```python
get_image_features(*args, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/minicpm_ocr/modeling_minicpm_ocr.py#L854)

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MiniCPMOcrConfig](/docs/transformers/v5.21.1/en/model_doc/minicpm_ocr#transformers.MiniCPMOcrConfig)) and inputs.

Extract image features: vision encoder, insert merger, then MLP merger.

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

Example:

```python
>>> from PIL import Image
>>> from transformers import AutoProcessor, MiniCPMOcrForConditionalGeneration

>>> model = MiniCPMOcrForConditionalGeneration.from_pretrained("openbmb/MiniCPM-OCR-1.0")
>>> processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-OCR-1.0")

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

#### get_video_features[[transformers.MiniCPMOcrForConditionalGeneration.get_video_features]]

```python
get_video_features(*args, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/minicpm_ocr/modeling_minicpm_ocr.py#L858)

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MiniCPMOcrConfig](/docs/transformers/v5.21.1/en/model_doc/minicpm_ocr#transformers.MiniCPMOcrConfig)) and inputs.

Extract video features: repack frames into NaViT format, then vision encoder + merger.

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

Example:

```python
>>> from PIL import Image
>>> from transformers import AutoProcessor, MiniCPMOcrForConditionalGeneration

>>> model = MiniCPMOcrForConditionalGeneration.from_pretrained("openbmb/MiniCPM-OCR-1.0")
>>> processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-OCR-1.0")

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

## MiniCPMOcrProcessor[[transformers.MiniCPMOcrProcessor]]

#### transformers.MiniCPMOcrProcessor[[transformers.MiniCPMOcrProcessor]]

```python
transformers.MiniCPMOcrProcessor(image_processor = None, video_processor = None, tokenizer = None, chat_template = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/minicpm_ocr/processing_minicpm_ocr.py#L47)

**Parameters:**

image_processor (`MiniCPMOcrImageProcessor`) : The image processor is a required input.

video_processor (`MiniCPMOcrVideoProcessor`) : The video processor is a required input.

tokenizer (`TokenizersBackend`) : The tokenizer is a required input.

chat_template (`str`) : A Jinja template to convert lists of messages in a chat into a tokenizable string.

Constructs a MiniCPMOcrProcessor which wraps a image processor, a video processor, and a tokenizer into a single processor.

[MiniCPMOcrProcessor](/docs/transformers/v5.21.1/en/model_doc/minicpm_ocr#transformers.MiniCPMOcrProcessor) offers all the functionalities of [MiniCPMOcrImageProcessor](/docs/transformers/v5.21.1/en/model_doc/minicpm_ocr#transformers.MiniCPMOcrImageProcessor), [MiniCPMOcrVideoProcessor](/docs/transformers/v5.21.1/en/model_doc/minicpm_ocr#transformers.MiniCPMOcrVideoProcessor), and [TokenizersBackend](/docs/transformers/v5.21.1/en/main_classes/tokenizer#transformers.TokenizersBackend). See the
[~MiniCPMOcrImageProcessor](/docs/transformers/v5.21.1/en/model_doc/minicpm_ocr#transformers.MiniCPMOcrImageProcessor), [~MiniCPMOcrVideoProcessor](/docs/transformers/v5.21.1/en/model_doc/minicpm_ocr#transformers.MiniCPMOcrVideoProcessor), and [~TokenizersBackend](/docs/transformers/v5.21.1/en/main_classes/tokenizer#transformers.TokenizersBackend) for more information.

#### __call__[[transformers.MiniCPMOcrProcessor.__call__]]

```python
__call__(images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None, text: str | list[str] | list[list[str]] | None = None, videos: typing.Union[list['PIL.Image.Image'], numpy.ndarray, ForwardRef('torch.Tensor'), list[numpy.ndarray], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list[numpy.ndarray]], list[list['torch.Tensor']], transformers.video_utils.URL, list[transformers.video_utils.URL], list[list[transformers.video_utils.URL]], transformers.video_utils.Path, list[transformers.video_utils.Path], list[list[transformers.video_utils.Path]], NoneType] = None, **kwargs: *<class 'transformers.processing_utils.ProcessingKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/minicpm_ocr/processing_minicpm_ocr.py#L70)

**Parameters:**

images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]`, *optional*) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

text (`Union[str, list[str], list[list[str]]]`, *optional*) : The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings (pretokenized string). If you pass a pretokenized input, set `is_split_into_words=True` to avoid ambiguity with batched inputs.

videos (`Union[list[PIL.Image.Image], numpy.ndarray, torch.Tensor, list[numpy.ndarray], list[torch.Tensor], list[list[PIL.Image.Image]], list[list[numpy.ndarray]], list[list[torch.Tensor]], ~video_utils.URL, list[~video_utils.URL], list[list[~video_utils.URL]], ~video_utils.Path, list[~video_utils.Path], list[list[~video_utils.Path]]]`, *optional*) : Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If passing in videos with pixel values between 0 and 1, set `do_rescale=False`.

return_tensors (`str` or [TensorType](/docs/transformers/v5.21.1/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

- ****kwargs** ([ProcessingKwargs](/docs/transformers/v5.21.1/en/main_classes/processors#transformers.ProcessingKwargs), *optional*) : Additional processing options for each modality (text, images, videos, audio). Model-specific parameters are listed above; see the TypedDict class for the complete list of supported arguments.

## MiniCPMOcrImageProcessor[[transformers.MiniCPMOcrImageProcessor]]

#### transformers.MiniCPMOcrImageProcessor[[transformers.MiniCPMOcrImageProcessor]]

```python
transformers.MiniCPMOcrImageProcessor(**kwargs: *<class 'transformers.models.minicpm_ocr.image_processing_minicpm_ocr.MiniCPMOcrImageProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/minicpm_ocr/image_processing_minicpm_ocr.py#L139)

**Parameters:**

do_convert_rgb (*bool*, *kwargs*, *optional*, defaults to *True*) : Whether to convert the image to RGB.

do_resize (*bool*, *kwargs*, *optional*, defaults to *True*) : Whether to resize the image.

size (*Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]*, *kwargs*) : Describes the maximum input dimensions to the model.

default_to_square (*bool*, *kwargs*, *optional*, defaults to *True*) : Whether to default to a square image when resizing, if size is an int.

crop_size (*Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]*, *kwargs*) : Size of the output image after applying *center_crop*.

resample (*Annotated[Union[int, PILImageResampling, NoneType], None]*, *kwargs*, defaults to *3*) : Resampling filter to use if resizing the image. This can be one of the enum *PILImageResampling*. Only has an effect if *do_resize* is set to *True*.

do_rescale (*bool*, *kwargs*, *optional*, defaults to *True*) : Whether to rescale the image.

rescale_factor (*float*, *kwargs*, *optional*, defaults to *0.00392156862745098*) : Rescale factor to rescale the image by if *do_rescale* is set to *True*.

do_normalize (*bool*, *kwargs*, *optional*, defaults to *True*) : Whether to normalize the image.

image_mean (*Union[float, list[float], tuple[float, ...]]*, *kwargs*, *optional*, defaults to *[0.5, 0.5, 0.5]*) : Image mean to use for normalization. Only has an effect if *do_normalize* is set to *True*.

image_std (*Union[float, list[float], tuple[float, ...]]*, *kwargs*, *optional*, defaults to *[0.5, 0.5, 0.5]*) : Image standard deviation to use for normalization. Only has an effect if *do_normalize* is set to *True*.

do_pad (*bool*, *kwargs*, *optional*) : Whether to pad the image. Padding is done either to the largest size in the batch or to a fixed square size per image. The exact padding strategy depends on the model.

pad_size (*Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]*, *kwargs*) : The size in *&amp;lcub;"height": int, "width" int}* to pad the images to. Must be larger than any image size provided for preprocessing. If *pad_size* is not provided, images will be padded to the largest height and width in the batch. Applied only when *do_pad=True.*

do_center_crop (*bool*, *kwargs*, *optional*) : Whether to center crop the image.

data_format (*Union[str, ~image_utils.ChannelDimension]*, *kwargs*, *optional*) : Only *ChannelDimension.FIRST* is supported. Added for compatibility with slow processors.

input_data_format (*Union[str, ~image_utils.ChannelDimension]*, *kwargs*, *optional*) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - *"channels_first"* or *ChannelDimension.FIRST*: image in (num_channels, height, width) format. - *"channels_last"* or *ChannelDimension.LAST*: image in (height, width, num_channels) format. - *"none"* or *ChannelDimension.NONE*: image in (height, width) format.

device (*Annotated[Union[str, torch.device, NoneType], None]*, *kwargs*) : The device to process the videos on. If unset, the device is inferred from the input videos.

return_tensors (*Annotated[str | ~utils.generic.TensorType | None, None]*, *kwargs*) : Returns stacked tensors if set to *'pt'*, otherwise returns a list of tensors.

disable_grouping (*bool*, *kwargs*, *optional*) : Whether to disable grouping of images by size to process them individually and not in batches. If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157

image_seq_length (*int*, *kwargs*, *optional*) : The number of image tokens to be used for each image in the input. Added for backward compatibility but this should be set as a processor attribute in future models.

max_slice_nums (*int*, *kwargs*, *optional*, defaults to 9) : Maximum number of slices when splitting a high-resolution image.

scale_resolution (*int*, *kwargs*, *optional*, defaults to 448) : Target resolution for individual slices.

patch_size (*int*, *kwargs*, *optional*, defaults to 14) : Spatial patch size of the vision encoder.

slice_mode (*bool*, *kwargs*, *optional*, defaults to *True*) : Whether to split images into multiple slices for higher resolution.

downsample_mode (*str*, *kwargs*, *optional*, defaults to *"4x"*) : Visual token downsampling mode. *"4x"* keeps 4x more tokens than *"16x"*, which applies the full merge. Overriding it per call is supported.

use_image_id (*bool*, *kwargs*, *optional*, defaults to *True*) : Whether to prepend an image-id tag (`<image_id>N</image_id>`) before each image placeholder. Consumed by the Processor for placeholder generation, not by the image processing pipeline itself.

small_image_upsize (*str*, *kwargs*, *optional*, defaults to *"always"*) : How images and slices no larger than *scale_resolution* squared are resized. *"always"* upscales them to *scale_resolution*, *"never"* keeps their native resolution aligned to the patch grid, and *"adaptive"* picks per image and per slice from glyph size and density (requires OpenCV, see [*adaptive_upsize_resolution*]).

Constructs a MiniCPMOcrImageProcessor image processor.

#### preprocess[[transformers.MiniCPMOcrImageProcessor.preprocess]]

```python
preprocess(images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']], **kwargs: *<class 'transformers.models.minicpm_ocr.image_processing_minicpm_ocr.MiniCPMOcrImageProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/minicpm_ocr/image_processing_minicpm_ocr.py#L246)

**Parameters:**

images (*Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]*) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set *do_rescale=False*.

do_convert_rgb (*bool*, *kwargs*, *optional*) : Whether to convert the image to RGB.

do_resize (*bool*, *kwargs*, *optional*) : Whether to resize the image.

size (*Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]*, *kwargs*) : Describes the maximum input dimensions to the model.

default_to_square (*bool*, *kwargs*, *optional*) : Whether to default to a square image when resizing, if size is an int.

crop_size (*Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]*, *kwargs*) : Size of the output image after applying *center_crop*.

resample (*Annotated[Union[int, PILImageResampling, NoneType], None]*, *kwargs*) : Resampling filter to use if resizing the image. This can be one of the enum *PILImageResampling*. Only has an effect if *do_resize* is set to *True*.

do_rescale (*bool*, *kwargs*, *optional*) : Whether to rescale the image.

rescale_factor (*float*, *kwargs*, *optional*) : Rescale factor to rescale the image by if *do_rescale* is set to *True*.

do_normalize (*bool*, *kwargs*, *optional*) : Whether to normalize the image.

image_mean (*Union[float, list[float], tuple[float, ...]]*, *kwargs*, *optional*) : Image mean to use for normalization. Only has an effect if *do_normalize* is set to *True*.

image_std (*Union[float, list[float], tuple[float, ...]]*, *kwargs*, *optional*) : Image standard deviation to use for normalization. Only has an effect if *do_normalize* is set to *True*.

do_pad (*bool*, *kwargs*, *optional*) : Whether to pad the image. Padding is done either to the largest size in the batch or to a fixed square size per image. The exact padding strategy depends on the model.

pad_size (*Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]*, *kwargs*) : The size in *&amp;lcub;"height": int, "width" int}* to pad the images to. Must be larger than any image size provided for preprocessing. If *pad_size* is not provided, images will be padded to the largest height and width in the batch. Applied only when *do_pad=True.*

do_center_crop (*bool*, *kwargs*, *optional*) : Whether to center crop the image.

data_format (*Union[str, ~image_utils.ChannelDimension]*, *kwargs*, *optional*) : Only *ChannelDimension.FIRST* is supported. Added for compatibility with slow processors.

input_data_format (*Union[str, ~image_utils.ChannelDimension]*, *kwargs*, *optional*) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - *"channels_first"* or *ChannelDimension.FIRST*: image in (num_channels, height, width) format. - *"channels_last"* or *ChannelDimension.LAST*: image in (height, width, num_channels) format. - *"none"* or *ChannelDimension.NONE*: image in (height, width) format.

device (*Annotated[Union[str, torch.device, NoneType], None]*, *kwargs*) : The device to process the videos on. If unset, the device is inferred from the input videos.

return_tensors (*Annotated[str | ~utils.generic.TensorType | None, None]*, *kwargs*) : Returns stacked tensors if set to *'pt'*, otherwise returns a list of tensors.

disable_grouping (*bool*, *kwargs*, *optional*) : Whether to disable grouping of images by size to process them individually and not in batches. If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157

image_seq_length (*int*, *kwargs*, *optional*) : The number of image tokens to be used for each image in the input. Added for backward compatibility but this should be set as a processor attribute in future models.

max_slice_nums (*int*, *kwargs*, *optional*, defaults to 9) : Maximum number of slices when splitting a high-resolution image.

scale_resolution (*int*, *kwargs*, *optional*, defaults to 448) : Target resolution for individual slices.

patch_size (*int*, *kwargs*, *optional*, defaults to 14) : Spatial patch size of the vision encoder.

slice_mode (*bool*, *kwargs*, *optional*, defaults to *True*) : Whether to split images into multiple slices for higher resolution.

downsample_mode (*str*, *kwargs*, *optional*, defaults to *"4x"*) : Visual token downsampling mode. *"4x"* keeps 4x more tokens than *"16x"*, which applies the full merge. Overriding it per call is supported.

use_image_id (*bool*, *kwargs*, *optional*, defaults to *True*) : Whether to prepend an image-id tag (`<image_id>N</image_id>`) before each image placeholder. Consumed by the Processor for placeholder generation, not by the image processing pipeline itself.

small_image_upsize (*str*, *kwargs*, *optional*, defaults to *"always"*) : How images and slices no larger than *scale_resolution* squared are resized. *"always"* upscales them to *scale_resolution*, *"never"* keeps their native resolution aligned to the patch grid, and *"adaptive"* picks per image and per slice from glyph size and density (requires OpenCV, see [*adaptive_upsize_resolution*]).

**Returns:** `*~feature_extraction_utils.BatchFeature*`

- **data** (*dict*, *optional*) -- Dictionary of lists/arrays/tensors returned by the __call__/pad methods ('input_values', 'attention_mask',
  etc.).
- **tensor_type** (*Union[None, str, TensorType]*, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.
- **skip_tensor_conversion** (*list[str]* or *set[str]*, *optional*) -- List or set of keys that should NOT be converted to tensors, even when *tensor_type* is specified.

## MiniCPMOcrVideoProcessor[[transformers.MiniCPMOcrVideoProcessor]]

#### transformers.MiniCPMOcrVideoProcessor[[transformers.MiniCPMOcrVideoProcessor]]

```python
transformers.MiniCPMOcrVideoProcessor(**kwargs: *<class 'transformers.models.minicpm_ocr.video_processing_minicpm_ocr.MiniCPMOcrVideoProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/minicpm_ocr/video_processing_minicpm_ocr.py#L77)

**Parameters:**

do_convert_rgb (*bool*, *kwargs*, *optional*, defaults to *True*) : Whether to convert the image to RGB.

do_resize (*bool*, *kwargs*, *optional*, defaults to *True*) : Whether to resize the image.

size (*Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]*, *kwargs*) : Describes the maximum input dimensions to the model.

default_to_square (*bool*, *kwargs*, *optional*, defaults to *True*) : Whether to default to a square image when resizing, if size is an int.

resample (*Annotated[Union[int, PILImageResampling, NoneType], None]*, *kwargs*, defaults to *3*) : Resampling filter to use if resizing the image. This can be one of the enum *PILImageResampling*. Only has an effect if *do_resize* is set to *True*.

do_rescale (*bool*, *kwargs*, *optional*, defaults to *True*) : Whether to rescale the image.

rescale_factor (*float*, *kwargs*, *optional*, defaults to *0.00392156862745098*) : Rescale factor to rescale the image by if *do_rescale* is set to *True*.

do_normalize (*bool*, *kwargs*, *optional*, defaults to *True*) : Whether to normalize the image.

image_mean (*Union[float, list[float], tuple[float, ...]]*, *kwargs*, *optional*, defaults to *[0.5, 0.5, 0.5]*) : Image mean to use for normalization. Only has an effect if *do_normalize* is set to *True*.

image_std (*Union[float, list[float], tuple[float, ...]]*, *kwargs*, *optional*, defaults to *[0.5, 0.5, 0.5]*) : Image standard deviation to use for normalization. Only has an effect if *do_normalize* is set to *True*.

do_center_crop (*bool*, *kwargs*, *optional*) : Whether to center crop the image.

do_pad (*bool*, *kwargs*, *optional*) : Whether to pad the image. Padding is done either to the largest size in the batch or to a fixed square size per image. The exact padding strategy depends on the model.

crop_size (*Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]*, *kwargs*) : Size of the output image after applying *center_crop*.

data_format (*Union[str, ~image_utils.ChannelDimension]*, *kwargs*, *optional*) : Only *ChannelDimension.FIRST* is supported. Added for compatibility with slow processors.

input_data_format (*Union[str, ~image_utils.ChannelDimension]*, *kwargs*, *optional*) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - *"channels_first"* or *ChannelDimension.FIRST*: image in (num_channels, height, width) format. - *"channels_last"* or *ChannelDimension.LAST*: image in (height, width, num_channels) format. - *"none"* or *ChannelDimension.NONE*: image in (height, width) format.

device (*Annotated[Union[str, torch.device, NoneType], None]*, *kwargs*) : The device to process the videos on. If unset, the device is inferred from the input videos.

do_sample_frames (*bool*, *kwargs*, *optional*, defaults to *True*) : Whether to sample frames from the video before processing or to process the whole video.

video_metadata (*Annotated[~video_utils.VideoMetadata | dict | list[dict | ~video_utils.VideoMetadata] | list[list[dict | ~video_utils.VideoMetadata]] | None, None]*, *kwargs*) : Metadata of the video containing information about total duration, fps and total number of frames. It will be used to sample frames from video or compute timestamps. Don't pass any metadata unless you are trying to decode the video manually before processing

fps (*Annotated[int | float | None, None]*, *kwargs*) : Target frames to sample per second when *do_sample_frames=True*.

num_frames (*Annotated[int | None, None]*, *kwargs*) : Maximum number of frames to sample when *do_sample_frames=True*.

return_metadata (*bool*, *kwargs*, *optional*, defaults to *False*) : Whether to return video metadata or not. Video metadats is an object containing info about video duration, fps, decoding backend, etc.

return_tensors (*Annotated[str | ~utils.generic.TensorType | None, None]*, *kwargs*) : If set, will return tensors of a particular framework. Acceptable values are:  - *'pt'*: Return PyTorch *torch.Tensor* objects. - *'np'*: Return NumPy *np.ndarray* objects.

max_num_frames (*int*, *kwargs*, *optional*, defaults to 128) : Maximum number of main frames to sample per video.

stack_frames (*int*, *kwargs*, *optional*, defaults to 1) : Sub-frames per second to stack.  `1` disables stacking.

max_slice_nums (*int*, *kwargs*, *optional*, defaults to 9) : Maximum number of slices when splitting a high-resolution image.

scale_resolution (*int*, *kwargs*, *optional*, defaults to 448) : Target resolution for individual slices.

patch_size (*int*, *kwargs*, *optional*, defaults to 14) : Spatial patch size of the vision encoder.

slice_mode (*bool*, *kwargs*, *optional*, defaults to *True*) : Whether to split images into multiple slices for higher resolution.

downsample_mode (*str*, *kwargs*, *optional*, defaults to *"4x"*) : Visual token downsampling mode. *"4x"* keeps 4x more tokens than *"16x"*, which applies the full merge.

use_image_id (*bool*, *kwargs*, *optional*, defaults to *True*) : Whether to prepend an image-id tag (`<image_id>N</image_id>`) before each image placeholder. Consumed by the Processor for placeholder generation, not by the image processing pipeline itself.

- ****kwargs** ([*ProcessingKwargs*], *optional*) : Additional processing options for each modality (text, images, videos, audio). Model-specific parameters are listed above; see the TypedDict class for the complete list of supported arguments.

Constructs a MiniCPMOcrVideoProcessor video processor.

#### preprocess[[transformers.MiniCPMOcrVideoProcessor.preprocess]]

```python
preprocess(videos: typing.Union[list['PIL.Image.Image'], numpy.ndarray, ForwardRef('torch.Tensor'), list[numpy.ndarray], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list[numpy.ndarray]], list[list['torch.Tensor']], transformers.video_utils.URL, list[transformers.video_utils.URL], list[list[transformers.video_utils.URL]], transformers.video_utils.Path, list[transformers.video_utils.Path], list[list[transformers.video_utils.Path]]], **kwargs: *<class 'transformers.processing_utils.VideosKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/minicpm_ocr/video_processing_minicpm_ocr.py#L292)

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

**Returns:** `~feature_extraction_utils.BatchFeature`

- **data** (`dict`, *optional*) -- Dictionary of lists/arrays/tensors returned by the __call__/pad methods ('input_values', 'attention_mask',
  etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.
- **skip_tensor_conversion** (`list[str]` or `set[str]`, *optional*) -- List or set of keys that should NOT be converted to tensors, even when `tensor_type` is specified.

### Longformer
https://huggingface.co/docs/transformers/v5.21.1/model_doc/longformer.md
