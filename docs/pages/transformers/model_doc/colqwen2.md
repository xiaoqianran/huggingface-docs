# ColQwen2

[ColQwen2](https://huggingface.co/papers/2407.01449) is a variant of the [ColPali](./colpali) model designed to retrieve documents by analyzing their visual features. Unlike traditional systems that rely heavily on text extraction and OCR, ColQwen2 treats each page as an image. It uses the [Qwen2-VL](./qwen2_vl) backbone to capture not only text, but also the layout, tables, charts, and other visual elements to create detailed multi-vector embeddings that can be used for retrieval by computing pairwise late interaction similarity scores. This offers a more comprehensive understanding of documents and enables more efficient and accurate retrieval.

This model was contributed by [@tonywu71](https://huggingface.co/tonywu71) (ILLUIN Technology) and [@yonigozlan](https://huggingface.co/yonigozlan) (HuggingFace).

You can find all the original ColPali checkpoints under Vidore's [Hf-native ColVision Models](https://huggingface.co/collections/vidore/hf-native-colvision-models-6755d68fc60a8553acaa96f7) collection.

> [!TIP]
> Click on the ColQwen2 models in the right sidebar for more examples of how to use ColQwen2 for image retrieval.

```python
import requests
import torch
from PIL import Image

from transformers import ColQwen2ForRetrieval, ColQwen2Processor
from transformers.utils.import_utils import is_flash_attn_2_available

# Load the model and the processor
model_name = "vidore/colqwen2-v1.0-hf"

model = ColQwen2ForRetrieval.from_pretrained(
    model_name,
    device_map="auto",  # "cpu", "cuda", "xpu" or "mps" for Apple Silicon
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa",
)
processor = ColQwen2Processor.from_pretrained(model_name)

# The document page screenshots from your corpus
url1 = "https://upload.wikimedia.org/wikipedia/commons/8/89/US-original-Declaration-1776.jpg"
url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Romeoandjuliet1597.jpg/500px-Romeoandjuliet1597.jpg"

images = [
    Image.open(requests.get(url1, stream=True).raw),
    Image.open(requests.get(url2, stream=True).raw),
]

# The queries you want to retrieve documents for
queries = [
    "When was the United States Declaration of Independence proclaimed?",
    "Who printed the edition of Romeo and Juliet?",
]

# Process the inputs
inputs_images = processor(images=images).to(model.device)
inputs_text = processor(text=queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**inputs_images).embeddings
    query_embeddings = model(**inputs_text).embeddings

# Score the queries against the images
scores = processor.score_retrieval(query_embeddings, image_embeddings)

print("Retrieval scores (query x image):")
print(scores)
```

If you have issue with loading the images with PIL, you can use the following code to create dummy images:

```python
images = [
    Image.new("RGB", (128, 128), color="white"),
    Image.new("RGB", (64, 32), color="black"),
]
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to quantize the weights to int4.

```python
import requests
import torch
from PIL import Image

from transformers import BitsAndBytesConfig, ColQwen2ForRetrieval, ColQwen2Processor

model_name = "vidore/colqwen2-v1.0-hf"

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = ColQwen2ForRetrieval.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
).eval()

processor = ColQwen2Processor.from_pretrained(model_name)

url1 = "https://upload.wikimedia.org/wikipedia/commons/8/89/US-original-Declaration-1776.jpg"
url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Romeoandjuliet1597.jpg/500px-Romeoandjuliet1597.jpg"

images = [
    Image.open(requests.get(url1, stream=True).raw),
    Image.open(requests.get(url2, stream=True).raw),
]

queries = [
    "When was the United States Declaration of Independence proclaimed?",
    "Who printed the edition of Romeo and Juliet?",
]

# Process the inputs
inputs_images = processor(images=images, return_tensors="pt").to(model.device)
inputs_text = processor(text=queries, return_tensors="pt").to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**inputs_images).embeddings
    query_embeddings = model(**inputs_text).embeddings

# Score the queries against the images
scores = processor.score_retrieval(query_embeddings, image_embeddings)

print("Retrieval scores (query x image):")
print(scores)
```

You can also use checkpoints for `ColQwen2.5` that are **compatible with the ColQwen2 architecture**. This version of the model uses [Qwen2_5_VL](./qwen2_5_vl) as the backbone.

```python
from transformers import ColQwen2ForRetrieval, ColQwen2Processor
from transformers.utils.import_utils import is_flash_attn_2_available

model_name = "Sahil-Kabir/colqwen2.5-v0.2-hf" # An existing compatible checkpoint

model = ColQwen2ForRetrieval.from_pretrained(
    model_name,
    device_map="auto",
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa"
)
processor = ColQwen2Processor.from_pretrained(model_name)
```

## Notes

- `score_retrieval()` returns a 2D tensor where the first dimension is the number of queries and the second dimension is the number of images. A higher score indicates more similarity between the query and image.
- Unlike ColPali, ColQwen2 supports arbitrary image resolutions and aspect ratios, which means images are not resized into fixed-size squares. This preserves more of the original input signal.
- Larger input images generate longer multi-vector embeddings, allowing users to adjust image resolution to balance performance and memory usage.

## ColQwen2Config[[transformers.ColQwen2Config]]

#### transformers.ColQwen2Config[[transformers.ColQwen2Config]]

```python
transformers.ColQwen2Config(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vlm_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, embedding_dim: int = 128, initializer_range: float = 0.02)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/colqwen2/configuration_colqwen2.py#L28)

**Parameters:**

vlm_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the vision-language backbone.

embedding_dim (`int`, *optional*, defaults to `128`) : Dimensionality of the embeddings and hidden states.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a Colqwen2Model. It is used to instantiate a Colqwen2
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [vidore/colqwen2-v1.0-hf](https://huggingface.co/vidore/colqwen2-v1.0-hf)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.20.2/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.20.2/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
from transformers.models.colqwen2 import ColQwen2Config, ColQwen2ForRetrieval

config = ColQwen2Config()
model = ColQwen2ForRetrieval(config)
```

## ColQwen2Processor[[transformers.ColQwen2Processor]]

#### transformers.ColQwen2Processor[[transformers.ColQwen2Processor]]

```python
transformers.ColQwen2Processor(image_processor = None, tokenizer = None, chat_template = None, visual_prompt_prefix: str | None = None, query_prefix: str | None = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/colqwen2/processing_colqwen2.py#L51)

**Parameters:**

image_processor (`Qwen2VLImageProcessor`) : The image processor is a required input.

tokenizer (`Qwen2Tokenizer`) : The tokenizer is a required input.

chat_template (`str`) : A Jinja template to convert lists of messages in a chat into a tokenizable string.

visual_prompt_prefix (`str`, *optional*, defaults to `"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>"`) : A string that gets tokenized and prepended to the image tokens.

query_prefix (`str`, *optional*, defaults to `"Query : "`): A prefix to be used for the query.

Constructs a ColQwen2Processor which wraps a image processor and a tokenizer into a single processor.

[ColQwen2Processor](/docs/transformers/v5.20.2/en/model_doc/colqwen2#transformers.ColQwen2Processor) offers all the functionalities of [Qwen2VLImageProcessor](/docs/transformers/v5.20.2/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor) and [Qwen2Tokenizer](/docs/transformers/v5.20.2/en/model_doc/qwen2#transformers.Qwen2Tokenizer). See the
[~Qwen2VLImageProcessor](/docs/transformers/v5.20.2/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor) and [~Qwen2Tokenizer](/docs/transformers/v5.20.2/en/model_doc/qwen2#transformers.Qwen2Tokenizer) for more information.

#### __call__[[transformers.ColQwen2Processor.__call__]]

```python
__call__(images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None, text: str | list[str] | list[list[str]] = None, **kwargs: *<class 'transformers.models.colqwen2.processing_colqwen2.ColQwen2ProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/colqwen2/processing_colqwen2.py#L77)

**Parameters:**

images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]`, *optional*) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

text (`Union[str, list[str], list[list[str]]]`, *optional*) : The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings (pretokenized string). If you pass a pretokenized input, set `is_split_into_words=True` to avoid ambiguity with batched inputs.

return_tensors (`str` or [TensorType](/docs/transformers/v5.20.2/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

- ****kwargs** ([ProcessingKwargs](/docs/transformers/v5.20.2/en/main_classes/processors#transformers.ProcessingKwargs), *optional*) : Additional processing options for each modality (text, images, videos, audio). Model-specific parameters are listed above; see the TypedDict class for the complete list of supported arguments.

**Returns:** [BatchFeature](/docs/transformers/v5.20.2/en/main_classes/feature_extractor#transformers.BatchFeature)

A [BatchFeature](/docs/transformers/v5.20.2/en/main_classes/feature_extractor#transformers.BatchFeature) with the following fields:

- **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`. If `suffix`
  is provided, the `input_ids` will also contain the suffix input ids.
- **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
  `None`).
- **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
- **labels** -- Labels compatible with training if `suffix` is not None

## ColQwen2ForRetrieval[[transformers.ColQwen2ForRetrieval]]

#### transformers.ColQwen2ForRetrieval[[transformers.ColQwen2ForRetrieval]]

```python
transformers.ColQwen2ForRetrieval(config: ColQwen2Config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/colqwen2/modeling_colqwen2.py#L92)

**Parameters:**

config ([ColQwen2Config](/docs/transformers/v5.20.2/en/model_doc/colqwen2#transformers.ColQwen2Config)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.20.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Following the ColPali approach, ColQwen2 leverages VLMs to construct efficient multi-vector embeddings directly
from document images (“screenshots”) for document retrieval. The model is trained to maximize the similarity
between these document embeddings and the corresponding query embeddings, using the late interaction method
introduced in ColBERT.

Using ColQwen2 removes the need for potentially complex and brittle layout recognition and OCR pipelines with
a single model that can take into account both the textual and visual content (layout, charts, ...) of a document.

ColQwen2 is part of the ColVision model family, which was introduced with ColPali in the following paper:
[*ColPali: Efficient Document Retrieval with Vision Language Models*](https://huggingface.co/papers/2407.01449).

This model inherits from [PreTrainedModel](/docs/transformers/v5.20.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.ColQwen2ForRetrieval.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, labels: typing.Optional[torch.LongTensor] = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, output_attentions: bool | None = None, output_hidden_states: bool | None = None, return_dict: bool | None = None, pixel_values: typing.Optional[torch.Tensor] = None, image_grid_thw: typing.Optional[torch.LongTensor] = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/colqwen2/modeling_colqwen2.py#L106)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.20.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.20.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.20.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.20.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.20.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

output_attentions (`bool`, *optional*) : Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.

output_hidden_states (`bool`, *optional*) : Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.

return_dict (`bool`, *optional*) : Whether or not to return a [ModelOutput](/docs/transformers/v5.20.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [Qwen2VLImageProcessor](/docs/transformers/v5.20.2/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor). See [Qwen2VLImageProcessor.__call__()](/docs/transformers/v5.20.2/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([ColQwen2Processor](/docs/transformers/v5.20.2/en/model_doc/colqwen2#transformers.ColQwen2Processor) uses [Qwen2VLImageProcessor](/docs/transformers/v5.20.2/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor) for processing images).

image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) : The temporal, height and width of feature shape of each image in LLM.

**Returns:** `ColQwen2ForRetrievalOutput` or `tuple(torch.FloatTensor)`

A `ColQwen2ForRetrievalOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ColQwen2Config](/docs/transformers/v5.20.2/en/model_doc/colqwen2#transformers.ColQwen2Config)) and inputs.

The [ColQwen2ForRetrieval](/docs/transformers/v5.20.2/en/model_doc/colqwen2#transformers.ColQwen2ForRetrieval) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **embeddings** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- The embeddings of the model.
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

Example:

```python
```

### Overview
https://huggingface.co/docs/transformers/v5.20.2/model_doc/kimi_linear.md

## Overview

Kimi Linear is a hybrid linear attention architecture from Moonshot AI, introduced in
[Kimi Linear: An Expressive, Efficient Attention Architecture](https://huggingface.co/papers/2510.26692).

At its core is **Kimi Delta Attention (KDA)**, a refinement of [Gated DeltaNet](https://huggingface.co/papers/2412.06464)
that gives each key channel its own forget gate, so the recurrent state decays per channel instead of per head. KDA is
used in most layers; every fourth layer keeps a full-attention block that reuses DeepSeek-V3's Multi-head Latent
Attention (MLA), and the feed-forward blocks are DeepSeek-V3-style MoE with a shared expert.

The abstract from the paper is the following:

*We introduce Kimi Linear, a hybrid linear attention architecture that, for the first time, outperforms full attention
under fair comparisons across various scenarios -- including short-context, long-context, and reinforcement learning
(RL) scaling regimes. At its core lies Kimi Delta Attention (KDA), an expressive linear attention module that extends
Gated DeltaNet with a finer-grained gating mechanism.*

Two things are worth knowing when reading the modeling code:

- **The model is NoPE.** Every released checkpoint sets `mla_use_nope=True`, so no rotary embedding is applied
  anywhere: the KDA layers encode position through their recurrence, and the full-attention layers are left without
  positional encoding. The `qk_rope_head_dim` slice still exists in the projections, it is simply never rotated.
- **The layer pattern comes from the checkpoint.** `linear_attn_config` lists `kda_layers` / `full_attn_layers` with
  1-based indices; the config converts them into the standard `layer_types` list.

This model was contributed by [Remi Ouazan](https://huggingface.co/ror).
The original code can be found [here](https://github.com/MoonshotAI/Kimi-Linear).

## Usage examples

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "moonshotai/Kimi-Linear-48B-A3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [{"role": "user", "content": "Tell me about the french revolution."}]
model_inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

generated_ids = model.generate(**model_inputs, max_new_tokens=128)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]

print(tokenizer.decode(output_ids, skip_special_tokens=True))
```

The KDA layers run on a pure PyTorch implementation by default. Installing
[`kernels`](https://github.com/huggingface/kernels) (`pip install -U kernels`) and passing `use_kernels=True`
in `from_pretrained` makes them dispatch to custom kernels instead, which is considerably faster for long sequences.

## Notes

- Use left padding for batched generation. See [Padding side](../llm_tutorial#padding-side).

## KimiLinearConfig[[transformers.KimiLinearConfig]]

#### transformers.KimiLinearConfig[[transformers.KimiLinearConfig]]

```python
transformers.KimiLinearConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vocab_size: int = 163840, hidden_size: int = 2304, intermediate_size: int = 9216, moe_intermediate_size: int = 1024, num_hidden_layers: int = 27, num_attention_heads: int = 32, num_key_value_heads: int | None = 32, n_shared_experts: int = 1, output_router_logits: bool = False, routed_scaling_factor: float = 2.446, kv_lora_rank: int = 512, q_lora_rank: int | None = None, qk_rope_head_dim: int = 64, v_head_dim: int | None = 128, qk_nope_head_dim: int = 128, n_group: int = 1, topk_group: int | None = 1, num_experts_per_tok: int | None = 8, norm_topk_prob: bool = True, hidden_act: str = 'silu', max_position_embeddings: int = 1048576, initializer_range: float = 0.02, rms_norm_eps: float = 1e-05, use_cache: bool = True, pad_token_id: int | None = 163839, bos_token_id: int | None = 163584, eos_token_id: int | list[int] | None = 163586, pretraining_tp: int | None = 1, tie_word_embeddings: bool = False, attention_bias: bool = False, attention_dropout: float | int | None = 0.0, num_local_experts: int = 256, mlp_layer_types: list[str] | None = None, layer_types: list[str] | None = None, linear_head_dim: int = 128, linear_num_heads: int = 32, linear_conv_kernel_dim: int = 4)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/kimi_linear/configuration_kimi_linear.py#L30)

**Parameters:**

vocab_size (`int`, *optional*, defaults to `163840`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

hidden_size (`int`, *optional*, defaults to `2304`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `9216`) : Dimension of the MLP representations.

moe_intermediate_size (`int`, *optional*, defaults to `1024`) : Intermediate size of the routed expert MLPs.

num_hidden_layers (`int`, *optional*, defaults to `27`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `32`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*, defaults to `32`) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

n_shared_experts (`int`, *optional*, defaults to `1`) : Number of shared experts.

output_router_logits (`bool`, *optional*, defaults to `False`) : Whether or not the router logits should be returned by the model. Enabling this will also allow the model to output the auxiliary loss, including load balancing loss and router z-loss.

routed_scaling_factor (`float`, *optional*, defaults to `2.446`) : Scaling factor or routed experts.

kv_lora_rank (`int`, *optional*, defaults to `512`) : Rank of the LoRA matrices for key and value projections.

q_lora_rank (`int`, *optional*) : Rank of the LoRA matrices for query projections.

qk_rope_head_dim (`int`, *optional*, defaults to `64`) : Dimension of the query/key heads that use rotary position embeddings.

v_head_dim (`int`, *optional*, defaults to `128`) : Dimension of the value heads.

qk_nope_head_dim (`int`, *optional*, defaults to `128`) : Dimension of the query/key heads that don't use rotary position embeddings.

n_group (`int`, *optional*, defaults to 8) : Number of groups for routed experts.

topk_group (`int`, *optional*, defaults to `1`) : Number of selected groups for each token (for each token, ensuring the selected experts is only within `topk_group` groups).

num_experts_per_tok (`int`, *optional*, defaults to `8`) : Number of experts to route each token to. This is the top-k value for the token-choice routing.

norm_topk_prob (`bool`, *optional*, defaults to `True`) : Whether to normalize the weights of the routed experts. 

hidden_act (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

max_position_embeddings (`int`, *optional*, defaults to `1048576`) : The maximum sequence length that this model might ever be used with.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rms_norm_eps (`float`, *optional*, defaults to `1e-05`) : The epsilon used by the rms normalization layers.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

pad_token_id (`int`, *optional*, defaults to `163839`) : Token id used for padding in the vocabulary.

bos_token_id (`int`, *optional*, defaults to `163584`) : Token id used for beginning-of-stream in the vocabulary.

eos_token_id (`Union[int, list[int]]`, *optional*, defaults to `163586`) : Token id used for end-of-stream in the vocabulary.

pretraining_tp (`int`, *optional*, defaults to `1`) : Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this document](https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism) to understand more about it. This value is necessary to ensure exact reproducibility of the pretraining results. Please refer to [this issue](https://github.com/pytorch/pytorch/issues/76232).

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

attention_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

num_local_experts (`int`, *optional*, defaults to `256`) : Number of local experts on each device. `num_experts` should be divisible by `num_local_experts`.

mlp_layer_types (`list[str]`, *optional*) : List of layer types for the MLP or MoE layers. Defaults to None.

layer_types (`list[str]`, *optional*) : A list that explicitly maps each layer index with its layer type. If not provided, it will be automatically generated based on config values.

linear_head_dim (`int`, *optional*) : Dimension of each head in linear attention layers. Defaults to 128.

linear_num_heads (`int`, *optional*) : Number of heads for the linear attention layers. Defaults to 32.

linear_conv_kernel_dim (`int`, *optional*, defaults to 4) : Kernel size for the short convolution applied to queries, keys, and values in linear attention layers.

This is the configuration class to store the configuration of a KimiLinearModel. It is used to instantiate a Kimi Linear
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [moonshotai/Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.20.2/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.20.2/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## KimiLinearModel[[transformers.KimiLinearModel]]

#### transformers.KimiLinearModel[[transformers.KimiLinearModel]]

```python
transformers.KimiLinearModel(config: KimiLinearConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/kimi_linear/modeling_kimi_linear.py#L876)

**Parameters:**

config ([KimiLinearConfig](/docs/transformers/v5.20.2/en/model_doc/kimi_linear#transformers.KimiLinearConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.20.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Kimi Linear Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.20.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.KimiLinearModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/kimi_linear/modeling_kimi_linear.py#L888)

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
elements depending on the configuration ([KimiLinearConfig](/docs/transformers/v5.20.2/en/model_doc/kimi_linear#transformers.KimiLinearConfig)) and inputs.

The [KimiLinearModel](/docs/transformers/v5.20.2/en/model_doc/kimi_linear#transformers.KimiLinearModel) forward method, overrides the `__call__` special method.

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

## KimiLinearForCausalLM[[transformers.KimiLinearForCausalLM]]

#### transformers.KimiLinearForCausalLM[[transformers.KimiLinearForCausalLM]]

```python
transformers.KimiLinearForCausalLM(config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/kimi_linear/modeling_kimi_linear.py#L951)

**Parameters:**

config ([KimiLinearForCausalLM](/docs/transformers/v5.20.2/en/model_doc/kimi_linear#transformers.KimiLinearForCausalLM)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.20.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Kimi Linear Model for causal language modeling.

This model inherits from [PreTrainedModel](/docs/transformers/v5.20.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.KimiLinearForCausalLM.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/kimi_linear/modeling_kimi_linear.py#L966)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.20.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.20.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.20.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.20.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.20.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

**Returns:** `MoeCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `MoeCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([KimiLinearConfig](/docs/transformers/v5.20.2/en/model_doc/kimi_linear#transformers.KimiLinearConfig)) and inputs.

The [KimiLinearForCausalLM](/docs/transformers/v5.20.2/en/model_doc/kimi_linear#transformers.KimiLinearForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **aux_loss** (`torch.FloatTensor`, *optional*, returned when `output_router_logits=True` is passed or when `config.output_router_logits=True`, and the model trains its router with a load-balancing loss) -- Load-balancing auxiliary loss for the sparse modules. Models that balance their experts with a router bias
  instead (DeepSeek-V3 and the architectures derived from it) return `None` here while still returning
  `router_logits`.
- **router_logits** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_logits=True` is passed or when `config.output_router_logits=True`) -- Tuple of `torch.FloatTensor` (one for each sparse layer) of shape `(batch_size * sequence_length, num_experts)`.

  Raw router logits computed by the MoE routers. They can be used to compute a load-balancing loss or to
  inspect how tokens are dispatched to experts.
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

Example:

```python
>>> from transformers import AutoTokenizer, KimiLinearForCausalLM

>>> model = KimiLinearForCausalLM.from_pretrained("meta-kimi_linear/KimiLinear-2-7b-hf")
>>> tokenizer = AutoTokenizer.from_pretrained("meta-kimi_linear/KimiLinear-2-7b-hf")

>>> prompt = "Hey, are you conscious? Can you talk to me?"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
```

### RADIO
https://huggingface.co/docs/transformers/v5.20.2/model_doc/radio.md
