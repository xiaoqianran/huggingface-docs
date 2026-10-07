# HyperCLOVAX Vision V2

HyperCLOVAX Vision V2 is a multimodal vision-language model developed by NAVER. It combines the [HyperClovaX](./hyperclovax) language model backbone with a [Qwen2.5-VL](./qwen2_5_vl) vision encoder. The model supports text, image, and video inputs and is capable of chain-of-thought reasoning via built-in thinking tokens (`<think>...</think>`).

You can find the original HyperCLOVAX-SEED-Think-32B checkpoint on the [naver-hyperclovax/HyperCLOVAX-SEED-Think-32B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-32B) page.

The example below demonstrates how to generate text based on an image with [AutoModelForImageTextToText](/docs/transformers/v5.19.0/en/model_doc/auto#transformers.AutoModelForImageTextToText).

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained(
    "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained(
    "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "url": "/path/to/video.mp4",
            },
            {"type": "text", "text": "Describe this video."},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to load the model in 4-bit.

```python
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForImageTextToText.from_pretrained(
    "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B",
    device_map="auto",
    quantization_config=quantization_config,
)
processor = AutoProcessor.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")
```

## Notes

- The model supports chain-of-thought reasoning. By default, the generation prompt prepends an empty `<think>\n\n</think>` block. To generate an explicit reasoning trace inside `<think>...</think>` tags, pass `thinking=True` to `apply_chat_template` (image/text inputs only):

    ```python
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        thinking=True,
    ).to(model.device)
    ```

- The model supports multi-turn conversations with mixed media. Images and videos can appear across multiple turns.

    ```python
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://example.com/image1.jpg"},
                {"type": "text", "text": "What do you see in this image?"},
            ],
        },
        {
            "role": "assistant",
            "content": "I see a cat sitting on a couch.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://example.com/image2.jpg"},
                {"type": "text", "text": "How does this compare to the first image?"},
            ],
        },
    ]
    ```

- The model supports function/tool calling. Pass tools using the `tools` parameter in `apply_chat_template`:

    ```python
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [
        {"role": "user", "content": "What is the weather in Seoul?"}
    ]

    inputs = processor.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    ```

## HyperCLOVAXVisionV2Config[[transformers.HyperCLOVAXVisionV2Config]]

#### transformers.HyperCLOVAXVisionV2Config[[transformers.HyperCLOVAXVisionV2Config]]

```python
transformers.HyperCLOVAXVisionV2Config(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, text_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, vision_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, image_token_id: int = 128060, video_token_id: int = 128061, tie_word_embeddings: bool = True)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/hyperclovax_vision_v2/configuration_hyperclovax_vision_v2.py#L29)

**Parameters:**

text_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the text backbone.

vision_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the vision backbone.

image_token_id (`int`, *optional*, defaults to `128060`) : The image token index used as a placeholder for input images.

video_token_id (`int`, *optional*, defaults to `128061`) : The video token index used as a placeholder for input videos.

tie_word_embeddings (`bool`, *optional*, defaults to `True`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

This is the configuration class to store the configuration of a HyperCLOVAXVisionV2Model. It is used to instantiate a Hyperclovax Vision V2
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [naver-hyperclovax/HyperCLOVAX-SEED-Think-32B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-32B)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

```python
>>> from transformers import HyperCLOVAXVisionV2Config, HyperCLOVAXVisionV2ForConditionalGeneration

>>> # Initializing a HyperCLOVAX Vision V2 configuration with defaults
>>> configuration = HyperCLOVAXVisionV2Config()

>>> # Initializing a model from the configuration
>>> model = HyperCLOVAXVisionV2ForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## HyperCLOVAXVisionV2Processor[[transformers.HyperCLOVAXVisionV2Processor]]

#### transformers.HyperCLOVAXVisionV2Processor[[transformers.HyperCLOVAXVisionV2Processor]]

```python
transformers.HyperCLOVAXVisionV2Processor(image_processor = None, tokenizer = None, video_processor = None, chat_template = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/hyperclovax_vision_v2/processing_hyperclovax_vision_v2.py#L39)

**Parameters:**

image_processor (`Qwen2VLImageProcessor`) : The image processor is a required input.

tokenizer (`GPT2Tokenizer`) : The tokenizer is a required input.

video_processor (`Qwen2VLVideoProcessor`) : The video processor is a required input.

chat_template (`str`) : A Jinja template to convert lists of messages in a chat into a tokenizable string.

Constructs a HyperCLOVAXVisionV2Processor which wraps a image processor, a tokenizer, and a video processor into a single processor.

[HyperCLOVAXVisionV2Processor](/docs/transformers/v5.19.0/en/model_doc/hyperclovax_vision_v2#transformers.HyperCLOVAXVisionV2Processor) offers all the functionalities of [Qwen2VLImageProcessor](/docs/transformers/v5.19.0/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor), [GPT2Tokenizer](/docs/transformers/v5.19.0/en/model_doc/gpt2#transformers.GPT2Tokenizer), and [Qwen2VLVideoProcessor](/docs/transformers/v5.19.0/en/model_doc/qwen2_vl#transformers.Qwen2VLVideoProcessor). See the
[~Qwen2VLImageProcessor](/docs/transformers/v5.19.0/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor), [~GPT2Tokenizer](/docs/transformers/v5.19.0/en/model_doc/gpt2#transformers.GPT2Tokenizer), and [~Qwen2VLVideoProcessor](/docs/transformers/v5.19.0/en/model_doc/qwen2_vl#transformers.Qwen2VLVideoProcessor) for more information.

#### post_process_image_text_to_text[[transformers.HyperCLOVAXVisionV2Processor.post_process_image_text_to_text]]

```python
post_process_image_text_to_text(generated_outputs, skip_special_tokens = True, clean_up_tokenization_spaces = False, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/hyperclovax_vision_v2/processing_hyperclovax_vision_v2.py#L113)

**Parameters:**

generated_outputs (`torch.Tensor` or `np.ndarray`) : The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)` or `(sequence_length,)`.

skip_special_tokens (`bool`, *optional*, defaults to `True`) : Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.

clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`) : Whether or not to clean up the tokenization spaces. Argument passed to the tokenizer's `batch_decode` method.

- ****kwargs** : Additional arguments to be passed to the tokenizer's `batch_decode method`.

**Returns:** `list[str]`

The decoded text.

Post-process the output of the model to decode the text.

## HyperCLOVAXVisionV2Model[[transformers.HyperCLOVAXVisionV2Model]]

#### transformers.HyperCLOVAXVisionV2Model[[transformers.HyperCLOVAXVisionV2Model]]

```python
transformers.HyperCLOVAXVisionV2Model(config: HyperCLOVAXVisionV2Config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/hyperclovax_vision_v2/modeling_hyperclovax_vision_v2.py#L51)

**Parameters:**

config ([HyperCLOVAXVisionV2Config](/docs/transformers/v5.19.0/en/model_doc/hyperclovax_vision_v2#transformers.HyperCLOVAXVisionV2Config)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Hyperclovax Vision V2 Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.HyperCLOVAXVisionV2Model.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, pixel_values: typing.Optional[torch.Tensor] = None, pixel_values_videos: typing.Optional[torch.FloatTensor] = None, image_grid_thw: typing.Optional[torch.LongTensor] = None, video_grid_thw: typing.Optional[torch.LongTensor] = None, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/hyperclovax_vision_v2/modeling_hyperclovax_vision_v2.py#L142)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.19.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

pixel_values (`torch.FloatTensor`, *optional*) : Pixel values of input images after preprocessing by [Qwen2VLImageProcessor](/docs/transformers/v5.19.0/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor). A 2D tensor of shape `(total_num_patches, channels * patch_size^2 * temporal_patch_size)`. In the input token sequence, each image position should contain `config.image_token_id`.

pixel_values_videos (`torch.FloatTensor`, *optional*) : Pixel values of input videos, with the same format as `pixel_values`.

image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) : The temporal, height and width dimensions of the feature grid for each image. Each row contains `[temporal, height, width]` grid counts.

video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) : The temporal, height and width dimensions of the feature grid for each video.

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** [CausalLMOutputWithPast](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

A [CausalLMOutputWithPast](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([HyperCLOVAXVisionV2Config](/docs/transformers/v5.19.0/en/model_doc/hyperclovax_vision_v2#transformers.HyperCLOVAXVisionV2Config)) and inputs.

The [HyperCLOVAXVisionV2Model](/docs/transformers/v5.19.0/en/model_doc/hyperclovax_vision_v2#transformers.HyperCLOVAXVisionV2Model) forward method, overrides the `__call__` special method.

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

#### get_image_features[[transformers.HyperCLOVAXVisionV2Model.get_image_features]]

```python
get_image_features(pixel_values: FloatTensor, image_grid_thw: LongTensor, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/hyperclovax_vision_v2/modeling_hyperclovax_vision_v2.py#L80)

**Parameters:**

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input images.

image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`) : The temporal, height and width of feature shape of each image in LLM.

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([HyperCLOVAXVisionV2Config](/docs/transformers/v5.19.0/en/model_doc/hyperclovax_vision_v2#transformers.HyperCLOVAXVisionV2Config)) and inputs.

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

#### get_video_features[[transformers.HyperCLOVAXVisionV2Model.get_video_features]]

```python
get_video_features(pixel_values_videos: FloatTensor, video_grid_thw: LongTensor, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/hyperclovax_vision_v2/modeling_hyperclovax_vision_v2.py#L65)

**Parameters:**

pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input videos.

video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`) : The temporal, height and width of feature shape of each video in LLM.

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([HyperCLOVAXVisionV2Config](/docs/transformers/v5.19.0/en/model_doc/hyperclovax_vision_v2#transformers.HyperCLOVAXVisionV2Config)) and inputs.

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

## HyperCLOVAXVisionV2ForConditionalGeneration[[transformers.HyperCLOVAXVisionV2ForConditionalGeneration]]

#### transformers.HyperCLOVAXVisionV2ForConditionalGeneration[[transformers.HyperCLOVAXVisionV2ForConditionalGeneration]]

```python
transformers.HyperCLOVAXVisionV2ForConditionalGeneration(config: HyperCLOVAXVisionV2Config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/hyperclovax_vision_v2/modeling_hyperclovax_vision_v2.py#L230)

**Parameters:**

config ([HyperCLOVAXVisionV2Config](/docs/transformers/v5.19.0/en/model_doc/hyperclovax_vision_v2#transformers.HyperCLOVAXVisionV2Config)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Hyperclovax Vision V2 Model for token generation conditioned on other modalities (e.g. image-text-to-text generation).

This model inherits from [PreTrainedModel](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.HyperCLOVAXVisionV2ForConditionalGeneration.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, pixel_values: typing.Optional[torch.FloatTensor] = None, pixel_values_videos: typing.Optional[torch.FloatTensor] = None, image_grid_thw: typing.Optional[torch.LongTensor] = None, video_grid_thw: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/hyperclovax_vision_v2/modeling_hyperclovax_vision_v2.py#L275)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.19.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

pixel_values (`torch.FloatTensor`, *optional*) : Pixel values of input images after preprocessing.

pixel_values_videos (`torch.FloatTensor`, *optional*) : Pixel values of input videos, same format as `pixel_values`.

image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) : `[temporal, height, width]` grid counts per image.

video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) : `[temporal, height, width]` grid counts per video.

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

logits_to_keep (`int` or `torch.Tensor`, *optional*, defaults to 0) : If an `int`, compute logits for the last `logits_to_keep` tokens.

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** [CausalLMOutputWithPast](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

A [CausalLMOutputWithPast](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([HyperCLOVAXVisionV2Config](/docs/transformers/v5.19.0/en/model_doc/hyperclovax_vision_v2#transformers.HyperCLOVAXVisionV2Config)) and inputs.

The [HyperCLOVAXVisionV2ForConditionalGeneration](/docs/transformers/v5.19.0/en/model_doc/hyperclovax_vision_v2#transformers.HyperCLOVAXVisionV2ForConditionalGeneration) forward method, overrides the `__call__` special method.

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
>>> import requests
>>> from transformers import AutoProcessor, HyperCLOVAXVisionV2ForConditionalGeneration

>>> model = HyperCLOVAXVisionV2ForConditionalGeneration.from_pretrained(
...     "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B", device_map="auto"
... )
>>> processor = AutoProcessor.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")

>>> messages = [
...     {"role": "user", "content": [
...         {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
...         {"type": "text", "text": "Describe this image in detail."},
...     ]}
... ]
>>> inputs = processor.apply_chat_template(
...     messages, tokenize=True, return_dict=True, add_generation_prompt=True, return_tensors="pt"
... ).to(model.device)
>>> output = model.generate(**inputs, max_new_tokens=200)
>>> processor.decode(output[0], skip_special_tokens=True)
```

#### get_image_features[[transformers.HyperCLOVAXVisionV2ForConditionalGeneration.get_image_features]]

```python
get_image_features(pixel_values: FloatTensor, image_grid_thw: typing.Optional[torch.LongTensor] = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/hyperclovax_vision_v2/modeling_hyperclovax_vision_v2.py#L262)

**Parameters:**

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input images.

image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) : The temporal, height and width of feature shape of each image in LLM.

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([HyperCLOVAXVisionV2Config](/docs/transformers/v5.19.0/en/model_doc/hyperclovax_vision_v2#transformers.HyperCLOVAXVisionV2Config)) and inputs.

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
>>> from transformers import AutoProcessor, HyperCLOVAXVisionV2ForConditionalGeneration

>>> model = HyperCLOVAXVisionV2ForConditionalGeneration.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")
>>> processor = AutoProcessor.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")

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

#### get_video_features[[transformers.HyperCLOVAXVisionV2ForConditionalGeneration.get_video_features]]

```python
get_video_features(pixel_values_videos: FloatTensor, video_grid_thw: typing.Optional[torch.LongTensor] = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/hyperclovax_vision_v2/modeling_hyperclovax_vision_v2.py#L249)

**Parameters:**

pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input videos.

video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) : The temporal, height and width of feature shape of each video in LLM.

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([HyperCLOVAXVisionV2Config](/docs/transformers/v5.19.0/en/model_doc/hyperclovax_vision_v2#transformers.HyperCLOVAXVisionV2Config)) and inputs.

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
>>> from transformers import AutoProcessor, HyperCLOVAXVisionV2ForConditionalGeneration

>>> model = HyperCLOVAXVisionV2ForConditionalGeneration.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")
>>> processor = AutoProcessor.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")

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

### Code World Model (CWM)
https://huggingface.co/docs/transformers/v5.19.0/model_doc/cwm.md
