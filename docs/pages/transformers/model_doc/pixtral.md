# Pixtral

[Pixtral](https://huggingface.co/papers/2410.07073) is a multimodal model trained to understand natural images and documents. It accepts images in their natural resolution and aspect ratio without resizing or padding due to its 2D RoPE embeddings. In addition, Pixtral has a long 128K token context window for processing a large number of images. Pixtral couples a 400M vision encoder with a 12B Mistral Nemo decoder.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/pixtral_architecture.webp"
alt="drawing" width="600"/>

 Pixtral architecture. Taken from the blog post. 

You can find all the original Pixtral checkpoints under the [Mistral AI](https://huggingface.co/mistralai/models?search=pixtral) organization.

> [!TIP]
> This model was contributed by [amyeroberts](https://huggingface.co/amyeroberts) and [ArthurZ](https://huggingface.co/ArthurZ).
> Click on the Pixtral models in the right sidebar for more examples of how to apply Pixtral to different vision and language tasks.

```python
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "mistral-community/pixtral-12b"
model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

url_dog = "https://picsum.photos/id/237/200/300"
url_mountain = "https://picsum.photos/seed/picsum/200/300"

chat = [
    {
      "role": "user", "content": [
        {"type": "text", "content": "Can this animal"}, 
        {"type": "image", "url": url_dog}, 
        {"type": "text", "content": "live here?"}, 
        {"type": "image", "url" : url_mountain}
      ]
    }
]

inputs = processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
generate_ids = model.generate(**inputs, max_new_tokens=500)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to quantize the model to 4-bits.

```python
import requests
import torch
from PIL import Image

from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

model_id = "mistral-community/pixtral-12b"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

dog_url = "https://picsum.photos/id/237/200/300"
mountain_url = "https://picsum.photos/seed/picsum/200/300"
dog_image = Image.open(requests.get(dog_url, stream=True).raw)
mountain_image = Image.open(requests.get(mountain_url, stream=True).raw)

chat = [
    {
      "role": "user", "content": [
        {"type": "text", "text": "Can this animal"},
        {"type": "image"},
        {"type": "text", "text": "live here?"},
        {"type": "image"}
      ]
    }
]

prompt = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
inputs = processor(text=prompt, images=[dog_image, mountain_image], return_tensors="pt").to(model.device)

inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

generate_ids = model.generate(**inputs, max_new_tokens=100)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output)
```

## Notes

- Pixtral uses [PixtralVisionModel](/docs/transformers/v5.19.0/en/model_doc/pixtral#transformers.PixtralVisionModel) as the vision encoder and [MistralForCausalLM](/docs/transformers/v5.19.0/en/model_doc/mistral#transformers.MistralForCausalLM)  for its language decoder.
- The model internally replaces `[IMG]` token placeholders with image embeddings.

    ```py
    "<s>[INST][IMG]\nWhat are the things I should be cautious about when I visit this place?[/INST]"
    ```

    The `[IMG]` tokens are replaced with a number of `[IMG]` tokens that depend on the height and width of each image. Each row of the image is separated by a `[IMG_BREAK]` token and each image is separated by a `[IMG_END]` token. Use the `~Processor.apply_chat_template` method to handle these tokens for you.

## PixtralVisionConfig[[transformers.PixtralVisionConfig]]

#### transformers.PixtralVisionConfig[[transformers.PixtralVisionConfig]]

```python
transformers.PixtralVisionConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, hidden_size: int = 1024, intermediate_size: int = 4096, num_hidden_layers: int = 24, num_attention_heads: int = 16, num_channels: int = 3, image_size: int | list[int] | tuple[int, int] = 1024, patch_size: int | list[int] | tuple[int, int] = 16, hidden_act: str = 'gelu', attention_dropout: float | int = 0.0, rope_parameters: transformers.modeling_rope_utils.RopeParameters | dict | None = None, initializer_range: float = 0.02)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/pixtral/configuration_pixtral.py#L24)

**Parameters:**

hidden_size (`int`, *optional*, defaults to `1024`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `4096`) : Dimension of the MLP representations.

num_hidden_layers (`int`, *optional*, defaults to `24`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `16`) : Number of attention heads for each attention layer in the Transformer decoder.

num_channels (`int`, *optional*, defaults to `3`) : The number of input channels.

image_size (`Union[int, list[int], tuple[int, int]]`, *optional*, defaults to `1024`) : The size (resolution) of each image.

patch_size (`Union[int, list[int], tuple[int, int]]`, *optional*, defaults to `16`) : The size (resolution) of each patch.

hidden_act (`str`, *optional*, defaults to `gelu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

rope_parameters (`Union[~modeling_rope_utils.RopeParameters, dict]`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a PixtralVisionModel. It is used to instantiate a Pixtral
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [mistral-labs/pixtral-12b](https://huggingface.co/mistral-labs/pixtral-12b)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import PixtralVisionModel, PixtralVisionConfig

>>> # Initializing a Pixtral-12B style configuration
>>> config = PixtralVisionConfig()

>>> # Initializing a model (with randomly initialized weights) from the configuration
>>> model = PixtralVisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## MistralCommonBackend[[transformers.MistralCommonBackend]]

#### transformers.MistralCommonBackend[[transformers.MistralCommonBackend]]

```python
transformers.MistralCommonBackend(tokenizer_path: str | os.PathLike | pathlib.Path, mode: str | ValidationMode = 'test', model_max_length: int = 1000000000000000019884624838656, padding_side: str = 'left', truncation_side: str = 'right', model_input_names: list[str] | None = None, clean_up_tokenization_spaces: bool = False, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/tokenization_mistral_common.py#L194)

Class to wrap `mistral-common` tokenizers.

`mistral-common` is the official tokenizer library for Mistral AI models. To use it, you need to install it with:

```bash
pip install transformers[mistral-common]
```

Otherwise the tokenizer falls back to the Transformers implementation of the tokenizer.

For more info on `mistral-common`, see [mistral-common](https://github.com/mistralai/mistral-common).

This class is a wrapper around a `mistral_common.tokens.tokenizers.mistral.MistralTokenizer`.
It provides a Hugging Face compatible interface to tokenize using the official mistral-common tokenizer and inherits from the `PreTrainedTokenizerBase` class.

Here are the key behavior differences with the `PythonBackend` class:

- Pair of sequences are not supported. The signature has been kept for compatibility but all arguments related to pair of sequences are ignored. The return values for pairs are returned as `None`.
- The `is_split_into_words` argument is not supported.
- It is not possible to add new tokens to the tokenizer. Special tokens are handled differently from Transformers. In `mistral-common`, special tokens are never encoded directly. This means that: `tokenizer.encode("<s>")` will not return the ID of the `<s>` token. Instead, it will return a list of IDs corresponding to the tokenization of the string `"<s>"`. For more information, see the [mistral-common documentation](https://mistralai.github.io/mistral-common/usage/tokenizers/#special-tokens).

If you have suggestions to improve this class, please open an issue on the [mistral-common GitHub repository](https://github.com/mistralai/mistral-common/issues) if it is related to the tokenizer or on the [Transformers GitHub repository](https://github.com/huggingface/transformers/issues) if it is related to the Hugging Face interface.

#### add_special_tokens[[transformers.MistralCommonBackend.add_special_tokens]]

```python
add_special_tokens(special_tokens_dict: dict, replace_extra_special_tokens: bool = True)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/tokenization_mistral_common.py#L1675)

`MistralCommonBackend` does not implement `add_special_tokens` by design.

If you would like this behaviour to be implemented, please open an issue in the `Transformers` or `mistral-common` repositories to request it.

#### add_tokens[[transformers.MistralCommonBackend.add_tokens]]

```python
add_tokens(special_tokens_dict: dict, replace_extra_special_tokens: bool = True)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/tokenization_mistral_common.py#L1687)

`MistralCommonBackend` does not implement `add_special_tokens` by design.

If you would like this behaviour to be implemented, please open an issue in the `Transformers` or `mistral-common` repositories to request it.

#### apply_chat_template[[transformers.MistralCommonBackend.apply_chat_template]]

```python
apply_chat_template(conversation: list[dict[str, str]] | list[list[dict[str, str]]], tools: list[dict | collections.abc.Callable] | None = None, add_generation_prompt: bool = False, continue_final_message: bool = False, tokenize: bool = True, padding: bool | str | transformers.utils.generic.PaddingStrategy = False, truncation: bool = False, max_length: int | None = None, return_tensors: str | transformers.utils.generic.TensorType | None = None, return_dict: bool = True, reasoning_effort: ReasoningEffort | None = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/tokenization_mistral_common.py#L1034)

**Parameters:**

conversation (Union[List[Dict[str, str]], List[List[Dict[str, str]]]]) : A list of dicts with "role" and "content" keys, representing the chat history so far.

tools (`List[Union[Dict, Callable]]`, *optional*) : A list of tools (callable functions) that will be accessible to the model. If the template does not support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema, giving the name, description and argument types for the tool. See our [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use) for more information.

add_generation_prompt (`bool`, *optional*) : This argument is a no-op for `MistralCommonBackend`. However, it cannot be used at the same time as `continue_final_message` to keep the API consistent. If any conversation ends with an assistant message, it will raise an error. In such cases, use `continue_final_message` instead.

continue_final_message (bool, *optional*) : If this is set, the chat will be formatted so that the final message in the chat is open-ended, without any EOS tokens. The model will continue this message rather than starting a new one. This allows you to "prefill" part of the model's response for it. Cannot be used at the same time as `add_generation_prompt`.

tokenize (`bool`, defaults to `True`) : Whether to tokenize the output. If `False`, the output will be a string.

padding (`bool`, `str` or [PaddingStrategy](/docs/transformers/v5.19.0/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) : Select a strategy to pad the returned sequences (according to the model's padding side and padding index) among:  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence if provided). - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).

truncation (`bool`, defaults to `False`) : Whether to truncate sequences at the maximum length. Has no effect if tokenize is `False`.

max_length (`int`, *optional*) : Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is `False`. If not specified, the tokenizer's `max_length` attribute will be used as a default.

return_tensors (`str` or [TensorType](/docs/transformers/v5.19.0/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Has no effect if tokenize is `False`. Acceptable values are: - `'pt'`: Return PyTorch `torch.Tensor` objects.

return_dict (`bool`, defaults to `False`) : Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`. If at least one conversation contains an image, its pixel values will be returned in the `pixel_values` key and image sizes in the `image_sizes` key.

reasoning_effort (`ReasoningEffort`, *optional*) : The reasoning effort to use for the chat completion for models that support it. Possible values are: - `ReasoningEffort.none`: The model will not reason. - `ReasoningEffort.high`: The model will use a reasoning approach. If not specified, the default reasoning effort will be used. 

kwargs (additional keyword arguments, *optional*) : Additional arguments passed to the mistral-common `ChatCompletionRequest.from_openai` method.

**Returns:** `Union[str, list[int], list[str], list[list[int]], BatchEncoding]`

The tokenized chat so far, including control tokens. This output is ready to pass to the model, either directly or via methods like `generate()`.

Converts a list of dictionaries with `"role"` and `"content"` keys to a list of token
ids.

#### batch_decode[[transformers.MistralCommonBackend.batch_decode]]

```python
batch_decode(sequences: typing.Union[list[int], list[list[int]], numpy.ndarray, ForwardRef('torch.Tensor')], skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool | None = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/tokenization_mistral_common.py#L508)

**Parameters:**

sequences (`Union[list[int], list[list[int]], np.ndarray, torch.Tensor]`) : List of tokenized input ids. Can be obtained using the `__call__` method.

skip_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not to remove special tokens in the decoding.

clean_up_tokenization_spaces (`bool`, *optional*) : Whether or not to clean up the tokenization spaces. If `None`, will default to `self.clean_up_tokenization_spaces`.

kwargs (additional keyword arguments, *optional*) : Not supported by `MistralCommonBackend.batch_decode`. Will raise an error if used.

**Returns:** `list[str]`

The list of decoded sentences.

Convert a list of lists of token ids into a list of strings by calling decode.

This method is provided for backwards compatibility. The `decode` method now handles batched input natively,
so you can use `decode` directly instead of `batch_decode`.

#### build_inputs_with_special_tokens[[transformers.MistralCommonBackend.build_inputs_with_special_tokens]]

```python
build_inputs_with_special_tokens(token_ids_0: list, token_ids_1: None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/tokenization_mistral_common.py#L1280)

**Parameters:**

token_ids_0 (`list[int]`) : List of IDs to which the special tokens will be added.

token_ids_1 (`None`, *optional*) : None, kept to match Transformers' signature.

**Returns:** `list[int]`

List of input IDs with the appropriate special tokens.

Build model inputs from a sequence by adding special tokens.

This method dynamically builds inputs based on the tokenizer's `mode`:
- `"test"`: seq0 [EOS]
- `"finetuning"`: [BOS] seq0

#### convert_added_tokens[[transformers.MistralCommonBackend.convert_added_tokens]]

```python
convert_added_tokens(obj: typing.Union[tokenizers.AddedToken, typing.Any], save: bool = False, add_type_field: bool = True)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/tokenization_mistral_common.py#L1700)

`MistralCommonBackend` does not implement `convert_added_tokens` by design.

If you would like this behaviour to be implemented, please open an issue in the `Transformers` or `mistral-common` repositories to request it.

#### create_token_type_ids_from_sequences[[transformers.MistralCommonBackend.create_token_type_ids_from_sequences]]

```python
create_token_type_ids_from_sequences(token_ids_0: list, token_ids_1: None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/tokenization_mistral_common.py#L1309)

**Parameters:**

token_ids_0 (`list[int]`) : List of IDs.

token_ids_1 (`None`, *optional*) : None, kept to match Transformers' signature.

**Returns:** `list[int]`

Token type IDs according to the configured pattern.

Create a mask of zeroes from the token ids with special tokens added.

Kept to match Transformers' implementation.

#### decode[[transformers.MistralCommonBackend.decode]]

```python
decode(token_ids: typing.Union[int, list[int], list[list[int]], numpy.ndarray, ForwardRef('torch.Tensor')], skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool | None = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/tokenization_mistral_common.py#L471)

**Parameters:**

token_ids (`Union[int, list[int], list[list[int]], np.ndarray, torch.Tensor]`) : A single sequence or a batch (list of sequences) of tokenized input ids. Can be obtained using the `__call__` method.

skip_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not to remove special tokens in the decoding.

clean_up_tokenization_spaces (`bool`, *optional*) : Whether or not to clean up the tokenization spaces. If `None`, will default to `self.clean_up_tokenization_spaces`.

kwargs (additional keyword arguments, *optional*) : Not supported by `MistralCommonBackend.decode`. Will raise an error if used.

**Returns:** `Union[str, list[str]]`

The decoded string for a single sequence, or a list of decoded strings for a
batch of sequences.

Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
tokens and clean up tokenization spaces.

#### encode[[transformers.MistralCommonBackend.encode]]

```python
encode(text: str | list[int], text_pair: None = None, add_special_tokens: bool = True, padding: bool | str | transformers.utils.generic.PaddingStrategy = False, truncation: bool | str | transformers.tokenization_utils_base.TruncationStrategy | None = None, max_length: int | None = None, stride: int = 0, pad_to_multiple_of: int | None = None, padding_side: str | None = None, return_tensors: str | transformers.utils.generic.TensorType | None = None, verbose: bool = True, return_offsets_mapping: typing.Literal[False] = False, split_special_tokens: typing.Literal[False] = False, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/tokenization_mistral_common.py#L374)

**Parameters:**

text (`str` or `list[int]`) : The first sequence to be encoded. This can be a string or a list of integers (tokenized string ids).

text_pair (`None`, *optional*) : Not supported by `MistralCommonBackend.encode`. Kept to match `PreTrainedTokenizerBase.encode` signature. 

add_special_tokens (`bool`, *optional*, defaults to `True`) : Whether or not to add special tokens when encoding the sequences. This will use the underlying `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are automatically added to the input ids. This is useful if you want to add `bos` or `eos` tokens automatically. When Tokenizer is loading with `finetuning` mode it adds both `bos` and `eos`. Else, for "test" mode it only adds `bos`.

padding (`bool`, `str` or [PaddingStrategy](/docs/transformers/v5.19.0/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) : Activates and controls padding. Accepts the following values:  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence is provided). - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).

truncation (`bool`, `str` or [TruncationStrategy](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) : Activates and controls truncation. Accepts the following values:  - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater than the model maximum admissible input size).

max_length (`int`, *optional*) : Controls the maximum length to use by one of the truncation/padding parameters.  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.

stride (`int`, *optional*, defaults to 0) : If set to a number along with `max_length`, the overflowing tokens returned when `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence returned to provide some overlap between truncated and overflowing sequences. The value of this argument defines the number of overlapping tokens.

pad_to_multiple_of (`int`, *optional*) : If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).

padding_side (`str`, *optional*) : The side on which the model should have padding applied. Should be selected between ['right', 'left']. Default value is picked from the class attribute of the same name.

return_tensors (`str` or [TensorType](/docs/transformers/v5.19.0/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors instead of list of python integers. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. 

- ****kwargs** : Not supported by `MistralCommonBackend.encode`. Will raise an error if used.

**Returns:** `list[int]`, `torch.Tensor`

The tokenized ids of the text.

Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

#### from_pretrained[[transformers.MistralCommonBackend.from_pretrained]]

```python
from_pretrained(pretrained_model_name_or_path: str | os.PathLike, *init_inputs, mode: str | ValidationMode = 'test', cache_dir: str | os.PathLike | None = None, force_download: bool = False, local_files_only: bool = False, token: str | bool | None = None, revision: str = 'main', model_max_length: int = 1000000000000000019884624838656, padding_side: str = 'left', truncation_side: str = 'right', model_input_names: list[str] | None = None, clean_up_tokenization_spaces: bool = False, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/tokenization_mistral_common.py#L1433)

**Parameters:**

pretrained_model_name_or_path (`str` or `os.PathLike`) : Can be either:  - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co. - A path to a *directory* containing the tokenizer config, for instance saved using the `MistralCommonBackend.tokenization_mistral_common.save_pretrained` method, e.g., `./my_model_directory/`.

mode (`Union[str, ValidationMode]`, *optional*, defaults to `ValidationMode.test`) : Validation mode for the `MistralTokenizer` tokenizer. Possible values are: - `"finetuning"` or `ValidationMode.finetuning`: The fine-tuning mode. - `"test"` or `ValidationMode.test`: The test mode. It changes how the tokenizer validates the input and prepares the request to the model.

cache_dir (`str` or `os.PathLike`, *optional*) : Path to a directory in which a downloaded predefined tokenizer vocabulary files should be cached if the standard cache should not be used.

force_download (`bool`, *optional*, defaults to `False`) : Whether or not to force the (re-)download the vocabulary files and override the cached versions if they exist.

token (`str` or *bool*, *optional*) : The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated when running `hf auth login` (stored in `~/.huggingface`).

local_files_only (`bool`, *optional*, defaults to `False`) : Whether or not to only rely on local files and not to attempt to download any files.

revision (`str`, *optional*, defaults to `"main"`) : The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git.

max_length (`int`, *optional*) : Controls the maximum length to use by one of the truncation/padding parameters.  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.

padding_side (`str`, *optional*, defaults to `"left"`) : The side on which the model should have padding applied. Should be selected between ['right', 'left']. Default value is picked from the class attribute of the same name.

truncation_side (`str`, *optional*, defaults to `"right"`) : The side on which the model should have truncation applied. Should be selected between ['right', 'left'].

model_input_names (`List[str]`, *optional*) : The list of inputs accepted by the forward pass of the model (like `"token_type_ids"` or `"attention_mask"`). Default value is picked from the class attribute of the same name.

clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`) : Whether or not the model should clean up the spaces that were added when splitting the input text during the tokenization process.

kwargs (additional keyword arguments, *optional*) : Not supported by `MistralCommonBackend.from_pretrained`. Will raise an error if used.

Instantiate a `MistralCommonBackend` from a predefined
tokenizer.

#### get_chat_template[[transformers.MistralCommonBackend.get_chat_template]]

```python
get_chat_template(chat_template: str | None = None, tools: list[dict] | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/tokenization_mistral_common.py#L1709)

`MistralCommonBackend` does not implement `get_chat_template` by design as `mistral-common` does not use chat templates.

#### get_special_tokens_mask[[transformers.MistralCommonBackend.get_special_tokens_mask]]

```python
get_special_tokens_mask(token_ids_0: list, token_ids_1: None = None, already_has_special_tokens: bool = False)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/tokenization_mistral_common.py#L687)

**Parameters:**

token_ids_0 (`list[int]`) : List of ids of the sequence.

token_ids_1 (`None`, *optional*) : None, kept to match Transformers' implementation.

already_has_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not the token list is already formatted with special tokens for the model.

**Returns:** A list of integers in the range [0, 1]

1 for a special token, 0 for a sequence token.

Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

#### get_vocab[[transformers.MistralCommonBackend.get_vocab]]

```python
get_vocab()
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/tokenization_mistral_common.py#L348)

**Returns:** `Dict[str, int]`

The vocabulary.

Returns the vocabulary as a dictionary of token to index.

This is a lossy conversion. There may be multiple token ids that decode to the same
string due to partial UTF-8 byte sequences being converted to �.

#### num_special_tokens_to_add[[transformers.MistralCommonBackend.num_special_tokens_to_add]]

```python
num_special_tokens_to_add(pair: typing.Literal[False] = False)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/tokenization_mistral_common.py#L1333)

**Parameters:**

pair (`Literal[False]`, *optional*) : False, kept to match Transformer's signature.

**Returns:** `int`

Number of special tokens added to sequences.

Returns the number of added tokens when encoding a sequence with special tokens.

This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
this inside your training loop.

#### prepare_for_model[[transformers.MistralCommonBackend.prepare_for_model]]

```python
prepare_for_model(ids: list, pair_ids: None = None, add_special_tokens: bool = True, padding: bool | str | transformers.utils.generic.PaddingStrategy = False, truncation: bool | str | transformers.tokenization_utils_base.TruncationStrategy | None = None, max_length: int | None = None, stride: int = 0, pad_to_multiple_of: int | None = None, padding_side: str | None = None, return_tensors: str | transformers.utils.generic.TensorType | None = None, return_token_type_ids: bool | None = None, return_attention_mask: bool | None = None, return_overflowing_tokens: bool = False, return_special_tokens_mask: bool = False, return_length: bool = False, verbose: bool = True, prepend_batch_axis: bool = False, return_offsets_mapping: typing.Literal[False] = False, split_special_tokens: typing.Literal[False] = False, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/tokenization_mistral_common.py#L840)

**Parameters:**

ids (`list[int]`) : Tokenized input ids of the first sequence.

pair_ids (`None`, *optional*) : Not supported by `MistralCommonBackend`. Kept to match the interface of `PreTrainedTokenizerBase`. 

add_special_tokens (`bool`, *optional*, defaults to `True`) : Whether or not to add special tokens when encoding the sequences. This will use the underlying `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are automatically added to the input ids. This is useful if you want to add `bos` or `eos` tokens automatically. When Tokenizer is loading with `finetuning` mode it adds both `bos` and `eos`. Else, for "test" mode it only adds `bos`.

padding (`bool`, `str` or [PaddingStrategy](/docs/transformers/v5.19.0/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) : Activates and controls padding. Accepts the following values:  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence is provided). - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).

truncation (`bool`, `str` or [TruncationStrategy](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) : Activates and controls truncation. Accepts the following values:  - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater than the model maximum admissible input size).

max_length (`int`, *optional*) : Controls the maximum length to use by one of the truncation/padding parameters.  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.

stride (`int`, *optional*, defaults to 0) : If set to a number along with `max_length`, the overflowing tokens returned when `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence returned to provide some overlap between truncated and overflowing sequences. The value of this argument defines the number of overlapping tokens.

pad_to_multiple_of (`int`, *optional*) : If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).

padding_side (`str`, *optional*) : The side on which the model should have padding applied. Should be selected between ['right', 'left']. Default value is picked from the class attribute of the same name.

return_tensors (`str` or [TensorType](/docs/transformers/v5.19.0/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors instead of list of python integers. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. 

return_token_type_ids (`bool`, *optional*) : Whether to return token type IDs. For `MistralCommonBackend` it returns a list of zeros of the sequence length as only one sequence is supported.  [What are token type IDs?](../glossary#token-type-ids)

return_attention_mask (`bool`, *optional*) : Whether to return the attention mask. If left to the default, will return the attention mask according to the specific tokenizer's default, defined by the `return_outputs` attribute.  [What are attention masks?](../glossary#attention-mask)

return_overflowing_tokens (`bool`, *optional*, defaults to `False`) : Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead of returning overflowing tokens.

return_special_tokens_mask (`bool`, *optional*, defaults to `False`) : Whether or not to return special tokens mask information.

return_length  (`bool`, *optional*, defaults to `False`) : Whether or not to return the lengths of the encoded inputs.

verbose (`bool`, *optional*, defaults to `True`) : Whether or not to print more information and warnings.

return_offsets_mapping (`Literal[False]`, *optional*) : False, kept to match Transformers' signature.

split_special_tokens (`Literal[False]`, *optional*) : False, kept to match Transformers' signature.

- ****kwargs** : passed to the `self.tokenize()` method

**Returns:** [BatchEncoding](/docs/transformers/v5.19.0/en/main_classes/tokenizer#transformers.BatchEncoding)

A [BatchEncoding](/docs/transformers/v5.19.0/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

- **input_ids** -- List of token ids to be fed to a model.

  [What are input IDs?](../glossary#input-ids)

- **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).

  [What are attention masks?](../glossary#attention-mask)

- **overflowing_tokens** -- List of overflowing tokens sequences (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
- **num_truncated_tokens** -- Number of tokens truncated (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
- **special_tokens_mask** -- List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
  regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
- **length** -- The length of the inputs (when `return_length=True`)

Prepares a sequence of input id so that it can be used by the model. It
adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
manages a moving window (with user defined stride) for overflowing tokens.

#### save_chat_templates[[transformers.MistralCommonBackend.save_chat_templates]]

```python
save_chat_templates(save_directory: str | os.PathLike, tokenizer_config: dict, filename_prefix: str | None, save_jinja_files: bool)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/tokenization_mistral_common.py#L1714)

`MistralCommonBackend` does not implement `save_chat_templates` by design as `mistral-common` does not use chat templates.

#### save_pretrained[[transformers.MistralCommonBackend.save_pretrained]]

```python
save_pretrained(save_directory: str | os.PathLike | pathlib.Path, push_to_hub: bool = False, token: str | bool | None = None, commit_message: str | None = None, repo_id: str | None = None, private: bool | None = None, save_format: str | None = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/tokenization_mistral_common.py#L1548)

**Parameters:**

save_directory (`str | os.PathLike | Path`) : Directory where the tokenizer will be saved.

push_to_hub (`bool`, *optional*, defaults to `False`) : Push to the HF Hub after saving.

token (`str` or `bool`, *optional*) : Token for Hub authentication.

commit_message (`str`, *optional*) : Commit message when pushing.

repo_id (`str`, *optional*) : Hub repository id.

private (`bool`, *optional*) : Whether the repository is private.

save_format (`str`, *optional*) : `"mistral"` to save native tekken format, `"hf"` for HuggingFace format.  *None* preserves the original format.

kwargs (`dict`, *optional*) : Not supported — will raise an error.

**Returns:** `tuple[str, ...]`

Paths of the saved files.

Save the full tokenizer state.

When `save_format` is `"mistral"` (or *None* for a tekken tokenizer),
the original tokenizer file (e.g. `tekken.json`) is written to
*save_directory*.  When `save_format` is `"hf"`, HF-format files
(`tokenizer.json` + `tokenizer_config.json`) are produced instead.

#### save_vocabulary[[transformers.MistralCommonBackend.save_vocabulary]]

```python
save_vocabulary(save_directory: str, filename_prefix: str | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/tokenization_mistral_common.py#L1725)

`MistralCommonBackend` does not implement `save_vocabulary` by design.

This is because `mistral-common` is configured by one tokenizer file. If you'd like to save the vocabulary, please consider using the `save_pretrained` method instead.

#### tokenize[[transformers.MistralCommonBackend.tokenize]]

```python
tokenize(text: str, return_offsets_mapping: typing.Literal[False] = False, split_special_tokens: typing.Literal[False] = False, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/tokenization_mistral_common.py#L641)

**Parameters:**

text (`str`) : The sequence to be encoded.

return_offsets_mapping (`Literal[False]`, *optional*) : False, kept to match Transformers' signature.

split_special_tokens (`Literal[False]`, *optional*) : False, kept to match Transformers' signature.

- ****kwargs** (additional keyword arguments) : Not supported by `MistralCommonBackend.tokenize`. Will raise an error if used.

**Returns:** `list[str]`

The list of tokens.

Converts a string into a sequence of tokens, using the tokenizer.

Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies.

#### truncate_sequences[[transformers.MistralCommonBackend.truncate_sequences]]

```python
truncate_sequences(ids: list, pair_ids: None = None, num_tokens_to_remove: int = 0, truncation_strategy: str | transformers.tokenization_utils_base.TruncationStrategy = 'longest_first', stride: int = 0, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/tokenization_mistral_common.py#L970)

**Parameters:**

ids (`list[int]`) : Tokenized input ids. Can be obtained from a string by chaining the `tokenize` and `convert_tokens_to_ids` methods.

pair_ids (`None`, *optional*) : Not supported by `MistralCommonBackend`. Kept to match the signature of `PreTrainedTokenizerBase.truncate_sequences`.

num_tokens_to_remove (`int`, *optional*, defaults to 0) : Number of tokens to remove using the truncation strategy.

truncation_strategy (`str` or [TruncationStrategy](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `'longest_first'`) : The strategy to follow for truncation. Can be:  - `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. - `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater than the model maximum admissible input size).

stride (`int`, *optional*, defaults to 0) : If set to a positive number, the overflowing tokens returned will contain some tokens from the main sequence returned. The value of this argument defines the number of additional tokens.

**Returns:** `Tuple[list[int], None, list[int]]`

The truncated `ids` and the list of
overflowing tokens. `None` is returned to match Transformers signature.

Truncates a sequence pair in-place following the strategy.

## PixtralVisionModel[[transformers.PixtralVisionModel]]

#### transformers.PixtralVisionModel[[transformers.PixtralVisionModel]]

```python
transformers.PixtralVisionModel(config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/pixtral/modeling_pixtral.py#L379)

**Parameters:**

config ([PixtralVisionModel](/docs/transformers/v5.19.0/en/model_doc/pixtral#transformers.PixtralVisionModel)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Pixtral Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.PixtralVisionModel.forward]]

```python
forward(pixel_values: Tensor, image_sizes: typing.Optional[torch.Tensor] = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/pixtral/modeling_pixtral.py#L402)

**Parameters:**

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input images. Pixel values can be obtained using [PixtralImageProcessor](/docs/transformers/v5.19.0/en/model_doc/pixtral#transformers.PixtralImageProcessor). See `PixtralImageProcessor.__call__()` for details ([PixtralProcessor](/docs/transformers/v5.19.0/en/model_doc/pixtral#transformers.PixtralProcessor) uses [PixtralImageProcessor](/docs/transformers/v5.19.0/en/model_doc/pixtral#transformers.PixtralImageProcessor) for processing images).

image_sizes (`torch.Tensor` of shape `(batch_size, 2)`, *optional*) : The sizes of the images in the batch, being (height, width) for each image.

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PixtralVisionConfig](/docs/transformers/v5.19.0/en/model_doc/pixtral#transformers.PixtralVisionConfig)) and inputs.

The [PixtralVisionModel](/docs/transformers/v5.19.0/en/model_doc/pixtral#transformers.PixtralVisionModel) forward method, overrides the `__call__` special method.

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

## PixtralImageProcessor[[transformers.PixtralImageProcessor]]

#### transformers.PixtralImageProcessor[[transformers.PixtralImageProcessor]]

```python
transformers.PixtralImageProcessor(**kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/pixtral/image_processing_pixtral.py#L99)

**Parameters:**

do_convert_rgb (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to convert the image to RGB.

do_resize (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to resize the image.

size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*, defaults to `{'longest_edge' : 1024}`): Describes the maximum input dimensions to the model.

default_to_square (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to default to a square image when resizing, if size is an int.

crop_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : Size of the output image after applying `center_crop`.

resample (`Annotated[Union[int, PILImageResampling, NoneType], None]`, *kwargs*, defaults to `Resampling.BICUBIC`) : Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.

do_rescale (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to rescale the image.

rescale_factor (`float`, *kwargs*, *optional*, defaults to `0.00392156862745098`) : Rescale factor to rescale the image by if `do_rescale` is set to `True`.

do_normalize (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to normalize the image.

image_mean (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`) : Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.

image_std (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`) : Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to `True`.

do_pad (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to pad the image. Padding is done either to the largest size in the batch or to a fixed square size per image. The exact padding strategy depends on the model.

pad_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : The size in `{"height": int, "width" int}` to pad the images to. Must be larger than any image size provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest height and width in the batch. Applied only when `do_pad=True.`

do_center_crop (`bool`, *kwargs*, *optional*) : Whether to center crop the image.

data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.

input_data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format. - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

device (`Annotated[Union[str, torch.device, NoneType], None]`, *kwargs*) : The device to process the videos on. If unset, the device is inferred from the input videos.

return_tensors (`Annotated[str | ~utils.generic.TensorType | None, None]`, *kwargs*) : Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors.

disable_grouping (`bool`, *kwargs*, *optional*) : Whether to disable grouping of images by size to process them individually and not in batches. If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157

image_seq_length (`int`, *kwargs*, *optional*) : The number of image tokens to be used for each image in the input. Added for backward compatibility but this should be set as a processor attribute in future models.

patch_size (`Union[dict[str, *kwargs*, int], int]` *optional*, defaults to `{"height" : 16, "width": 16}`): Size of the patches in the model, used to calculate the output image size.

Constructs a PixtralImageProcessor image processor.

#### preprocess[[transformers.PixtralImageProcessor.preprocess]]

```python
preprocess(images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']], **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/pixtral/image_processing_pixtral.py#L118)

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

patch_size (`Union[dict[str, *kwargs*, int], int]` *optional*, defaults to `{"height" : 16, "width": 16}`): Size of the patches in the model, used to calculate the output image size.

**Returns:** `~image_processing_base.BatchFeature`

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

## PixtralImageProcessorPil[[transformers.PixtralImageProcessorPil]]

#### transformers.PixtralImageProcessorPil[[transformers.PixtralImageProcessorPil]]

```python
transformers.PixtralImageProcessorPil(**kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/pixtral/image_processing_pil_pixtral.py#L107)

**Parameters:**

do_convert_rgb (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to convert the image to RGB.

do_resize (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to resize the image.

size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*, defaults to `{'longest_edge' : 1024}`): Describes the maximum input dimensions to the model.

default_to_square (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to default to a square image when resizing, if size is an int.

crop_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : Size of the output image after applying `center_crop`.

resample (`Annotated[Union[int, PILImageResampling, NoneType], None]`, *kwargs*, defaults to `Resampling.BICUBIC`) : Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.

do_rescale (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to rescale the image.

rescale_factor (`float`, *kwargs*, *optional*, defaults to `0.00392156862745098`) : Rescale factor to rescale the image by if `do_rescale` is set to `True`.

do_normalize (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to normalize the image.

image_mean (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`) : Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.

image_std (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`) : Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to `True`.

do_pad (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to pad the image. Padding is done either to the largest size in the batch or to a fixed square size per image. The exact padding strategy depends on the model.

pad_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : The size in `{"height": int, "width" int}` to pad the images to. Must be larger than any image size provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest height and width in the batch. Applied only when `do_pad=True.`

do_center_crop (`bool`, *kwargs*, *optional*) : Whether to center crop the image.

data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.

input_data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format. - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

device (`Annotated[Union[str, torch.device, NoneType], None]`, *kwargs*) : The device to process the videos on. If unset, the device is inferred from the input videos.

return_tensors (`Annotated[str | ~utils.generic.TensorType | None, None]`, *kwargs*) : Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors.

disable_grouping (`bool`, *kwargs*, *optional*) : Whether to disable grouping of images by size to process them individually and not in batches. If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157

image_seq_length (`int`, *kwargs*, *optional*) : The number of image tokens to be used for each image in the input. Added for backward compatibility but this should be set as a processor attribute in future models.

patch_size (`Union[dict[str, *kwargs*, int], int]` *optional*, defaults to `{"height" : 16, "width": 16}`): Size of the patches in the model, used to calculate the output image size.

Constructs a PixtralImageProcessor image processor.

#### preprocess[[transformers.PixtralImageProcessorPil.preprocess]]

```python
preprocess(images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']], **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/pixtral/image_processing_pil_pixtral.py#L126)

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

patch_size (`Union[dict[str, *kwargs*, int], int]` *optional*, defaults to `{"height" : 16, "width": 16}`): Size of the patches in the model, used to calculate the output image size.

**Returns:** `~image_processing_base.BatchFeature`

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

## PixtralProcessor[[transformers.PixtralProcessor]]

#### transformers.PixtralProcessor[[transformers.PixtralProcessor]]

```python
transformers.PixtralProcessor(image_processor = None, tokenizer = None, patch_size: int = 16, spatial_merge_size: int = 1, chat_template = None, image_token = '[IMG]', image_break_token = '[IMG_BREAK]', image_end_token = '[IMG_END]', **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/pixtral/processing_pixtral.py#L50)

**Parameters:**

image_processor (`PixtralImageProcessor`) : The image processor is a required input.

tokenizer (`MistralCommonBackend`) : The tokenizer is a required input.

patch_size (`int`, *optional*, defaults to 16) : Patch size from the vision tower.

spatial_merge_size (`int`, *optional*, defaults to 1) : The downsampling factor for the spatial merge operation.

chat_template (`str`) : A Jinja template to convert lists of messages in a chat into a tokenizable string.

image_token (`str`, *optional*, defaults to `"[IMG]"`) : Special token used to denote image location.

image_break_token (`str`, *optional*, defaults to `"[IMG_BREAK]"`) : Special token used to denote the end of a line of pixels in an image.

image_end_token (`str`, *optional*, defaults to `"[IMG_END]"`) : Special token used to denote the end of an image input.

Constructs a PixtralProcessor which wraps a image processor and a tokenizer into a single processor.

[PixtralProcessor](/docs/transformers/v5.19.0/en/model_doc/pixtral#transformers.PixtralProcessor) offers all the functionalities of [PixtralImageProcessor](/docs/transformers/v5.19.0/en/model_doc/pixtral#transformers.PixtralImageProcessor) and [MistralCommonBackend](/docs/transformers/v5.19.0/en/model_doc/mixtral#transformers.MistralCommonBackend). See the
[~PixtralImageProcessor](/docs/transformers/v5.19.0/en/model_doc/pixtral#transformers.PixtralImageProcessor) and [~MistralCommonBackend](/docs/transformers/v5.19.0/en/model_doc/mixtral#transformers.MistralCommonBackend) for more information.

#### __call__[[transformers.PixtralProcessor.__call__]]

```python
__call__(images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None, text: str | list[str] | list[list[str]] | None = None, videos: typing.Union[list['PIL.Image.Image'], numpy.ndarray, ForwardRef('torch.Tensor'), list[numpy.ndarray], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list[numpy.ndarray]], list[list['torch.Tensor']], transformers.video_utils.URL, list[transformers.video_utils.URL], list[list[transformers.video_utils.URL]], transformers.video_utils.Path, list[transformers.video_utils.Path], list[list[transformers.video_utils.Path]], NoneType] = None, audio: typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence['torch.Tensor'], NoneType] = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/processing_utils.py#L657)

**Parameters:**

images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]`, *optional*) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

text (`Union[str, list[str], list[list[str]]]`, *optional*) : The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings (pretokenized string). If you pass a pretokenized input, set `is_split_into_words=True` to avoid ambiguity with batched inputs.

videos (`Union[list[PIL.Image.Image], numpy.ndarray, torch.Tensor, list[numpy.ndarray], list[torch.Tensor], list[list[PIL.Image.Image]], list[list[numpy.ndarray]], list[list[torch.Tensor]], ~video_utils.URL, list[~video_utils.URL], list[list[~video_utils.URL]], ~video_utils.Path, list[~video_utils.Path], list[list[~video_utils.Path]]]`, *optional*) : Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If passing in videos with pixel values between 0 and 1, set `do_rescale=False`.

audio (`Union[numpy.ndarray, torch.Tensor, collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence[torch.Tensor]]`, *optional*) : The audio or batch of audios to be prepared. Each audio can be a NumPy array or PyTorch tensor. In case of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels, and T is the sample length of the audio.

return_tensors (`str` or [TensorType](/docs/transformers/v5.19.0/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

- ****kwargs** ([ProcessingKwargs](/docs/transformers/v5.19.0/en/main_classes/processors#transformers.ProcessingKwargs), *optional*) : Additional processing options for each modality (text, images, videos, audio). Model-specific parameters are listed above; see the TypedDict class for the complete list of supported arguments.

### BLIP
https://huggingface.co/docs/transformers/v5.19.0/model_doc/blip.md
