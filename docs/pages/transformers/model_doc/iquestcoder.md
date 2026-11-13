# IQuestCoder

[IQuest-Coder-V1](https://huggingface.co/IQuestLab/IQuest-Coder-V1-40B-Instruct) is a family of code language models developed by [IQuest Lab](https://huggingface.co/IQuestLab), available in 7B, 14B and 40B sizes with base, instruct and thinking variants. Its architecture is the same as [Llama](./llama).

> [!NOTE]
> The looped `IQuest-Coder-V1-*-Loop-*` checkpoints use a different architecture and aren't supported by this model.

> [!WARNING]
> The tokenizer is built from the checkpoints' SentencePiece model. Many of its pieces share the same score, and SentencePiece breaks those ties by position, which the `tokenizers` library can't reproduce, so a small fraction of texts tokenize differently from the original remote code.

The auto classes load it with the Llama implementation ([LlamaModel](/docs/transformers/v5.23.0/en/model_doc/llama2#transformers.LlamaModel) and [LlamaForCausalLM](/docs/transformers/v5.23.0/en/model_doc/llama2#transformers.LlamaForCausalLM)), so only [IQuestCoderConfig](/docs/transformers/v5.23.0/en/model_doc/iquestcoder#transformers.IQuestCoderConfig) is specific to IQuestCoder.

The example below demonstrates how to generate text with the [AutoModelForCausalLM](/docs/transformers/v5.23.0/en/model_doc/auto#transformers.AutoModelForCausalLM) class.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("IQuestLab/IQuest-Coder-V1-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("IQuestLab/IQuest-Coder-V1-7B-Instruct", device_map="auto")

messages = [{"role": "user", "content": "Write a Python function that checks whether a number is prime."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
output = model.generate(**inputs.to(model.device), max_new_tokens=128)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

## IQuestCoderConfig[[transformers.IQuestCoderConfig]]

#### transformers.IQuestCoderConfig[[transformers.IQuestCoderConfig]]

```python
transformers.IQuestCoderConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vocab_size: int = 76800, hidden_size: int = 5120, intermediate_size: int = 27648, num_hidden_layers: int = 80, num_attention_heads: int = 40, num_key_value_heads: int | None = 8, hidden_act: str = 'silu', max_position_embeddings: int = 131072, initializer_range: float = 0.02, rms_norm_eps: float = 1e-05, use_cache: bool = True, pad_token_id: int | None = None, bos_token_id: int | None = 1, eos_token_id: int | list[int] | None = 2, tie_word_embeddings: bool = False, rope_parameters: transformers.modeling_rope_utils.RopeParameters | dict | None = None, attention_bias: bool = False, attention_dropout: float | int = 0.0, mlp_bias: bool = False, head_dim: int | None = 128)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/iquestcoder/configuration_iquestcoder.py#L30)

**Parameters:**

vocab_size (`int`, *optional*, defaults to `76800`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

hidden_size (`int`, *optional*, defaults to `5120`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `27648`) : Dimension of the MLP representations.

num_hidden_layers (`int`, *optional*, defaults to `80`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `40`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*, defaults to `8`) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

hidden_act (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

max_position_embeddings (`int`, *optional*, defaults to `131072`) : The maximum sequence length that this model might ever be used with.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rms_norm_eps (`float`, *optional*, defaults to `1e-05`) : The epsilon used by the rms normalization layers.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

pad_token_id (`int`, *optional*) : Token id used for padding in the vocabulary.

bos_token_id (`int`, *optional*, defaults to `1`) : Token id used for beginning-of-stream in the vocabulary.

eos_token_id (`Union[int, list[int]]`, *optional*, defaults to `2`) : Token id used for end-of-stream in the vocabulary.

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

rope_parameters (`Union[~modeling_rope_utils.RopeParameters, dict]`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

attention_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

mlp_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.

head_dim (`int`, *optional*, defaults to `128`) : The attention head dimension. If None, it will default to hidden_size // num_attention_heads

This is the configuration class to store the configuration of a LlamaModel. It is used to instantiate a Iquestcoder
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [IQuestLab/IQuest-Coder-V1-40B-Instruct](https://huggingface.co/IQuestLab/IQuest-Coder-V1-40B-Instruct)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

```python
>>> from transformers import AutoModelForCausalLM, IQuestCoderConfig

>>> # Initializing an IQuest-Coder-V1-40B style configuration
>>> configuration = IQuestCoderConfig()

>>> # Initializing a model (which uses the Llama implementation) from that configuration
>>> model = AutoModelForCausalLM.from_config(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

### ModernBERT Decoder
https://huggingface.co/docs/transformers/v5.23.0/model_doc/modernbert-decoder.md
