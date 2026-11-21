# EXAONE

EXAONE covers the [EXAONE 3.0](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct), [EXAONE 3.5](https://huggingface.co/collections/LGAI-EXAONE/exaone-35) and [EXAONE Deep](https://huggingface.co/collections/LGAI-EXAONE/exaone-deep) language models developed by [LG AI Research](https://huggingface.co/LGAI-EXAONE). Their architecture is the same as [Llama](./llama), with different parameter and config names.

The auto classes load them with the Llama implementation ([LlamaModel](/docs/transformers/v5.24.0/en/model_doc/llama2#transformers.LlamaModel) and [LlamaForCausalLM](/docs/transformers/v5.24.0/en/model_doc/llama2#transformers.LlamaForCausalLM)), so only [ExaoneConfig](/docs/transformers/v5.24.0/en/model_doc/exaone#transformers.ExaoneConfig) is specific to EXAONE. The released checkpoints are renamed to the Llama names when loading and back when saving.

The example below demonstrates how to generate text with the [AutoModelForCausalLM](/docs/transformers/v5.24.0/en/model_doc/auto#transformers.AutoModelForCausalLM) class.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct")
model = AutoModelForCausalLM.from_pretrained("LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct", device_map="auto")

messages = [{"role": "user", "content": "Explain the Pythagorean theorem in one sentence."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
output = model.generate(**inputs.to(model.device), max_new_tokens=64)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

## ExaoneConfig[[transformers.ExaoneConfig]]

#### transformers.ExaoneConfig[[transformers.ExaoneConfig]]

```python
transformers.ExaoneConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vocab_size: int = 102400, hidden_size: int = 4096, intermediate_size: int = 14336, num_hidden_layers: int = 32, num_attention_heads: int = 32, num_key_value_heads: int | None = 8, hidden_act: str = 'silu', max_position_embeddings: int = 4096, initializer_range: float = 0.02, rms_norm_eps: float = 1e-05, use_cache: bool = True, pad_token_id: int | None = 0, bos_token_id: int | None = 1, eos_token_id: int | list[int] | None = 361, tie_word_embeddings: bool = False, rope_parameters: transformers.modeling_rope_utils.RopeParameters | dict | None = None, attention_bias: bool = False, attention_dropout: int | float | None = 0.0, mlp_bias: bool = False, head_dim: int | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/exaone/configuration_exaone.py#L31)

**Parameters:**

vocab_size (`int`, *optional*, defaults to `102400`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

hidden_size (`int`, *optional*, defaults to `4096`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `14336`) : Dimension of the MLP representations.

num_hidden_layers (`int`, *optional*, defaults to `32`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `32`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*, defaults to `8`) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

hidden_act (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

max_position_embeddings (`int`, *optional*, defaults to `4096`) : The maximum sequence length that this model might ever be used with.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rms_norm_eps (`float`, *optional*, defaults to `1e-05`) : The epsilon used by the rms normalization layers.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

pad_token_id (`int`, *optional*, defaults to `0`) : Token id used for padding in the vocabulary.

bos_token_id (`int`, *optional*, defaults to `1`) : Token id used for beginning-of-stream in the vocabulary.

eos_token_id (`Union[int, list[int]]`, *optional*, defaults to `361`) : Token id used for end-of-stream in the vocabulary.

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

rope_parameters (`Union[~modeling_rope_utils.RopeParameters, dict]`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

attention_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

attention_dropout (`Union[int, float]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

mlp_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.

head_dim (`int`, *optional*) : The attention head dimension. If None, it will default to hidden_size // num_attention_heads

This is the configuration class to store the configuration of a LlamaModel. It is used to instantiate a Exaone
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.24.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.24.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

```python
>>> from transformers import AutoModelForCausalLM, ExaoneConfig

>>> # Initializing an EXAONE-3.0-7.8B style configuration
>>> configuration = ExaoneConfig()

>>> # Initializing a model (which uses the Llama implementation) from that configuration
>>> model = AutoModelForCausalLM.from_config(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

### KOSMOS-2
https://huggingface.co/docs/transformers/v5.24.0/model_doc/kosmos-2.md
