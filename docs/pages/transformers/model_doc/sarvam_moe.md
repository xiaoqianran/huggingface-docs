# SarvamMoE

SarvamMoE is the architecture of [Sarvam-30B](https://huggingface.co/sarvamai/sarvam-30b), a 30B parameter Mixture of Experts (MoE) language model developed by [Sarvam AI](https://www.sarvam.ai/). Its larger sibling, Sarvam-105B, uses [SarvamMLA](./sarvam_mla).

Key architectural features:

- **Grouped-query attention** with QK normalization and full rotary position embeddings.
- **Sparse Mixture of Experts**: 128 routed experts with 6 active per token, plus 1 shared expert. The first layer uses a dense MLP.
- **Sigmoid routing** with an expert bias correction and a routed scaling factor.

The architecture is the same as [GLM-4-MoE](./glm4_moe), with different parameter and config names. The auto classes load it with the GLM-4-MoE implementation ([Glm4MoeModel](/docs/transformers/v5.24.1/en/model_doc/glm4_moe#transformers.Glm4MoeModel) and [Glm4MoeForCausalLM](/docs/transformers/v5.24.1/en/model_doc/glm4_moe#transformers.Glm4MoeForCausalLM)), so only [SarvamMoEConfig](/docs/transformers/v5.24.1/en/model_doc/sarvam_moe#transformers.SarvamMoEConfig) is specific to SarvamMoE. The released checkpoints are converted to the GLM-4-MoE names when loading and back when saving.

The example below demonstrates how to generate text with the [AutoModelForCausalLM](/docs/transformers/v5.24.1/en/model_doc/auto#transformers.AutoModelForCausalLM) class.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-30b")
model = AutoModelForCausalLM.from_pretrained("sarvamai/sarvam-30b", device_map="auto")

messages = [{"role": "user", "content": "Explain the Pythagorean theorem in one sentence."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
output = model.generate(**inputs.to(model.device), max_new_tokens=64)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

## SarvamMoEConfig[[transformers.SarvamMoEConfig]]

#### transformers.SarvamMoEConfig[[transformers.SarvamMoEConfig]]

```python
transformers.SarvamMoEConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vocab_size: int = 262144, hidden_size: int = 4096, intermediate_size: int = 8192, num_hidden_layers: int = 19, num_attention_heads: int = 64, num_key_value_heads: int = 4, hidden_act: str = 'silu', max_position_embeddings: int = 131072, initializer_range: float = 0.006, rms_norm_eps: float = 1e-06, use_cache: bool = True, tie_word_embeddings: bool = False, rope_parameters: transformers.modeling_rope_utils.RopeParameters | dict | None = None, attention_bias: bool = False, attention_dropout: float | int = 0.0, moe_intermediate_size: int = 1024, num_experts_per_tok: int = 6, output_router_logits: bool = False, n_shared_experts: int = 1, n_routed_experts: int = 128, routed_scaling_factor: float = 2.5, n_group: int = 1, topk_group: int = 1, first_k_dense_replace: int = 1, norm_topk_prob: bool = True, use_qk_norm: bool = True, bos_token_id: int | None = None, eos_token_id: int | list[int] | None = 1, pad_token_id: int | None = 0, num_mtp_layers: int = 0, head_dim: int = 64)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/sarvam_moe/configuration_sarvam_moe.py#L30)

**Parameters:**

vocab_size (`int`, *optional*, defaults to `262144`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

hidden_size (`int`, *optional*, defaults to `4096`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `8192`) : Dimension of the MLP representations.

num_hidden_layers (`int`, *optional*, defaults to `19`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `64`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*, defaults to `4`) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

hidden_act (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

max_position_embeddings (`int`, *optional*, defaults to `131072`) : The maximum sequence length that this model might ever be used with.

initializer_range (`float`, *optional*, defaults to `0.006`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rms_norm_eps (`float`, *optional*, defaults to `1e-06`) : The epsilon used by the rms normalization layers.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

rope_parameters (`Union[~modeling_rope_utils.RopeParameters, dict]`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

attention_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

moe_intermediate_size (`int`, *optional*, defaults to `1024`) : Intermediate size of the routed expert MLPs.

num_experts_per_tok (`int`, *optional*, defaults to `6`) : Number of experts to route each token to. This is the top-k value for the token-choice routing.

output_router_logits (`bool`, *optional*, defaults to `False`) : Whether or not the router logits should be returned by the model. Enabling this will also allow the model to output the auxiliary loss, including load balancing loss and router z-loss.

n_shared_experts (`int`, *optional*, defaults to `1`) : Number of shared experts.

n_routed_experts (`int`, *optional*, defaults to `128`) : Number of routed experts.

routed_scaling_factor (`float`, *optional*, defaults to `2.5`) : Scaling factor or routed experts.

n_group (`int`, *optional*, defaults to 1) : Number of groups for routed experts.

topk_group (`int`, *optional*, defaults to `1`) : Number of selected groups for each token (for each token, ensuring the selected experts is only within `topk_group` groups).

first_k_dense_replace (`int`, *optional*, defaults to 1) : Number of dense layers in shallow layers(embed->dense->moe->moe...->lm_head). \--k dense layers--/

norm_topk_prob (`bool`, *optional*, defaults to `True`) : Whether to normalize the weights of the routed experts. 

use_qk_norm (`bool`, *optional*, defaults to `True`) : Whether to use query-key normalization in the attention.

bos_token_id (`int`, *optional*) : Token id used for beginning-of-stream in the vocabulary.

eos_token_id (`Union[int, list[int]]`, *optional*, defaults to `1`) : Token id used for end-of-stream in the vocabulary.

pad_token_id (`int`, *optional*, defaults to `0`) : Token id used for padding in the vocabulary.

num_mtp_layers (`int`, *optional*, defaults to 0) : Number of Multi-Token Prediction (MTP) modules available to append after the base transformer model. The released checkpoints do not include MTP weights.

head_dim (`int`, *optional*, defaults to `64`) : The attention head dimension. If None, it will default to hidden_size // num_attention_heads

This is the configuration class to store the configuration of a Glm4MoeModel. It is used to instantiate a Sarvam Moe
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [sarvamai/sarvam-30b](https://huggingface.co/sarvamai/sarvam-30b)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.24.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.24.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import AutoModelForCausalLM, SarvamMoEConfig

>>> # Initializing a Sarvam-30B style configuration
>>> configuration = SarvamMoEConfig()

>>> # Initializing a model (which uses the GLM-4-MoE implementation) from that configuration
>>> model = AutoModelForCausalLM.from_config(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

### PatchTSMixer
https://huggingface.co/docs/transformers/v5.24.1/model_doc/patchtsmixer.md
