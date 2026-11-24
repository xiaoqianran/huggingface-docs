# SarvamMLA

## Overview

SarvamMLA is a 105B parameter Mixture of Experts (MoE) language model developed by [Sarvam AI](https://www.sarvam.ai/). It uses Multi-head Latent Attention (MLA) combined with sparse MoE routing, architecturally identical to DeepSeek-V3.

Key architectural features:

- **Multi-head Latent Attention (MLA)**: Low-rank KV compression with decoupled RoPE, reducing KV cache memory while maintaining performance.
- **Sparse Mixture of Experts**: 128 routed experts with 8 active per token, plus 1 shared expert. The first layer uses a dense MLP.
- **DeepSeek YaRN RoPE**: Extended context support up to 131K tokens via YaRN rotary position embeddings.
- **Sigmoid routing with group-based top-k**: Token-choice routing using sigmoid scores with expert bias correction and group-aware selection.

This model uses the DeepSeek-V3 architecture with a custom configuration. See the [DeepSeek-V3 documentation](deepseek_v3) for model and forward reference.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "sarvamai/sarvam-105b",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-105b")

inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## SarvamMLAConfig[[transformers.SarvamMLAConfig]]

#### transformers.SarvamMLAConfig[[transformers.SarvamMLAConfig]]

```python
transformers.SarvamMLAConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vocab_size: int = 262144, hidden_size: int = 4096, intermediate_size: int = 16384, moe_intermediate_size: int = 2048, num_hidden_layers: int = 32, num_attention_heads: int = 64, num_key_value_heads: int | None = None, n_shared_experts: int = 1, n_routed_experts: int = 128, output_router_logits: bool = False, routed_scaling_factor: float = 2.5, kv_lora_rank: int = 512, q_lora_rank: int | None = None, qk_rope_head_dim: int = 64, v_head_dim: int | None = 128, qk_nope_head_dim: int = 128, n_group: int | None = 1, topk_group: int | None = 1, num_experts_per_tok: int | None = 8, first_k_dense_replace: int | None = 1, norm_topk_prob: bool | None = True, hidden_act: str = 'silu', max_position_embeddings: int = 4096, initializer_range: float = 0.006, rms_norm_eps: float = 1e-06, use_cache: bool = True, pad_token_id: int | None = None, bos_token_id: int | None = 0, eos_token_id: int | list[int] | None = 1, pretraining_tp: int | None = 1, tie_word_embeddings: bool = False, rope_parameters: transformers.modeling_rope_utils.RopeParameters | dict | None = None, rope_interleave: bool | None = True, attention_bias: bool = False, attention_dropout: float | int | None = 0.0, num_mtp_layers: int = 0)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/sarvam_mla/configuration_sarvam_mla.py#L33)

**Parameters:**

vocab_size (`int`, *optional*, defaults to `262144`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

hidden_size (`int`, *optional*, defaults to `4096`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `16384`) : Dimension of the MLP representations.

moe_intermediate_size (`int`, *optional*, defaults to `2048`) : Intermediate size of the routed expert MLPs.

num_hidden_layers (`int`, *optional*, defaults to `32`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `64`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

n_shared_experts (`int`, *optional*, defaults to `1`) : Number of shared experts.

n_routed_experts (`int`, *optional*, defaults to `128`) : Number of routed experts.

output_router_logits (`bool`, *optional*, defaults to `False`) : Whether or not the router logits should be returned by the model. Enabling this will also allow the model to output the auxiliary loss, including load balancing loss and router z-loss.

routed_scaling_factor (`float`, *optional*, defaults to `2.5`) : Scaling factor or routed experts.

kv_lora_rank (`int`, *optional*, defaults to `512`) : Rank of the LoRA matrices for key and value projections.

q_lora_rank (`int`, *optional*) : Rank of the LoRA matrices for query projections.

qk_rope_head_dim (`int`, *optional*, defaults to `64`) : Dimension of the query/key heads that use rotary position embeddings.

v_head_dim (`int`, *optional*, defaults to `128`) : Dimension of the value heads.

qk_nope_head_dim (`int`, *optional*, defaults to `128`) : Dimension of the query/key heads that don't use rotary position embeddings.

n_group (`int`, *optional*, defaults to 1) : Number of groups for routed experts. The released checkpoints route over all experts at once.

topk_group (`int`, *optional*, defaults to 1) : Number of selected groups for each token.

num_experts_per_tok (`int`, *optional*, defaults to `8`) : Number of experts to route each token to. This is the top-k value for the token-choice routing.

first_k_dense_replace (`int`, *optional*, defaults to 1) : Number of dense layers in shallow layers(embed->dense->moe->moe...->lm_head). \--k dense layers--/

norm_topk_prob (`bool`, *optional*, defaults to `True`) : Whether to normalize the weights of the routed experts. 

hidden_act (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

max_position_embeddings (`int`, *optional*, defaults to `4096`) : The maximum sequence length that this model might ever be used with.

initializer_range (`float`, *optional*, defaults to `0.006`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rms_norm_eps (`float`, *optional*, defaults to `1e-06`) : The epsilon used by the rms normalization layers.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

pad_token_id (`int`, *optional*) : Token id used for padding in the vocabulary.

bos_token_id (`int`, *optional*, defaults to `0`) : Token id used for beginning-of-stream in the vocabulary.

eos_token_id (`Union[int, list[int]]`, *optional*, defaults to `1`) : Token id used for end-of-stream in the vocabulary.

pretraining_tp (`int`, *optional*, defaults to `1`) : Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this document](https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism) to understand more about it. This value is necessary to ensure exact reproducibility of the pretraining results. Please refer to [this issue](https://github.com/pytorch/pytorch/issues/76232).

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

rope_parameters (`Union[~modeling_rope_utils.RopeParameters, dict]`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

rope_interleave (`bool`, *optional*, defaults to `True`) : Whether to interleave the rotary position embeddings.

attention_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

num_mtp_layers (`int`, *optional*, defaults to 0) : Number of Multi-Token Prediction (MTP) modules available to append after the base transformer model. The released checkpoints do not include MTP weights.

This is the configuration class to store the configuration of a DeepseekV3Model. It is used to instantiate a Sarvam Mla
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [sarvamai/sarvam-105b](https://huggingface.co/sarvamai/sarvam-105b)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.24.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.24.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import SarvamMLAConfig

>>> # Initializing a SarvamMLA style configuration
>>> configuration = SarvamMLAConfig()

>>> # Accessing the model configuration
>>> configuration = model.config
```

### PP-DocLayoutV3
https://huggingface.co/docs/transformers/v5.24.1/model_doc/pp_doclayout_v3.md
