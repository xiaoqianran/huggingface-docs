# Gradium Phonon TTS

## Overview

Gradium Phonon TTS is [Gradium](https://gradium.ai)'s family of on-device text-to-speech models, trained with the recipe of [Pocket TTS](./pocket_tts): a causal transformer reads a voice prompt and the text, then emits one continuous 32-dimensional audio latent per step at 12.5 Hz through a one-step flow head, and an end-of-speech head decides when the utterance is over. The latents are decoded to 24 kHz audio by a [Pocket TTS audio tokenizer](./pocket_tts_audio_tokenizer) held by the processor, Pocket TTS's [PocketTTSProcessor](/docs/transformers/v5.22.1/en/model_doc/pocket_tts#transformers.PocketTTSProcessor).

It differs from Pocket TTS in its voice conditioning: the voice prompt is made of 512-dimensional latents of a separate speaker codec, and a learnt speaker vector is added to every audio latent position. The speaker codec is not released, so voices come as precomputed latents (each checkpoint's `default-voice.safetensors`, and the catalog of [gradium/phonon-voices](https://huggingface.co/gradium/phonon-voices)) and the audio tokenizer is a decoder only: voices cannot be cloned from a recording.

| Checkpoint | Parameters | Languages | License |
|---|---|---|---|
| [gradium/phonon-31m-en](https://huggingface.co/gradium/phonon-31m-en) | 31M | English | Apache 2.0 |
| [gradium/phonon-40m-en](https://huggingface.co/gradium/phonon-40m-en) | 40M | English | CC BY-NC 4.0 |
| [gradium/phonon-40m-fr](https://huggingface.co/gradium/phonon-40m-fr) | 40M | French | CC BY-NC 4.0 |
| [gradium/phonon-40m-de](https://huggingface.co/gradium/phonon-40m-de) | 40M | German | CC BY-NC 4.0 |
| [gradium/phonon-40m-es](https://huggingface.co/gradium/phonon-40m-es) | 40M | Spanish | CC BY-NC 4.0 |
| [gradium/phonon-40m-pt](https://huggingface.co/gradium/phonon-40m-pt) | 40M | Portuguese | CC BY-NC 4.0 |
| [gradium/phonon-40m-multilingual](https://huggingface.co/gradium/phonon-40m-multilingual) | 40M | English, French, German, Spanish, Portuguese | CC BY-NC 4.0 |

They all load the same way: switching checkpoints means changing the repo id.

This model was contributed by [Eustache Le Bihan](https://huggingface.co/eustlb). The original code is [gradium-ai/xn-ptts](https://github.com/gradium-ai/xn-ptts).

## Usage tips

- Prepare inputs with [apply_chat_template()](/docs/transformers/v5.22.1/en/main_classes/processors#transformers.ProcessorMixin.apply_chat_template): one `user` message per text, an empty `{"type": "audio"}` item where the voice goes, the voice latents in `processor_kwargs={"audio_prompt_latents": [...]}`, and `add_generation_prompt=True`.
- The tokenizer does not spell out numbers, amounts, symbols or addresses, see [Text normalization](#text-normalization).
- `generate` returns `audio_latents` and `audio_latents_mask`; decode them with [PocketTTSProcessor.decode()](/docs/transformers/v5.22.1/en/model_doc/pocket_tts#transformers.PocketTTSProcessor.decode).
- Generation samples as Gradium's runtime does: `temperature` 0.3 (the variance of the flow's starting noise), and it stops shortly after the end-of-speech logit crosses `eos_threshold`. `do_sample=False` starts the flow from zero noise.
- The model loads in bf16, the dtype its weights are stored in; pass `dtype=torch.float32` on CPU to match Gradium's runtime there.
- Split long text with [PocketTTSProcessor.split_text()](/docs/transformers/v5.22.1/en/model_doc/pocket_tts#transformers.PocketTTSProcessor.split_text), one conversation per chunk.

## Usage

### Voice

A voice is a tensor of latents `(num_frames, 512)`. Each checkpoint ships one (`default-voice.safetensors`), and [gradium/phonon-voices](https://huggingface.co/gradium/phonon-voices) has about a hundred more, in every language, each under its voice ID (`voices.jsonl` lists their names, languages and descriptions). Any voice works with any checkpoint.

```python
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoProcessor, GradiumPhononTTSForConditionalGeneration

model_id = "gradium/phonon-31m-en"
processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
model = GradiumPhononTTSForConditionalGeneration.from_pretrained(model_id, device_map="auto")

def load_voice(voice_id):
    """A voice of gradium/phonon-voices, as latents of shape (num_frames, 512)."""
    path = hf_hub_download("gradium/phonon-voices", f"voices/{voice_id}.safetensors")
    return load_file(path)["speaker_wavs"][0].T.float()

# Easton (en-us); the checkpoint's own voice: hf_hub_download(model_id, "default-voice.safetensors")
voice = load_voice("LMDIYE0Oa377Qh6Z")
# an empty `audio` item marks the voice
conversation = [
    {"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": "Hello from Phonon."}]}
]
inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
    add_generation_prompt=True,
    processor_kwargs={"audio_prompt_latents": [voice]},
).to(model.device)

outputs = model.generate(**inputs)
audio = processor.decode(outputs.audio_latents, outputs.audio_latents_mask)
processor.save_audio(audio, "phonon.wav")
```

### Batches and long text

Batch several texts, each with its own voice, with long text split into chunks:

```python
text = (
    "Phonon is a small text-to-speech model that runs on your device. "
    "Long texts are split into sentences, and each chunk is one conversation of the batch."
)
chunks = processor.split_text(text)
texts = chunks + ["And this one is said by another voice."]
voices = [load_voice("LMDIYE0Oa377Qh6Z")] * len(chunks) + [load_voice("NbpkqMVS3CJeq2j8")]
conversations = [
    [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": text}]}] for text in texts
]
inputs = processor.apply_chat_template(
    conversations,
    tokenize=True,
    return_dict=True,
    add_generation_prompt=True,
    processor_kwargs={"audio_prompt_latents": voices},
).to(model.device)

outputs = model.generate(**inputs)
audio = processor.decode(outputs.audio_latents, outputs.audio_latents_mask)
processor.save_audio([torch.cat(audio[: len(chunks)]), audio[-1]], ["long.wav", "other.wav"])
```

### Text normalization

The tokenizer normalizes what does not depend on the language, as Gradium's runtime does:

| Text | Tokenizer |
|---|---|
| `“Hello” — it’s here…` | `"Hello" it's here.` |
| `hello (world); see you 🚀` | `Hello, world, see you.` |

It does not spell out numbers, amounts, symbols or addresses, which depend on the language:

| Text | Tokenizer | Gradium's runtime |
|---|---|---|
| `It costs $1,234.56.` | `It costs $1,234.56.` | `It costs 1 thousand 234 dollars and 56 cents.` |
| `Write to jean@gradium.ai` | `Write to jean@gradium.ai.` | `Write to jean at gradium dot A-I.` |
| `It is 21°C` | `It is 21°C.` | `It is 21 degrees Celsius.` |

For those, use the text normalization of Gradium's runtime, [ptts](https://github.com/gradium-ai/xn-ptts), or write them out.

## GradiumPhononTTSConfig[[transformers.GradiumPhononTTSConfig]]

#### transformers.GradiumPhononTTSConfig[[transformers.GradiumPhononTTSConfig]]

```python
transformers.GradiumPhononTTSConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, flow_head_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, vocab_size: int = 4001, hidden_size: int = 768, num_hidden_layers: int = 6, num_attention_heads: int = 12, num_key_value_heads: int | None = None, head_dim: int | None = None, intermediate_size: int = 3072, hidden_act: str = 'gelu', layer_norm_eps: float = 1e-05, attention_bias: bool = False, attention_dropout: float = 0.0, max_position_embeddings: int = 8192, rope_parameters: transformers.modeling_rope_utils.RopeParameters | dict | None = None, latent_dim: int = 32, pad_token_id: int = 3, bos_token_id: int | None = None, eos_token_id: int | None = None, audio_prompt_token_id: int = 5, audio_latent_token_id: int = 6, audio_bos_token_id: int = 4, frame_rate: float = 12.5, initializer_range: float = 0.02, use_cache: bool = True, eos_loss_weight: float = 0.1, flow_loss_num_samples: int = 4, lsd_distill_probability: float = 0.25, lsd_diagonal_weight: float = 0.75, lsd_time_logit_mean: float = 0.4, lsd_time_logit_std: float = 1.0, latent_stats_ema_steps: int = 1000, latent_stats_ema_decay: float = 0.999, audio_prompt_dim: int = 512, num_speakers: int = 1, num_speakers_vocab_size: int = 32, num_speakers_embedding_dim: int = 16, duration_delta: float = 0.0, duration_delta_embedding_dim: int = 128, duration_delta_scale: float = 1000.0, duration_delta_max_period: float = 10000.0)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/gradium_phonon_tts/configuration_gradium_phonon_tts.py#L82)

**Parameters:**

flow_head_config (`Union[GradiumPhononTTSFlowHeadConfig, dict]`, *optional*) : The config of the flow head that samples each latent from the backbone hidden state.

vocab_size (`int`, *optional*, defaults to `4001`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

hidden_size (`int`, *optional*, defaults to `768`) : Dimension of the hidden representations.

num_hidden_layers (`int`, *optional*, defaults to `6`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `12`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

head_dim (`int`, *optional*) : The attention head dimension. If None, it will default to hidden_size // num_attention_heads

intermediate_size (`int`, *optional*, defaults to `3072`) : Dimension of the MLP representations.

hidden_act (`str`, *optional*, defaults to `gelu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

layer_norm_eps (`float`, *optional*, defaults to 1e-05) : Epsilon of the layer normalizations of the backbone.

attention_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

attention_dropout (`float`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

max_position_embeddings (`int`, *optional*, defaults to `8192`) : The maximum sequence length that this model might ever be used with.

rope_parameters (`Union[~modeling_rope_utils.RopeParameters, dict]`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

latent_dim (`int`, *optional*, defaults to 32) : Dimension of the audio latents the backbone takes as input (`audio_latents`); equals the flow head's and the audio tokenizer's `latent_dim`.

pad_token_id (`int`, *optional*, defaults to `3`) : Token id used for padding in the vocabulary.

bos_token_id (`int`, *optional*) : Token id used for beginning-of-stream in the vocabulary.

eos_token_id (`int`, *optional*) : Token id used for end-of-stream in the vocabulary.

audio_prompt_token_id (`int`, *optional*, defaults to 5) : Placeholder token id of the voice-prompt frames in `input_ids` (`<|audio_prompt|>`).

audio_latent_token_id (`int`, *optional*, defaults to 6) : Placeholder token id of the audio latent positions in `input_ids` (`<|audio_latent|>`).

audio_bos_token_id (`int`, *optional*, defaults to 4) : Token id of the first audio position, closing the prompt: it holds the learnt BOS latent, from which the first frame is generated (`<|audio_bos|>`).

frame_rate (`float`, *optional*, defaults to 12.5) : Number of audio latents per second of audio: `generate` derives its default frame budget from it.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

eos_loss_weight (`float`, *optional*, defaults to 0.1) : Weight of the end-of-speech loss in the training loss.

flow_loss_num_samples (`int`, *optional*, defaults to 4) : Number of noise draws per target frame in the flow loss.

lsd_distill_probability (`float`, *optional*, defaults to 0.25) : Probability, per training step, of computing the self-distillation term of the LSD loss (reweighted to keep the expected loss); 1 computes it at every step.

lsd_diagonal_weight (`float`, *optional*, defaults to 0.75) : Weight of the diagonal (instantaneous flow) term of the LSD loss; the self-distillation term weighs the rest.

lsd_time_logit_mean (`float`, *optional*, defaults to 0.4) : Mean of the logit-normal distribution of the flow times drawn by the LSD loss.

lsd_time_logit_std (`float`, *optional*, defaults to 1.0) : Standard deviation of the logit-normal distribution of the flow times drawn by the LSD loss.

latent_stats_ema_steps (`int`, *optional*, defaults to 1000) : Number of first training steps during which `latent_mean` and `latent_std` follow an exponential moving average of the target latents; 0 keeps them frozen.

latent_stats_ema_decay (`float`, *optional*, defaults to 0.999) : Decay of that moving average.

audio_prompt_dim (`int`, *optional*, defaults to 512) : Dimension of the voice-prompt latents (`audio_prompt_latents`), the unquantized latents of Gradium's speaker codec (a Mimi encoder, not the audio tokenizer's), which the backbone projects to `hidden_size`.

num_speakers (`int`, *optional*, defaults to 1) : Value of the `num_speakers` condition: the row of its lookup table added to every audio latent position.

num_speakers_vocab_size (`int`, *optional*, defaults to 32) : Number of rows of the `num_speakers` lookup table.

num_speakers_embedding_dim (`int`, *optional*, defaults to 16) : Width of the `num_speakers` lookup table, projected to `hidden_size`.

duration_delta (`float`, *optional*, defaults to 0.0) : Value of the `duration_delta` condition, embedded sinusoidally and added to every audio latent position.

duration_delta_embedding_dim (`int`, *optional*, defaults to 128) : Size of the sinusoidal embedding of `duration_delta`, projected to `hidden_size`.

duration_delta_scale (`float`, *optional*, defaults to 1000.0) : Factor applied to `duration_delta` before its sinusoidal embedding.

duration_delta_max_period (`float`, *optional*, defaults to 10000.0) : Maximum period of the sinusoidal embedding of `duration_delta`.

This is the configuration class to store the configuration of a GradiumPhononTTSModel. It is used to instantiate a Gradium Phonon Tts
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [gradium/phonon-31m-en](https://huggingface.co/gradium/phonon-31m-en)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## GradiumPhononTTSFlowHeadConfig[[transformers.GradiumPhononTTSFlowHeadConfig]]

#### transformers.GradiumPhononTTSFlowHeadConfig[[transformers.GradiumPhononTTSFlowHeadConfig]]

```python
transformers.GradiumPhononTTSFlowHeadConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, hidden_size: int = 512, num_hidden_layers: int = 6, latent_dim: int = 32, condition_dim: int = 768, frequency_embedding_size: int = 256, hidden_act: str = 'silu', layer_norm_eps: float = 1e-06, rms_norm_eps: float = 1e-05, max_period: float = 10000.0, flow_type: typing.Literal['lsd', 'drifting'] = 'lsd', loss_weighting_hidden_size: int = 32, loss_weighting_num_layers: int = 3)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/gradium_phonon_tts/configuration_gradium_phonon_tts.py#L33)

**Parameters:**

hidden_size (`int`, *optional*, defaults to `512`) : Dimension of the hidden representations.

num_hidden_layers (`int`, *optional*, defaults to `6`) : Number of hidden layers in the Transformer decoder.

latent_dim (`int`, *optional*, defaults to 32) : Dimension of the continuous audio latents the head predicts (the latent dimension of the audio tokenizer).

condition_dim (`int`, *optional*, defaults to 768) : Dimension of the backbone hidden state the head is conditioned on (the backbone `hidden_size`).

frequency_embedding_size (`int`, *optional*, defaults to 256) : Size of the sinusoidal embedding of the flow times.

hidden_act (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

layer_norm_eps (`float`, *optional*, defaults to 1e-06) : Epsilon of the layer normalizations inside the residual blocks and the final layer.

rms_norm_eps (`float`, *optional*, defaults to 1e-05) : Epsilon of the normalization closing the time embeddings.

max_period (`float`, *optional*, defaults to 10000.0) : Maximum period of the sinusoidal time embedding, which the checkpoint stores as `freqs`.

flow_type (`str`, *optional*, defaults to `"lsd"`) : Objective the head was trained with, which sets how it is conditioned and sampled. `"lsd"` (Lagrangian self-distillation): conditioned on the start and end times of a flow step, sampled by Euler steps from Gaussian noise. `"drifting"`: no time conditioning, the head maps noise to the latent in one call.

loss_weighting_hidden_size (`int`, *optional*, defaults to 32) : Width of the MLP that learns the uncertainty weighting of the LSD loss terms from their flow times (training only).

loss_weighting_num_layers (`int`, *optional*, defaults to 3) : Number of hidden layers of that MLP.

This is the configuration class to store the configuration of a GradiumPhononTTSModel. It is used to instantiate a Gradium Phonon Tts
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [gradium/phonon-31m-en](https://huggingface.co/gradium/phonon-31m-en)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## GradiumPhononTTSModel[[transformers.GradiumPhononTTSModel]]

#### transformers.GradiumPhononTTSModel[[transformers.GradiumPhononTTSModel]]

```python
transformers.GradiumPhononTTSModel(config: GradiumPhononTTSConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/gradium_phonon_tts/modeling_gradium_phonon_tts.py#L393)

**Parameters:**

config ([GradiumPhononTTSConfig](/docs/transformers/v5.22.1/en/model_doc/gradium_phonon_tts#transformers.GradiumPhononTTSConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Gradium Phonon TTS backbone: a causal transformer over one stream made of the voice prompt (speaker-codec
latents), the text tokens and the audio latents, returning the normalised hidden states.

This model inherits from [PreTrainedModel](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.GradiumPhononTTSModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, audio_prompt_latents: typing.Optional[torch.FloatTensor] = None, audio_prompt_latents_mask: typing.Optional[torch.Tensor] = None, audio_latents: typing.Optional[torch.FloatTensor] = None, audio_latents_mask: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/gradium_phonon_tts/modeling_gradium_phonon_tts.py#L417)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.22.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.22.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.22.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

audio_prompt_latents (`torch.FloatTensor` of shape `(batch_size, num_prompt_frames, latent_dim)`, *optional*) : Voice-prompt latents of the audio tokenizer (encoded by the processor, or a checkpoint's `voices/`), right-padded to the longest voice of the batch, scattered into the `audio_prompt_token_id` positions.

audio_prompt_latents_mask (`torch.Tensor` of shape `(batch_size, num_prompt_frames)`, *optional*) : Mask of the voice frames (`1`) as opposed to padding (`0`) in `audio_prompt_latents`.

audio_latents (`torch.FloatTensor` of shape `(batch_size, num_latents, latent_dim)`, *optional*) : Normalised audio latents, scattered into the `audio_latent_token_id` positions. The `audio_bos_token_id` position before them, which closes the prompt and from which the first frame is generated, always gets the learnt BOS latent.

audio_latents_mask (`torch.Tensor` of shape `(batch_size, num_latents)`, *optional*) : Mask of each row's latents (`1`) as opposed to padding (`0`) in `audio_latents`, when the rows hold different numbers of latent positions (training).

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.22.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.22.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

**Returns:** [BaseModelOutputWithPast](/docs/transformers/v5.22.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPast](/docs/transformers/v5.22.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([GradiumPhononTTSConfig](/docs/transformers/v5.22.1/en/model_doc/gradium_phonon_tts#transformers.GradiumPhononTTSConfig)) and inputs.

The [GradiumPhononTTSModel](/docs/transformers/v5.22.1/en/model_doc/gradium_phonon_tts#transformers.GradiumPhononTTSModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.22.1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

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

## GradiumPhononTTSForConditionalGeneration[[transformers.GradiumPhononTTSForConditionalGeneration]]

#### transformers.GradiumPhononTTSForConditionalGeneration[[transformers.GradiumPhononTTSForConditionalGeneration]]

```python
transformers.GradiumPhononTTSForConditionalGeneration(config: GradiumPhononTTSConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/gradium_phonon_tts/modeling_gradium_phonon_tts.py#L693)

**Parameters:**

config ([GradiumPhononTTSConfig](/docs/transformers/v5.22.1/en/model_doc/gradium_phonon_tts#transformers.GradiumPhononTTSConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Gradium Phonon TTS text-to-speech model: the backbone, the flow head that samples one audio latent per step and
the end-of-speech head. It generates audio tokenizer latents, which the processor decodes into 24 kHz audio.

This model inherits from [PreTrainedModel](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.GradiumPhononTTSForConditionalGeneration.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, audio_prompt_latents: typing.Optional[torch.FloatTensor] = None, audio_prompt_latents_mask: typing.Optional[torch.Tensor] = None, audio_latents: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.FloatTensor] = None, labels_mask: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/gradium_phonon_tts/modeling_gradium_phonon_tts.py#L725)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.22.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.22.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.22.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

audio_prompt_latents (`torch.FloatTensor` of shape `(batch_size, num_prompt_frames, latent_dim)`, *optional*) : Voice-prompt latents of the audio tokenizer, right-padded, scattered into the `audio_prompt_token_id` positions.

audio_prompt_latents_mask (`torch.Tensor` of shape `(batch_size, num_prompt_frames)`, *optional*) : Mask of the voice frames (`1`) as opposed to padding (`0`) in `audio_prompt_latents`.

audio_latents (`torch.FloatTensor` of shape `(batch_size, num_latents, latent_dim)`, *optional*) : Normalised audio latents, scattered into the `audio_latent_token_id` positions.

labels (`torch.FloatTensor` of shape `(batch_size, num_frames, latent_dim)`, *optional*) : Target latents of the audio tokenizer (training), right-padded. Each row's prompt ends with the `audio_bos_token_id` position followed by one `audio_latent_token_id` position per frame: the model feeds the normalised targets (teacher forcing), applies the flow loss to the outputs of the BOS and all but the last frame, which predict the frames, and the end-of-speech loss to all of them, the last one being the end of speech.

labels_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*) : Mask of each row's target frames (`1`) as opposed to padding (`0`) in `labels`.

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.22.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.22.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

**Returns:** `GradiumPhononTTSOutputWithPast` or `tuple(torch.FloatTensor)`

A `GradiumPhononTTSOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([GradiumPhononTTSConfig](/docs/transformers/v5.22.1/en/model_doc/gradium_phonon_tts#transformers.GradiumPhononTTSConfig)) and inputs.

The [GradiumPhononTTSForConditionalGeneration](/docs/transformers/v5.22.1/en/model_doc/gradium_phonon_tts#transformers.GradiumPhononTTSForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `()`, *optional*, returned when `labels` is passed) -- Training loss: the LSD flow loss of the target latents plus `config.eos_loss_weight` times the end-of-speech loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`) -- End-of-speech logits, one per position.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.22.1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Normalised hidden states of the backbone, which condition the flow head.

Example:

```python
>>> from transformers import AutoProcessor, GradiumPhononTTSForConditionalGeneration

>>> model = GradiumPhononTTSForConditionalGeneration.from_pretrained("gradium/phonon-31m-en")

>>> processor = AutoProcessor.from_pretrained("gradium/phonon-31m-en")
>>> inputs = processor(text="Hello, my dog is cute", return_tensors="pt")

>>> # generate speech
>>> speech = model(inputs["input_ids"])
```

#### generate[[transformers.GradiumPhononTTSForConditionalGeneration.generate]]

```python
generate(inputs: typing.Optional[torch.Tensor] = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/pocket_tts/generation_pocket_tts.py#L122)

**Parameters:**

inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : The `input_ids` prepared by [PocketTTSProcessor](/docs/transformers/v5.22.1/en/model_doc/pocket_tts#transformers.PocketTTSProcessor). Can also be passed as `input_ids=`.

kwargs (`dict[str, Any]`, *optional*) : `attention_mask`, the voice prompt (`audio_prompt_latents` and `audio_prompt_latents_mask`, or the cache of a voice prefix as `past_key_values`, the shipped voices, with prompts of the text only), `frames_after_eos` from the processor, the generation parameters (`do_sample`, `temperature`, `max_new_tokens`, ...), and the Pocket TTS-specific `num_flow_steps`, `eos_threshold` and `min_frames_before_eos`, which default to the generation config.

**Returns:**

[PocketTTSGenerateOutput](/docs/transformers/v5.22.1/en/model_doc/pocket_tts#transformers.models.pocket_tts.generation_pocket_tts.PocketTTSGenerateOutput) when `return_dict_in_generate=True` (the
checkpoints' generation config): the padded `audio_latents` and their `audio_latents_mask`, which
[PocketTTSProcessor.decode()](/docs/transformers/v5.22.1/en/model_doc/pocket_tts#transformers.PocketTTSProcessor.decode) decodes as one batch. Otherwise the list of generated latents, one
`(num_frames, latent_dim)` tensor per sample, each decoded on its own.

Generates speech. This method overrides [generate()](/docs/transformers/v5.22.1/en/main_classes/text_generation#transformers.GenerationMixin.generate) to default
`max_new_tokens` to the original runtime's frame budget, `ceil((num_text_tokens / 3 + 2) * frame_rate)` for the
longest text of the batch. Everything else goes through the regular generation hooks.

### CohereAsr
https://huggingface.co/docs/transformers/v5.22.1/model_doc/cohere_asr.md
