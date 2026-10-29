# Pocket TTS

## Overview

Pocket TTS is [Kyutai](https://kyutai.org)'s 100M-parameter text-to-speech model, a [Continuous Audio Language Model](https://huggingface.co/papers/2509.06926) that runs faster than real time on CPU. A causal transformer emits one continuous audio latent per frame (12.5 Hz) through a one-step flow head, and an end-of-speech head decides when to stop.

The processor holds the [Pocket TTS audio tokenizer](./pocket_tts_audio_tokenizer), a Mimi-based codec that encodes voice prompts and decodes generated latents to 24 kHz audio. Predefined voices ship as KV caches (`voices/`).

Two flow heads are released: Lagrangian self-distillation (`flow_type="lsd"`, default) and drifting (`flow_type="drifting"`, `english_drifting_26-09`). Finetuning is supported, see [Training](#training).

This model was contributed by [Eustache Le Bihan](https://huggingface.co/eustlb). The original code is [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts).

## Usage tips

- Prepare inputs with [apply_chat_template()](/docs/transformers/v5.21.1/en/main_classes/processors#transformers.ProcessorMixin.apply_chat_template) (one `user` message per text, `add_generation_prompt=True`). The normalizer of the tokenizer prepares the text like the original, with the settings of the checkpoint's language.
- `generate` returns `audio_latents` and `audio_latents_mask`; decode them with [PocketTTSProcessor.decode()](/docs/transformers/v5.21.1/en/model_doc/pocket_tts#transformers.PocketTTSProcessor.decode).
- `max_new_tokens` defaults to a budget based on text length, and generation stops shortly after the end-of-speech logit crosses `eos_threshold`.
- `do_sample=False` matches the original at temperature 0. With sampling, `temperature` (default 0.3) is the variance of the flow's starting noise.
- Split long text with [PocketTTSProcessor.split_text()](/docs/transformers/v5.21.1/en/model_doc/pocket_tts#transformers.PocketTTSProcessor.split_text), one conversation per chunk.

## Usage

### Voice

The voice can be given in three ways.

**1. Cloning a recording**: an `audio` item in the conversation.

```python
from transformers import AutoModelForTextToWaveform, AutoProcessor

model_id = "eustlb/pocket-tts"
processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")

url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/en-Alice_woman.wav"
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "url": url},
            {"type": "text", "text": "Hello world. I am Kyutai's Pocket TTS."},
        ],
    }
]
inputs = processor.apply_chat_template(
    conversation, tokenize=True, return_dict=True, add_generation_prompt=True
).to(model.device)

outputs = model.generate(**inputs)
audio = processor.decode(outputs.audio_latents, outputs.audio_latents_mask)
processor.save_audio(audio, "cloned.wav")
```

**2. Precomputed latents**: an empty `audio` item, with the latents passed to the processor.

```python
from transformers import AutoModelForTextToWaveform, AutoProcessor
from transformers.audio_utils import load_audio

model_id = "eustlb/pocket-tts"
processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")

# e.g. a recording encoded once by the processor's audio tokenizer,
# to reuse without re-encoding it
url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/en-Alice_woman.wav"
features = processor.feature_extractor(
    load_audio(url, sampling_rate=24000), sampling_rate=24000
).to(processor.audio_tokenizer.device)
# (num_frames, latent_dim)
voice = processor.audio_tokenizer.encode(**features).latents[0]

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio"},
            {"type": "text", "text": "Hello world. I am Kyutai's Pocket TTS."},
        ],
    }
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
processor.save_audio(audio, "latents.wav")
```

**3. Cached context**: the voice prompt's KV cache, passed as `past_key_values`. The shipped voices (`voices/`) use this form.

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForTextToWaveform, AutoProcessor, DynamicCache

model_id = "eustlb/pocket-tts"
processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")

voice = load_file(hf_hub_download(model_id, "voices/alba.safetensors"))
layers = []
for i, layer in enumerate(model.model.layers):
    # each layer's cache on that layer's device
    device = layer.self_attn.k_proj.weight.device
    layers.append(
        (voice[f"layers.{i}.keys"].to(device), voice[f"layers.{i}.values"].to(device))
    )
voice_cache = DynamicCache(layers, config=model.config)

conversation = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "Hello world. I am Kyutai's Pocket TTS."}],
    }
]
inputs = processor.apply_chat_template(
    conversation, tokenize=True, return_dict=True, add_generation_prompt=True
).to(model.device)

outputs = model.generate(**inputs, past_key_values=voice_cache)
audio = processor.decode(outputs.audio_latents, outputs.audio_latents_mask)
processor.save_audio(audio, "cached.wav")
```

### Batches and long text

Batch several texts and voices, with long text split into chunks:

```python
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForTextToWaveform, AutoProcessor, DynamicCache

model_id = "eustlb/pocket-tts"
processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")

def load_voices(names):
    """The shipped voices, one per conversation, as one cache, left-padded to the
    longest voice, and the mask of their positions.

    `generate` extends the cache: load it for every call.
    """
    voices = [
        load_file(hf_hub_download(model_id, f"voices/{name}.safetensors"))
        for name in names
    ]
    lengths = [voice["layers.0.keys"].shape[2] for voice in voices]
    length = max(lengths)
    layers = []
    for i, layer in enumerate(model.model.layers):
        device = layer.self_attn.k_proj.weight.device
        keys, values = (
            torch.cat(
                [
                    torch.nn.functional.pad(voice[f"layers.{i}.{kind}"], (0, 0, length - n, 0))
                    for voice, n in zip(voices, lengths)
                ]
            ).to(device)
            for kind in ("keys", "values")
        )
        layers.append((keys, values))
    mask = torch.tensor([[0] * (length - n) + [1] * n for n in lengths])
    return DynamicCache(layers, config=model.config), mask

text = (
    "A first sentence that is long enough to be worth splitting. "
    "Then a second one, and a third one for good measure."
)
chunks = processor.split_text(text)
conversations = [
    [{"role": "user", "content": [{"type": "text", "text": chunk}]}]
    for chunk in chunks + ["Hello world"]
]
inputs = processor.apply_chat_template(
    conversations, tokenize=True, return_dict=True, add_generation_prompt=True
).to(model.device)

voice_cache, voice_mask = load_voices(["alba"] * len(chunks) + ["jean"])
# the mask covers the cached voices too
inputs["attention_mask"] = torch.cat(
    [voice_mask.to(inputs["attention_mask"]), inputs["attention_mask"]], dim=-1
)
outputs = model.generate(**inputs, past_key_values=voice_cache)
audio = processor.decode(outputs.audio_latents, outputs.audio_latents_mask)
processor.save_audio(
    [torch.cat(audio[: len(chunks)]), audio[-1]], ["long.wav", "hello.wav"]
)
```

### Drifting checkpoint

Same usage, with the checkpoint's own voices:

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForTextToWaveform, AutoProcessor, DynamicCache

model_id = "eustlb/pocket-tts-english-drifting"
processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")

voice = load_file(hf_hub_download(model_id, "voices/alba.safetensors"))
layers = []
for i, layer in enumerate(model.model.layers):
    device = layer.self_attn.k_proj.weight.device
    layers.append(
        (voice[f"layers.{i}.keys"].to(device), voice[f"layers.{i}.values"].to(device))
    )

conversation = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "Hello world. I am Kyutai's Pocket TTS."}],
    }
]
inputs = processor.apply_chat_template(
    conversation, tokenize=True, return_dict=True, add_generation_prompt=True
).to(model.device)

outputs = model.generate(
    **inputs, past_key_values=DynamicCache(layers, config=model.config)
)
audio = processor.decode(outputs.audio_latents, outputs.audio_latents_mask)
processor.save_audio(audio, "drifting.wav")
```

## Training

Put the target audio in the `assistant` turn. The processor returns its latents as `labels`, and the model returns the original's loss (LSD flow loss plus `config.eos_loss_weight` times the end-of-speech loss). The voice prompt is typically the beginning of the same utterance.

```python
import torch
from transformers import AutoModelForTextToWaveform, AutoProcessor

model_id = "eustlb/pocket-tts"
processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
model = AutoModelForTextToWaveform.from_pretrained(model_id, device_map="auto")
model.train()

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": voice_prompt},
            {"type": "text", "text": target_text},
        ],
    },
    {"role": "assistant", "content": [{"type": "audio", "audio": target_audio}]},
]
inputs = processor.apply_chat_template(
    conversation, tokenize=True, return_dict=True
).to(model.device)
loss = model(**inputs).loss
loss.backward()
```

- Give the voice as a recording or latents, not a cache.
- For classifier-free guidance, drop the `audio` or `text` item of the `user` turn (the original drops each with probability 0.2).
- Precomputed latents go in `processor_kwargs={"audio_prompt_latents": [...], "labels": [...]}`, with empty `{"type": "audio"}` items as placeholders.
- Original recipe: AdamW, `learning_rate=2e-5` (2e-4 for a new language), `adam_beta2=0.95`, `weight_decay=0.1`, `max_grad_norm=1.0`, 1000 warmup steps then constant, EMA of the weights (decay 0.999).

## PocketTTSConfig[[transformers.PocketTTSConfig]]

#### transformers.PocketTTSConfig[[transformers.PocketTTSConfig]]

```python
transformers.PocketTTSConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, flow_head_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, vocab_size: int = 4004, hidden_size: int = 1024, num_hidden_layers: int = 6, num_attention_heads: int = 16, num_key_value_heads: int | None = None, head_dim: int | None = None, intermediate_size: int = 4096, hidden_act: str = 'gelu_pytorch_tanh', layer_norm_eps: float = 1e-05, attention_bias: bool = False, attention_dropout: float = 0.0, max_position_embeddings: int = 8192, rope_parameters: transformers.modeling_rope_utils.RopeParameters | dict | None = None, latent_dim: int = 32, pad_token_id: int = 3, bos_token_id: int | None = None, eos_token_id: int | None = None, audio_prompt_token_id: int = 4000, audio_latent_token_id: int = 4001, audio_prompt_bos_token_id: int = 4002, audio_bos_token_id: int = 4003, frame_rate: float = 12.5, initializer_range: float = 0.02, use_cache: bool = True, eos_loss_weight: float = 0.1, flow_loss_num_samples: int = 4, lsd_distill_probability: float = 0.25, lsd_diagonal_weight: float = 0.75, lsd_time_logit_mean: float = 0.4, lsd_time_logit_std: float = 1.0, latent_stats_ema_steps: int = 1000, latent_stats_ema_decay: float = 0.999)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts/configuration_pocket_tts.py#L75)

**Parameters:**

flow_head_config (`Union[PocketTTSFlowHeadConfig, dict]`, *optional*) : The config of the flow head that samples each latent from the backbone hidden state.

vocab_size (`int`, *optional*, defaults to `4004`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

hidden_size (`int`, *optional*, defaults to `1024`) : Dimension of the hidden representations.

num_hidden_layers (`int`, *optional*, defaults to `6`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `16`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

head_dim (`int`, *optional*) : The attention head dimension. If None, it will default to hidden_size // num_attention_heads

intermediate_size (`int`, *optional*, defaults to `4096`) : Dimension of the MLP representations.

hidden_act (`str`, *optional*, defaults to `gelu_pytorch_tanh`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

layer_norm_eps (`float`, *optional*, defaults to 1e-05) : Epsilon of the layer normalizations of the backbone.

attention_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

attention_dropout (`float`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

max_position_embeddings (`int`, *optional*, defaults to `8192`) : The maximum sequence length that this model might ever be used with.

rope_parameters (`Union[~modeling_rope_utils.RopeParameters, dict]`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

latent_dim (`int`, *optional*, defaults to 32) : Dimension of the audio latents the backbone takes as input (`audio_prompt_latents`, `audio_latents`); equals the flow head's and the audio tokenizer's `latent_dim`.

pad_token_id (`int`, *optional*, defaults to `3`) : Token id used for padding in the vocabulary.

bos_token_id (`int`, *optional*) : Token id used for beginning-of-stream in the vocabulary.

eos_token_id (`int`, *optional*) : Token id used for end-of-stream in the vocabulary.

audio_prompt_token_id (`int`, *optional*, defaults to 4000) : Placeholder token id of the voice-prompt frames in `input_ids`.

audio_latent_token_id (`int`, *optional*, defaults to 4001) : Placeholder token id of the audio latent positions in `input_ids`.

audio_prompt_bos_token_id (`int`, *optional*, defaults to 4002) : Token id placed before the voice prompt; its embedding is the learnt vector the original prepends to the voice.

audio_bos_token_id (`int`, *optional*, defaults to 4003) : Token id of the first audio position, closing the prompt: it holds the learnt BOS latent, from which the first frame is generated.

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

This is the configuration class to store the configuration of a PocketTTSModel. It is used to instantiate a Pocket Tts
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [eustlb/pocket-tts](https://huggingface.co/eustlb/pocket-tts)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## PocketTTSFlowHeadConfig[[transformers.PocketTTSFlowHeadConfig]]

#### transformers.PocketTTSFlowHeadConfig[[transformers.PocketTTSFlowHeadConfig]]

```python
transformers.PocketTTSFlowHeadConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, hidden_size: int = 512, num_hidden_layers: int = 6, latent_dim: int = 32, condition_dim: int = 1024, frequency_embedding_size: int = 256, hidden_act: str = 'silu', layer_norm_eps: float = 1e-06, rms_norm_eps: float = 1e-05, max_period: float = 10000.0, flow_type: typing.Literal['lsd', 'drifting'] = 'lsd', loss_weighting_hidden_size: int = 32, loss_weighting_num_layers: int = 3)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts/configuration_pocket_tts.py#L27)

**Parameters:**

hidden_size (`int`, *optional*, defaults to `512`) : Dimension of the hidden representations.

num_hidden_layers (`int`, *optional*, defaults to `6`) : Number of hidden layers in the Transformer decoder.

latent_dim (`int`, *optional*, defaults to 32) : Dimension of the continuous audio latents the head predicts (the latent dimension of the audio tokenizer).

condition_dim (`int`, *optional*, defaults to 1024) : Dimension of the backbone hidden state the head is conditioned on (the backbone `hidden_size`).

frequency_embedding_size (`int`, *optional*, defaults to 256) : Size of the sinusoidal embedding of the flow times.

hidden_act (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

layer_norm_eps (`float`, *optional*, defaults to 1e-06) : Epsilon of the layer normalizations inside the residual blocks and the final layer.

rms_norm_eps (`float`, *optional*, defaults to 1e-05) : Epsilon of the normalization closing the time embeddings.

max_period (`float`, *optional*, defaults to 10000.0) : Maximum period of the sinusoidal time embedding, which the checkpoint stores as `freqs`.

flow_type (`str`, *optional*, defaults to `"lsd"`) : Objective the head was trained with, which sets how it is conditioned and sampled. `"lsd"` (Lagrangian self-distillation): conditioned on the start and end times of a flow step, sampled by Euler steps from Gaussian noise. `"drifting"`: no time conditioning, the head maps noise to the latent in one call.

loss_weighting_hidden_size (`int`, *optional*, defaults to 32) : Width of the MLP that learns the uncertainty weighting of the LSD loss terms from their flow times (training only).

loss_weighting_num_layers (`int`, *optional*, defaults to 3) : Number of hidden layers of that MLP.

This is the configuration class to store the configuration of a PocketTTSModel. It is used to instantiate a Pocket Tts
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [eustlb/pocket-tts](https://huggingface.co/eustlb/pocket-tts)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## PocketTTSFeatureExtractor[[transformers.PocketTTSFeatureExtractor]]

#### transformers.PocketTTSFeatureExtractor[[transformers.PocketTTSFeatureExtractor]]

```python
transformers.PocketTTSFeatureExtractor(feature_size = 1, sampling_rate = 24000, padding_value = 0.0, frame_size = 1920, pad_to_multiple_of = 1920, end_on_pause = True, pause_duration = 0.08, fade_out_duration = 0.02, silence_floor_db = 35.0, analysis_frame_duration = 0.02, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts/feature_extraction_pocket_tts.py#L29)

**Parameters:**

feature_size (`int`, *optional*, defaults to 1) : The number of channels.

sampling_rate (`int`, *optional*, defaults to 24000) : The sampling rate at which the audio waveform should be digitalized, expressed in hertz (Hz).

padding_value (`float`, *optional*, defaults to 0.0) : The value that is used for padding.

frame_size (`int`, *optional*, defaults to 1920) : Number of audio samples per latent frame of the audio tokenizer (its `frame_size`).

pad_to_multiple_of (`int`, *optional*, defaults to 1920) : Pad the audio to a multiple of this value; a multiple of `frame_size` keeps every latent frame whole.

end_on_pause (`bool`, *optional*, defaults to `True`) : Whether to end each waveform on exactly `pause_duration` seconds of silence: the trailing silence (frames of `analysis_frame_duration` seconds more than `silence_floor_db` below the loudest one) is cut, the last `fade_out_duration` seconds of what remains are faded out, and `pause_duration` seconds of zeros appended.

pause_duration (`float`, *optional*, defaults to 0.08) : Duration of the silence appended by `end_on_pause`, in seconds.

fade_out_duration (`float`, *optional*, defaults to 0.02) : Duration of the fade-out applied by `end_on_pause` before the pause, in seconds.

silence_floor_db (`float`, *optional*, defaults to 35.0) : Level under the loudest analysis frame, in dB, below which `end_on_pause` treats trailing frames as silence.

analysis_frame_duration (`float`, *optional*, defaults to 0.02) : Duration of the frames `end_on_pause` measures the level of, in seconds.

Constructs a Pocket TTS feature extractor, which prepares voice-prompt waveforms for the audio tokenizer's encoder.

Each waveform optionally ends on a short pause (the original's `end_on_pause`: training prompts end inside the pause
between two words, a prompt that stops on speech makes the model continue it and one that ends on a long silence
delays the onset), then the batch is right-padded to a multiple of the tokenizer's frame size.

#### __call__[[transformers.PocketTTSFeatureExtractor.__call__]]

```python
__call__(audio: typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence['torch.Tensor']], sampling_rate: int | None = None, end_on_pause: bool | None = None, padding: bool | str | transformers.utils.generic.PaddingStrategy | None = True, pad_to_multiple_of: int | None = None, max_length: int | None = None, return_attention_mask: bool | None = True, return_tensors: str | None = 'pt', **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts/feature_extraction_pocket_tts.py#L106)

**Parameters:**

audio (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`) : The mono waveform or batch of mono waveforms of the voice prompts.

sampling_rate (`int`, *optional*) : The sampling rate at which the `audio` input was sampled. It is strongly recommended to pass `sampling_rate` at the forward call to prevent silent errors.

end_on_pause (`bool`, *optional*) : Whether to end each waveform on a short pause, see the class docstring. Defaults to `self.end_on_pause`.

padding (`bool`, `str` or [PaddingStrategy](/docs/transformers/v5.21.1/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `True`) : Padding strategy of the batch, see [pad()](/docs/transformers/v5.21.1/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor.pad).

pad_to_multiple_of (`int`, *optional*) : Pad the batch to a multiple of this value. Defaults to `self.pad_to_multiple_of`.

## PocketTTSProcessor[[transformers.PocketTTSProcessor]]

#### transformers.PocketTTSProcessor[[transformers.PocketTTSProcessor]]

```python
transformers.PocketTTSProcessor(feature_extractor, tokenizer, audio_tokenizer = None, chat_template = None, short_text_max_words: int = 4, frames_after_eos_short_text: int = 4, frames_after_eos: int = 2, fade_in_duration: float = 0.005)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts/processing_pocket_tts.py#L51)

**Parameters:**

feature_extractor (`PocketTTSFeatureExtractor`) : The feature extractor is a required input.

tokenizer (`TokenizersBackend`) : The tokenizer is a required input.

audio_tokenizer (`PocketTTSAudioTokenizerModel`, *optional*) : The codec: it encodes audio into latents, and decodes generated latents into audio.

chat_template (`str`) : A Jinja template to convert lists of messages in a chat into a tokenizable string.

short_text_max_words (`int`, *optional*, defaults to 4) : Texts of at most this many words get `frames_after_eos_short_text` frames after the end-of-speech frame.

frames_after_eos_short_text (`int`, *optional*, defaults to 4) : Number of frames generated after the end-of-speech frame for short texts.

frames_after_eos (`int`, *optional*, defaults to 2) : Number of frames generated after the end-of-speech frame for longer texts.

fade_in_duration (`float`, *optional*, defaults to 0.005) : Duration in seconds of the linear fade-in applied to the start of each decoded waveform: a fresh decoder state starts with a small step, heard as a click.

Constructs a PocketTTSProcessor which wraps a feature extractor and a tokenizer into a single processor.

[PocketTTSProcessor](/docs/transformers/v5.21.1/en/model_doc/pocket_tts#transformers.PocketTTSProcessor) offers all the functionalities of [PocketTTSFeatureExtractor](/docs/transformers/v5.21.1/en/model_doc/pocket_tts#transformers.PocketTTSFeatureExtractor) and [TokenizersBackend](/docs/transformers/v5.21.1/en/main_classes/tokenizer#transformers.TokenizersBackend). See the
[~PocketTTSFeatureExtractor](/docs/transformers/v5.21.1/en/model_doc/pocket_tts#transformers.PocketTTSFeatureExtractor) and [~TokenizersBackend](/docs/transformers/v5.21.1/en/main_classes/tokenizer#transformers.TokenizersBackend) for more information.

#### __call__[[transformers.PocketTTSProcessor.__call__]]

```python
__call__(text: str | list[str] | None = None, audio: typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence['torch.Tensor'], NoneType] = None, audio_prompt_latents: list[torch.Tensor] | None = None, labels: list[torch.Tensor] | None = None, **kwargs: *<class 'transformers.models.pocket_tts.processing_pocket_tts.PocketTTSProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts/processing_pocket_tts.py#L109)

**Parameters:**

text (`str` or `list[str]`) : The prompt(s) rendered by [apply_chat_template()](/docs/transformers/v5.21.1/en/main_classes/processors#transformers.ProcessorMixin.apply_chat_template).

audio (`np.ndarray`, `torch.Tensor`, or a list of them, *optional*) : The waveforms of the conversations, mono at 24 kHz, in conversation order: each voice, and in training each target.

audio_prompt_latents (`list[torch.Tensor]`, *optional*) : Precomputed voices, one `(num_frames, latent_dim)` tensor per voice placeholder, instead of waveforms.

labels (`list[torch.Tensor]`, *optional*) : Precomputed targets (training), one `(num_frames, latent_dim)` tensor per target placeholder, instead of waveforms.

end_on_pause (`bool`, *kwargs*, *optional*) : Whether to end each voice-prompt waveform on a short pause, as the original does before cloning a voice. Defaults to the feature extractor's `end_on_pause`.

return_tensors (`str` or [TensorType](/docs/transformers/v5.21.1/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

- ****kwargs** ([ProcessingKwargs](/docs/transformers/v5.21.1/en/main_classes/processors#transformers.ProcessingKwargs), *optional*) : Additional processing options for each modality (text, images, videos, audio). Model-specific parameters are listed above; see the TypedDict class for the complete list of supported arguments.

**Returns:**

[BatchFeature](/docs/transformers/v5.21.1/en/main_classes/feature_extractor#transformers.BatchFeature) with `input_ids`, `attention_mask`, `audio_prompt_latents` and `audio_prompt_latents_mask`;
for generation `frames_after_eos` (frames generated after the end-of-speech frame, by text length), for
training `labels` and `labels_mask`.

#### split_text[[transformers.PocketTTSProcessor.split_text]]

```python
split_text(text: str, max_tokens: int = 50)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts/processing_pocket_tts.py#L282)

**Parameters:**

text (`str`) : The text to split.

max_tokens (`int`, *optional*, defaults to 50) : Maximum number of tokens per chunk.

Splits `text` into sentence-aligned chunks of at most `max_tokens` tokens, as the original's
`split_into_best_sentences`: the prepared text is cut after sentence ends (not on decimal points), sentences
longer than `max_tokens` after clause ends, and the pieces are greedily grouped. Each chunk is meant to be one
conversation of a batch.

#### decode[[transformers.PocketTTSProcessor.decode]]

```python
decode(audio_latents: torch.Tensor, audio_latents_mask: torch.Tensor | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts/processing_pocket_tts.py#L340)

**Parameters:**

audio_latents (`torch.Tensor` of shape `(batch_size, num_frames, latent_dim)` or `(num_frames, latent_dim)`) : The latents `generate` returns (`PocketTTSGenerateOutput.audio_latents`), decoded as one batch, or one sample's latents.

audio_latents_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*) : Mask of each sample's frames (`PocketTTSGenerateOutput.audio_latents_mask`). Without it, every row is decoded at full length.

**Returns:**

`list[torch.Tensor]` of 1D waveforms on CPU, one per row, or one waveform for one sample's latents.

Decodes generated latents into waveforms with the audio tokenizer, and fades the first `fade_in_duration` seconds
of each in, as the original does.

#### save_audio[[transformers.PocketTTSProcessor.save_audio]]

```python
save_audio(audio: typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence['torch.Tensor']], saving_path: str | pathlib.Path | list[str | pathlib.Path], sampling_rate: int | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts/processing_pocket_tts.py#L389)

Writes the generated waveform(s) to WAV file(s).

audio (`torch.Tensor`, `np.ndarray` or a list of them):
The waveform(s), as returned by [decode()](/docs/transformers/v5.21.1/en/model_doc/pocket_tts#transformers.PocketTTSProcessor.decode).
saving_path (`str`, `Path` or a list of them):
One path per waveform.
sampling_rate (`int`, *optional*):
Sampling rate of the waveforms, the feature extractor's (the audio tokenizer's) by default.

## PocketTTSModel[[transformers.PocketTTSModel]]

#### transformers.PocketTTSModel[[transformers.PocketTTSModel]]

```python
transformers.PocketTTSModel(config: PocketTTSConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts/modeling_pocket_tts.py#L498)

**Parameters:**

config ([PocketTTSConfig](/docs/transformers/v5.21.1/en/model_doc/pocket_tts#transformers.PocketTTSConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Pocket TTS backbone: a causal transformer over one stream made of the voice prompt, the text tokens and the audio
latents, returning the normalised hidden states. The voice prompt and the audio latents are latents of the audio
tokenizer, which the processor holds.

This model inherits from [PreTrainedModel](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.PocketTTSModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, audio_prompt_latents: typing.Optional[torch.FloatTensor] = None, audio_prompt_latents_mask: typing.Optional[torch.Tensor] = None, audio_latents: typing.Optional[torch.FloatTensor] = None, audio_latents_mask: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts/modeling_pocket_tts.py#L518)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.21.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

audio_prompt_latents (`torch.FloatTensor` of shape `(batch_size, num_prompt_frames, latent_dim)`, *optional*) : Voice-prompt latents of the audio tokenizer (encoded by the processor, or a checkpoint's `voices/`), right-padded to the longest voice of the batch, scattered into the `audio_prompt_token_id` positions.

audio_prompt_latents_mask (`torch.Tensor` of shape `(batch_size, num_prompt_frames)`, *optional*) : Mask of the voice frames (`1`) as opposed to padding (`0`) in `audio_prompt_latents`.

audio_latents (`torch.FloatTensor` of shape `(batch_size, num_latents, latent_dim)`, *optional*) : Normalised audio latents, scattered into the `audio_latent_token_id` positions. The `audio_bos_token_id` position before them, which closes the prompt and from which the first frame is generated, always gets the learnt BOS latent.

audio_latents_mask (`torch.Tensor` of shape `(batch_size, num_latents)`, *optional*) : Mask of each row's latents (`1`) as opposed to padding (`0`) in `audio_latents`, when the rows hold different numbers of latent positions (training).

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.21.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.21.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

**Returns:** [BaseModelOutputWithPast](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPast](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PocketTTSConfig](/docs/transformers/v5.21.1/en/model_doc/pocket_tts#transformers.PocketTTSConfig)) and inputs.

The [PocketTTSModel](/docs/transformers/v5.21.1/en/model_doc/pocket_tts#transformers.PocketTTSModel) forward method, overrides the `__call__` special method.

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

## PocketTTSForConditionalGeneration[[transformers.PocketTTSForConditionalGeneration]]

#### transformers.PocketTTSForConditionalGeneration[[transformers.PocketTTSForConditionalGeneration]]

```python
transformers.PocketTTSForConditionalGeneration(config: PocketTTSConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts/modeling_pocket_tts.py#L672)

**Parameters:**

config ([PocketTTSConfig](/docs/transformers/v5.21.1/en/model_doc/pocket_tts#transformers.PocketTTSConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Pocket TTS text-to-speech model: the backbone, the flow head that samples one audio latent per step and the
end-of-speech head. It generates audio tokenizer latents, which the processor decodes into 24 kHz audio.

This model inherits from [PreTrainedModel](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.PocketTTSForConditionalGeneration.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, audio_prompt_latents: typing.Optional[torch.FloatTensor] = None, audio_prompt_latents_mask: typing.Optional[torch.Tensor] = None, audio_latents: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.FloatTensor] = None, labels_mask: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts/modeling_pocket_tts.py#L702)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.21.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

audio_prompt_latents (`torch.FloatTensor` of shape `(batch_size, num_prompt_frames, latent_dim)`, *optional*) : Voice-prompt latents of the audio tokenizer, right-padded, scattered into the `audio_prompt_token_id` positions.

audio_prompt_latents_mask (`torch.Tensor` of shape `(batch_size, num_prompt_frames)`, *optional*) : Mask of the voice frames (`1`) as opposed to padding (`0`) in `audio_prompt_latents`.

audio_latents (`torch.FloatTensor` of shape `(batch_size, num_latents, latent_dim)`, *optional*) : Normalised audio latents, scattered into the `audio_latent_token_id` positions.

labels (`torch.FloatTensor` of shape `(batch_size, num_frames, latent_dim)`, *optional*) : Target latents of the audio tokenizer (training), right-padded. Each row's prompt ends with the `audio_bos_token_id` position followed by one `audio_latent_token_id` position per frame: the model feeds the normalised targets (teacher forcing), applies the flow loss to the outputs of the BOS and all but the last frame, which predict the frames, and the end-of-speech loss to all of them, the last one being the end of speech.

labels_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*) : Mask of each row's target frames (`1`) as opposed to padding (`0`) in `labels`.

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.21.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.21.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

**Returns:** `PocketTTSOutputWithPast` or `tuple(torch.FloatTensor)`

A `PocketTTSOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PocketTTSConfig](/docs/transformers/v5.21.1/en/model_doc/pocket_tts#transformers.PocketTTSConfig)) and inputs.

The [PocketTTSForConditionalGeneration](/docs/transformers/v5.21.1/en/model_doc/pocket_tts#transformers.PocketTTSForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `()`, *optional*, returned when `labels` is passed) -- Training loss: the LSD flow loss of the target latents plus `config.eos_loss_weight` times the end-of-speech loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`) -- End-of-speech logits, one per position.
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
- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Normalised hidden states of the backbone, which condition the flow head.

Example:

```python
>>> from transformers import AutoProcessor, PocketTTSForConditionalGeneration

>>> model = PocketTTSForConditionalGeneration.from_pretrained("eustlb/pocket-tts")

>>> processor = AutoProcessor.from_pretrained("eustlb/pocket-tts")
>>> inputs = processor(text="Hello, my dog is cute", return_tensors="pt")

>>> # generate speech
>>> speech = model(inputs["input_ids"])
```

#### generate[[transformers.PocketTTSForConditionalGeneration.generate]]

```python
generate(inputs: typing.Optional[torch.Tensor] = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts/generation_pocket_tts.py#L122)

**Parameters:**

inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : The `input_ids` prepared by [PocketTTSProcessor](/docs/transformers/v5.21.1/en/model_doc/pocket_tts#transformers.PocketTTSProcessor). Can also be passed as `input_ids=`.

kwargs (`dict[str, Any]`, *optional*) : `attention_mask`, the voice prompt (`audio_prompt_latents` and `audio_prompt_latents_mask`, or the cache of a voice prefix as `past_key_values`, the shipped voices, with prompts of the text only), `frames_after_eos` from the processor, the generation parameters (`do_sample`, `temperature`, `max_new_tokens`, ...), and the Pocket TTS-specific `num_flow_steps`, `eos_threshold` and `min_frames_before_eos`, which default to the generation config.

**Returns:**

[PocketTTSGenerateOutput](/docs/transformers/v5.21.1/en/model_doc/pocket_tts#transformers.models.pocket_tts.generation_pocket_tts.PocketTTSGenerateOutput) when `return_dict_in_generate=True` (the
checkpoints' generation config): the padded `audio_latents` and their `audio_latents_mask`, which
[PocketTTSProcessor.decode()](/docs/transformers/v5.21.1/en/model_doc/pocket_tts#transformers.PocketTTSProcessor.decode) decodes as one batch. Otherwise the list of generated latents, one
`(num_frames, latent_dim)` tensor per sample, each decoded on its own.

Generates speech. This method overrides [generate()](/docs/transformers/v5.21.1/en/main_classes/text_generation#transformers.GenerationMixin.generate) to default
`max_new_tokens` to the original runtime's frame budget, `ceil((num_text_tokens / 3 + 2) * frame_rate)` for the
longest text of the batch. Everything else goes through the regular generation hooks.

#### sample_latent[[transformers.PocketTTSForConditionalGeneration.sample_latent]]

```python
sample_latent(condition: <module 'torch.Tensor' (<doc_builder.mock_imports.MockFinder object at 0x7fef3bad2690>)>, noise: <module 'torch.Tensor' (<doc_builder.mock_imports.MockFinder object at 0x7fef3bad2690>)>, num_flow_steps: int)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts/modeling_pocket_tts.py#L807)

**Parameters:**

condition (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) : Backbone hidden state of the position being generated.

noise (`torch.FloatTensor` of shape `(batch_size, latent_dim)`) : Gaussian noise scaled by `sqrt(temperature)`, or zeros.

num_flow_steps (`int`) : Number of Euler steps of the LSD sampler (ignored by the drifting head, which takes one call).

**Returns:** `torch.FloatTensor` of shape `(batch_size, latent_dim)`

the sampled latent, in the normalised space.

Samples the next latent from `noise` with the flow head.

#### denormalize_latents[[transformers.PocketTTSForConditionalGeneration.denormalize_latents]]

```python
denormalize_latents(audio_latents: <module 'torch.Tensor' (<doc_builder.mock_imports.MockFinder object at 0x7fef3bad2690>)>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts/modeling_pocket_tts.py#L854)

Maps latents from the flow head's normalised space to the audio tokenizer's.

## PocketTTSGenerateOutput[[transformers.models.pocket_tts.generation_pocket_tts.PocketTTSGenerateOutput]]

#### transformers.models.pocket_tts.generation_pocket_tts.PocketTTSGenerateOutput[[transformers.models.pocket_tts.generation_pocket_tts.PocketTTSGenerateOutput]]

```python
transformers.models.pocket_tts.generation_pocket_tts.PocketTTSGenerateOutput(sequences: LongTensor, scores: tuple[torch.FloatTensor] | None = None, logits: tuple[torch.FloatTensor] | None = None, attentions: tuple[tuple[torch.FloatTensor]] | None = None, hidden_states: tuple[tuple[torch.FloatTensor]] | None = None, past_key_values: transformers.cache_utils.Cache | None = None, audio_latents: typing.Optional[torch.FloatTensor] = None, audio_latents_mask: typing.Optional[torch.LongTensor] = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts/generation_pocket_tts.py#L40)

**Parameters:**

sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`) : The prompt token ids followed by one `audio_latent_token_id` per generated frame. The audio itself is in `audio_latents`.

scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`) : The end-of-speech logit of every generated frame, one tensor of shape `(batch_size,)` per frame.

logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`) : Same as `scores`: Pocket TTS has no vocabulary, so there is nothing to process between the two.

attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`) : Tuple (one element for each generated frame) of tuples (one element for each layer) of `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.

hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`) : Tuple (one element for each generated frame) of tuples (one element for each layer) of `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.

past_key_values (`Cache`, *optional*, returned when `use_cache=True`) : The key/value cache of the backbone at the end of generation.

audio_latents (`torch.FloatTensor` of shape `(batch_size, num_frames, latent_dim)`) : The generated latents, in the audio tokenizer's latent space, right-padded with zeros past each sample's own frames. [PocketTTSProcessor.decode()](/docs/transformers/v5.21.1/en/model_doc/pocket_tts#transformers.PocketTTSProcessor.decode) turns them into waveforms.

audio_latents_mask (`torch.LongTensor` of shape `(batch_size, num_frames)`) : Mask of each sample's frames (`1`), the end-of-speech frame and the frames after it included, as opposed to padding (`0`).

Outputs of `PocketTTSForConditionalGeneration.generate`.

### DPR
https://huggingface.co/docs/transformers/v5.21.1/model_doc/dpr.md
