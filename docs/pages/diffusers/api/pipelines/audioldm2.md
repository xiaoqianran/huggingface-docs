# AudioLDM 2

AudioLDM 2 was proposed in [AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://huggingface.co/papers/2308.05734) by Haohe Liu et al. AudioLDM 2 takes a text prompt as input and predicts the corresponding audio. It can generate text-conditional sound effects, human speech and music.

Inspired by [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview), AudioLDM 2 is a text-to-audio _latent diffusion model (LDM)_ that learns continuous audio representations from text embeddings. Two text encoder models are used to compute the text embeddings from a prompt input: the text-branch of [CLAP](https://huggingface.co/docs/transformers/main/en/model_doc/clap) and the encoder of [Flan-T5](https://huggingface.co/docs/transformers/main/en/model_doc/flan-t5). These text embeddings are then projected to a shared embedding space by an [AudioLDM2ProjectionModel](https://huggingface.co/docs/diffusers/main/api/pipelines/audioldm2#diffusers.AudioLDM2ProjectionModel). A [GPT2](https://huggingface.co/docs/transformers/main/en/model_doc/gpt2) _language model (LM)_ is used to auto-regressively predict eight new embedding vectors, conditional on the projected CLAP and Flan-T5 embeddings. The generated embedding vectors and Flan-T5 text embeddings are used as cross-attention conditioning in the LDM. The [UNet](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm2#diffusers.AudioLDM2UNet2DConditionModel) of AudioLDM 2 is unique in the sense that it takes **two** cross-attention embeddings, as opposed to one cross-attention conditioning, as in most other LDMs.

The abstract of the paper is the following:

*Although audio generation shares commonalities across different types of audio, such as speech, music, and sound effects, designing models for each type requires careful consideration of specific objectives and biases that can significantly differ from those of other types. To bring us closer to a unified perspective of audio generation, this paper proposes a framework that utilizes the same learning method for speech, music, and sound effect generation. Our framework introduces a general representation of audio, called "language of audio" (LOA). Any audio can be translated into LOA based on AudioMAE, a self-supervised pre-trained representation learning model. In the generation process, we translate any modalities into LOA by using a GPT-2 model, and we perform self-supervised audio generation learning with a latent diffusion model conditioned on LOA. The proposed framework naturally brings advantages such as in-context learning abilities and reusable self-supervised pretrained AudioMAE and latent diffusion models. Experiments on the major benchmarks of text-to-audio, text-to-music, and text-to-speech demonstrate state-of-the-art or competitive performance against previous approaches. Our code, pretrained model, and demo are available at [this https URL](https://audioldm.github.io/audioldm2).*

This pipeline was contributed by [sanchit-gandhi](https://huggingface.co/sanchit-gandhi) and [Nguyễn Công Tú Anh](https://github.com/tuanh123789). The original codebase can be
found at [haoheliu/audioldm2](https://github.com/haoheliu/audioldm2).

## Tips

### Choosing a checkpoint

AudioLDM2 comes in three variants. Two of these checkpoints are applicable to the general task of text-to-audio generation. The third checkpoint is trained exclusively on text-to-music generation.

All checkpoints share the same model size for the text encoders and VAE. They differ in the size and depth of the UNet.
See table below for details on the three checkpoints:

| Checkpoint                                                      | Task          | UNet Model Size | Total Model Size | Training Data / h |
|-----------------------------------------------------------------|---------------|-----------------|------------------|-------------------|
| [audioldm2](https://huggingface.co/cvssp/audioldm2)             | Text-to-audio | 350M            | 1.1B             | 1150k             |
| [audioldm2-large](https://huggingface.co/cvssp/audioldm2-large) | Text-to-audio | 750M            | 1.5B             | 1150k             |
| [audioldm2-music](https://huggingface.co/cvssp/audioldm2-music) | Text-to-music | 350M            | 1.1B             | 665k              |
| [audioldm2-gigaspeech](https://huggingface.co/anhnct/audioldm2_gigaspeech) | Text-to-speech | 350M            | 1.1B             |10k              |
| [audioldm2-ljspeech](https://huggingface.co/anhnct/audioldm2_ljspeech) | Text-to-speech | 350M            | 1.1B             |              |

### Constructing a prompt

* Descriptive prompt inputs work best: use adjectives to describe the sound (e.g. "high quality" or "clear") and make the prompt context specific (e.g. "water stream in a forest" instead of "stream").
* It's best to use general terms like "cat" or "dog" instead of specific names or abstract objects the model may not be familiar with.
* Using a **negative prompt** can significantly improve the quality of the generated waveform, by guiding the generation away from terms that correspond to poor quality audio. Try using a negative prompt of "Low quality."

### Controlling inference

* The _quality_ of the predicted audio sample can be controlled by the `num_inference_steps` argument; higher steps give higher quality audio at the expense of slower inference.
* The _length_ of the predicted audio sample can be controlled by varying the `audio_length_in_s` argument.

### Evaluating generated waveforms:

* The quality of the generated waveforms can vary significantly based on the seed. Try generating with different seeds until you find a satisfactory generation.
* Multiple waveforms can be generated in one go: set `num_waveforms_per_prompt` to a value greater than 1. Automatic scoring will be performed between the generated waveforms and prompt text, and the audios ranked from best to worst accordingly.

The following example demonstrates how to construct good music and speech generation using the aforementioned tips: [example](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm2#diffusers.AudioLDM2Pipeline.__call__.example).

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

## AudioLDM2Pipeline[[diffusers.AudioLDM2Pipeline]]

#### diffusers.AudioLDM2Pipeline[[diffusers.AudioLDM2Pipeline]]

```python
diffusers.AudioLDM2Pipeline(vae: AutoencoderKL, text_encoder: ClapModel, text_encoder_2: transformers.models.t5.modeling_t5.T5EncoderModel | transformers.models.vits.modeling_vits.VitsModel, projection_model: AudioLDM2ProjectionModel, language_model: GPT2LMHeadModel, tokenizer: transformers.models.roberta.tokenization_roberta.RobertaTokenizer | transformers.models.roberta.tokenization_roberta_old.RobertaTokenizerFast, tokenizer_2: transformers.models.t5.tokenization_t5.T5Tokenizer | transformers.models.vits.tokenization_vits.VitsTokenizer, feature_extractor: ClapFeatureExtractor, unet: AudioLDM2UNet2DConditionModel, scheduler: KarrasDiffusionSchedulers, vocoder: SpeechT5HifiGan)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/audioldm2/pipeline_audioldm2.py#L149)

**Parameters:**

vae ([AutoencoderKL](/docs/diffusers/v0.40.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.

text_encoder ([ClapModel](https://huggingface.co/docs/transformers/v5.15.1/en/model_doc/clap#transformers.ClapModel)) : First frozen text-encoder. AudioLDM2 uses the joint audio-text embedding model [CLAP](https://huggingface.co/docs/transformers/model_doc/clap#transformers.CLAPTextModelWithProjection), specifically the [laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused) variant. The text branch is used to encode the text prompt to a prompt embedding. The full audio-text model is used to rank generated waveforms against the text prompt by computing similarity scores.

text_encoder_2 ([`~transformers.T5EncoderModel`, `~transformers.VitsModel`]) : Second frozen text-encoder. AudioLDM2 uses the encoder of [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the [google/flan-t5-large](https://huggingface.co/google/flan-t5-large) variant. Second frozen text-encoder use for TTS. AudioLDM2 uses the encoder of [Vits](https://huggingface.co/docs/transformers/model_doc/vits#transformers.VitsModel).

projection_model ([AudioLDM2ProjectionModel](/docs/diffusers/v0.40.0/en/api/pipelines/audioldm2#diffusers.AudioLDM2ProjectionModel)) : A trained model used to linearly project the hidden-states from the first and second text encoder models and insert learned SOS and EOS token embeddings. The projected hidden-states from the two text encoders are concatenated to give the input to the language model. A Learned Position Embedding for the Vits hidden-states

language_model ([GPT2Model](https://huggingface.co/docs/transformers/v5.15.1/en/model_doc/gpt2#transformers.GPT2Model)) : An auto-regressive language model used to generate a sequence of hidden-states conditioned on the projected outputs from the two text encoders.

tokenizer ([RobertaTokenizer](https://huggingface.co/docs/transformers/v5.15.1/en/model_doc/roberta#transformers.RobertaTokenizer)) : Tokenizer to tokenize text for the first frozen text-encoder.

tokenizer_2 ([`~transformers.T5Tokenizer`, `~transformers.VitsTokenizer`]) : Tokenizer to tokenize text for the second frozen text-encoder.

feature_extractor ([ClapFeatureExtractor](https://huggingface.co/docs/transformers/v5.15.1/en/model_doc/clap#transformers.ClapFeatureExtractor)) : Feature extractor to pre-process generated audio waveforms to log-mel spectrograms for automatic scoring.

unet ([UNet2DConditionModel](/docs/diffusers/v0.40.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel)) : A `UNet2DConditionModel` to denoise the encoded audio latents.

scheduler ([SchedulerMixin](/docs/diffusers/v0.40.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `unet` to denoise the encoded audio latents. Can be one of [DDIMScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), [LMSDiscreteScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/lms_discrete#diffusers.LMSDiscreteScheduler), or [PNDMScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/pndm#diffusers.PNDMScheduler).

vocoder ([SpeechT5HifiGan](https://huggingface.co/docs/transformers/v5.15.1/en/model_doc/speecht5#transformers.SpeechT5HifiGan)) : Vocoder of class `SpeechT5HifiGan` to convert the mel-spectrogram latents to the final audio waveform.

Pipeline for text-to-audio generation using AudioLDM2.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### __call__[[diffusers.AudioLDM2Pipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, transcription: str | list[str] = None, audio_length_in_s: float | None = None, num_inference_steps: int = 200, guidance_scale: float = 3.5, negative_prompt: str | list[str] | None = None, num_waveforms_per_prompt: int | None = 1, eta: float = 0.0, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, generated_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_generated_prompt_embeds: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.LongTensor] = None, negative_attention_mask: typing.Optional[torch.LongTensor] = None, max_new_tokens: int | None = None, return_dict: bool = True, callback: typing.Optional[typing.Callable[[int, int, torch.Tensor], NoneType]] = None, callback_steps: int | None = 1, cross_attention_kwargs: dict[str, typing.Any] | None = None, output_type: str | None = 'np')
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/audioldm2/pipeline_audioldm2.py#L841)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide audio generation. If not defined, you need to pass `prompt_embeds`.

transcription (`str` or `list[str]`, *optional*) --\ The transcript for text to speech.

audio_length_in_s (`int`, *optional*, defaults to 10.24) : The length of the generated audio sample in seconds.

num_inference_steps (`int`, *optional*, defaults to 200) : The number of denoising steps. More denoising steps usually lead to a higher quality audio at the expense of slower inference.

guidance_scale (`float`, *optional*, defaults to 3.5) : A higher guidance scale value encourages the model to generate audio that is closely linked to the text `prompt` at the expense of lower sound quality. Guidance scale is enabled when `guidance_scale > 1`.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide what to not include in audio generation. If not defined, you need to pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).

num_waveforms_per_prompt (`int`, *optional*, defaults to 1) : The number of waveforms to generate per prompt. If `num_waveforms_per_prompt > 1`, then automatic scoring is performed between the generated outputs and the text prompt. This scoring ranks the generated waveforms based on their cosine similarity with the text input in the joint text-audio embedding space.

eta (`float`, *optional*, defaults to 0.0) : Corresponds to parameter eta (η) from the [DDIM](https://huggingface.co/papers/2010.02502) paper. Only applies to the [DDIMScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), and is ignored in other schedulers.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for spectrogram generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor is generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.

generated_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings from the GPT2 language model. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_generated_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings from the GPT2 language model. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be computed from `negative_prompt` input argument.

attention_mask (`torch.LongTensor`, *optional*) : Pre-computed attention mask to be applied to the `prompt_embeds`. If not provided, attention mask will be computed from `prompt` input argument.

negative_attention_mask (`torch.LongTensor`, *optional*) : Pre-computed attention mask to be applied to the `negative_prompt_embeds`. If not provided, attention mask will be computed from `negative_prompt` input argument.

max_new_tokens (`int`, *optional*, defaults to None) : Number of new tokens to generate with the GPT2 language model. If not provided, number of tokens will be taken from the config of the model.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [StableDiffusionPipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/stable_diffusion/image_variation#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) instead of a plain tuple.

callback (`Callable`, *optional*) : A function that calls every `callback_steps` steps during inference. The function is called with the following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.

callback_steps (`int`, *optional*, defaults to 1) : The frequency at which the `callback` function is called. If not specified, the callback is called at every step.

cross_attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined in [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

output_type (`str`, *optional*, defaults to `"np"`) : The output format of the generated audio. Choose between `"np"` to return a NumPy `np.ndarray` or `"pt"` to return a PyTorch `torch.Tensor` object. Set to `"latent"` to return the latent diffusion model (LDM) output.

**Returns:** [StableDiffusionPipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/stable_diffusion/image_variation#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) or `tuple`

If `return_dict` is `True`, [StableDiffusionPipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/stable_diffusion/image_variation#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) is returned,
otherwise a `tuple` is returned where the first element is a list with the generated audio.

The call function to the pipeline for generation.

Examples:
```py
>>> import scipy
>>> import torch
>>> from diffusers import AudioLDM2Pipeline

>>> repo_id = "cvssp/audioldm2"
>>> pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
>>> pipe = pipe.to("cuda")

>>> # define the prompts
>>> prompt = "The sound of a hammer hitting a wooden surface."
>>> negative_prompt = "Low quality."

>>> # set the seed for generator
>>> generator = torch.Generator("cuda").manual_seed(0)

>>> # run the generation
>>> audio = pipe(
...     prompt,
...     negative_prompt=negative_prompt,
...     num_inference_steps=200,
...     audio_length_in_s=10.0,
...     num_waveforms_per_prompt=3,
...     generator=generator,
... ).audios

>>> # save the best audio sample (index 0) as a .wav file
>>> scipy.io.wavfile.write("techno.wav", rate=16000, data=audio[0])
```

```
#Using AudioLDM2 for Text To Speech
>>> import scipy
>>> import torch
>>> from diffusers import AudioLDM2Pipeline

>>> repo_id = "anhnct/audioldm2_gigaspeech"
>>> pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
>>> pipe = pipe.to("cuda")

>>> # define the prompts
>>> prompt = "A female reporter is speaking"
>>> transcript = "wish you have a good day"

>>> # set the seed for generator
>>> generator = torch.Generator("cuda").manual_seed(0)

>>> # run the generation
>>> audio = pipe(
...     prompt,
...     transcription=transcript,
...     num_inference_steps=200,
...     audio_length_in_s=10.0,
...     num_waveforms_per_prompt=2,
...     generator=generator,
...     max_new_tokens=512,          #Must set max_new_tokens equa to 512 for TTS
... ).audios

>>> # save the best audio sample (index 0) as a .wav file
>>> scipy.io.wavfile.write("tts.wav", rate=16000, data=audio[0])
```

#### enable_model_cpu_offload[[diffusers.AudioLDM2Pipeline.enable_model_cpu_offload]]

```python
enable_model_cpu_offload(gpu_id: int | None = None, device: typing.Union[torch.device, str] = 'cuda')
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/audioldm2/pipeline_audioldm2.py#L225)

Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
`enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.

#### encode_prompt[[diffusers.AudioLDM2Pipeline.encode_prompt]]

```python
encode_prompt(prompt, device, num_waveforms_per_prompt, do_classifier_free_guidance, transcription = None, negative_prompt = None, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, generated_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_generated_prompt_embeds: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.LongTensor] = None, negative_attention_mask: typing.Optional[torch.LongTensor] = None, max_new_tokens: int | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/audioldm2/pipeline_audioldm2.py#L328)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

transcription (`str` or `list[str]`) : transcription of text to speech

device (`torch.device`) : torch device

num_waveforms_per_prompt (`int`) : number of waveforms that should be generated per prompt

do_classifier_free_guidance (`bool`) : whether to use classifier free guidance or not

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the audio generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

prompt_embeds (`torch.Tensor`, *optional*) : Pre-computed text embeddings from the Flan T5 model. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be computed from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-computed negative text embeddings from the Flan T5 model. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be computed from `negative_prompt` input argument.

generated_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings from the GPT2 language model. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_generated_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings from the GPT2 language model. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be computed from `negative_prompt` input argument.

attention_mask (`torch.LongTensor`, *optional*) : Pre-computed attention mask to be applied to the `prompt_embeds`. If not provided, attention mask will be computed from `prompt` input argument.

negative_attention_mask (`torch.LongTensor`, *optional*) : Pre-computed attention mask to be applied to the `negative_prompt_embeds`. If not provided, attention mask will be computed from `negative_prompt` input argument.

max_new_tokens (`int`, *optional*, defaults to None) : The number of new tokens to generate with the GPT2 language model.

**Returns:** prompt_embeds (`torch.Tensor`)

Text embeddings from the Flan T5 model.
attention_mask (`torch.LongTensor`):
Attention mask to be applied to the `prompt_embeds`.
generated_prompt_embeds (`torch.Tensor`):
Text embeddings generated from the GPT2 language model.

Encodes the prompt into text encoder hidden states.

Example:

```python
>>> import scipy
>>> import torch
>>> from diffusers import AudioLDM2Pipeline

>>> repo_id = "cvssp/audioldm2"
>>> pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
>>> pipe = pipe.to("cuda")

>>> # Get text embedding vectors
>>> prompt_embeds, attention_mask, generated_prompt_embeds = pipe.encode_prompt(
...     prompt="Techno music with a strong, upbeat tempo and high melodic riffs",
...     device="cuda",
...     do_classifier_free_guidance=True,
... )

>>> # Pass text embeddings to pipeline for text-conditional audio generation
>>> audio = pipe(
...     prompt_embeds=prompt_embeds,
...     attention_mask=attention_mask,
...     generated_prompt_embeds=generated_prompt_embeds,
...     num_inference_steps=200,
...     audio_length_in_s=10.0,
... ).audios[0]

>>> # save generated audio sample
>>> scipy.io.wavfile.write("techno.wav", rate=16000, data=audio)
```

#### generate_language_model[[diffusers.AudioLDM2Pipeline.generate_language_model]]

```python
generate_language_model(inputs_embeds: Tensor = None, max_new_tokens: int = 8, **model_kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/audioldm2/pipeline_audioldm2.py#L275)

**Parameters:**

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) : The sequence used as a prompt for the generation.

max_new_tokens (`int`) : Number of new tokens to generate.

model_kwargs (`dict[str, Any]`, *optional*) : Ad hoc parametrization of additional model-specific kwargs that will be forwarded to the `forward` function of the model.

**Returns:** `inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`)

The sequence of generated hidden-states.

Generates a sequence of hidden-states from the language model, conditioned on the embedding inputs.

## AudioLDM2ProjectionModel[[diffusers.AudioLDM2ProjectionModel]]

#### diffusers.AudioLDM2ProjectionModel[[diffusers.AudioLDM2ProjectionModel]]

```python
diffusers.AudioLDM2ProjectionModel(text_encoder_dim, text_encoder_1_dim, langauge_model_dim, use_learned_position_embedding = None, max_seq_length = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/audioldm2/modeling_audioldm2.py#L78)

**Parameters:**

text_encoder_dim (`int`) : Dimensionality of the text embeddings from the first text encoder (CLAP).

text_encoder_1_dim (`int`) : Dimensionality of the text embeddings from the second text encoder (T5 or VITS).

langauge_model_dim (`int`) : Dimensionality of the text embeddings from the language model (GPT2).

A simple linear projection model to map two text embeddings to a shared latent space. It also inserts learned
embedding vectors at the start and end of each text embedding sequence respectively. Each variable appended with
`_1` refers to that corresponding to the second text encoder. Otherwise, it is from the first.

#### forward[[diffusers.AudioLDM2ProjectionModel.forward]]

```python
forward(hidden_states: typing.Optional[torch.Tensor] = None, hidden_states_1: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.LongTensor] = None, attention_mask_1: typing.Optional[torch.LongTensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/audioldm2/modeling_audioldm2.py#L122)

## AudioLDM2UNet2DConditionModel[[diffusers.AudioLDM2UNet2DConditionModel]]

#### diffusers.AudioLDM2UNet2DConditionModel[[diffusers.AudioLDM2UNet2DConditionModel]]

```python
diffusers.AudioLDM2UNet2DConditionModel(sample_size: int | None = None, in_channels: int = 4, out_channels: int = 4, flip_sin_to_cos: bool = True, freq_shift: int = 0, down_block_types: tuple = ('CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D'), mid_block_type: str = 'UNetMidBlock2DCrossAttn', up_block_types: tuple = ('UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D'), only_cross_attention: bool | tuple[bool] = False, block_out_channels: tuple = (320, 640, 1280, 1280), layers_per_block: int | tuple[int] = 2, downsample_padding: int = 1, mid_block_scale_factor: float = 1, act_fn: str = 'silu', norm_num_groups: int | None = 32, norm_eps: float = 1e-05, cross_attention_dim: int | tuple[int] = 1280, transformer_layers_per_block: int | tuple[int] = 1, attention_head_dim: int | tuple[int] = 8, num_attention_heads: int | tuple[int] | None = None, use_linear_projection: bool = False, class_embed_type: str | None = None, num_class_embeds: int | None = None, upcast_attention: bool = False, resnet_time_scale_shift: str = 'default', time_embedding_type: str = 'positional', time_embedding_dim: int | None = None, time_embedding_act_fn: str | None = None, timestep_post_act: str | None = None, time_cond_proj_dim: int | None = None, conv_in_kernel: int = 3, conv_out_kernel: int = 3, projection_class_embeddings_input_dim: int | None = None, class_embeddings_concat: bool = False)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/audioldm2/modeling_audioldm2.py#L163)

**Parameters:**

sample_size (`int` or `tuple[int, int]`, *optional*, defaults to `None`) : Height and width of input/output sample.

in_channels (`int`, *optional*, defaults to 4) : Number of channels in the input sample.

out_channels (`int`, *optional*, defaults to 4) : Number of channels in the output.

flip_sin_to_cos (`bool`, *optional*, defaults to `False`) : Whether to flip the sin to cos in the time embedding.

freq_shift (`int`, *optional*, defaults to 0) : The frequency shift to apply to the time embedding.

down_block_types (`tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`) : The tuple of downsample blocks to use.

mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2DCrossAttn"`) : Block type for middle of UNet, it can only be `UNetMidBlock2DCrossAttn` for AudioLDM2.

up_block_types (`tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")`) : The tuple of upsample blocks to use.

only_cross_attention (`bool` or `tuple[bool]`, *optional*, default to `False`) : Whether to include self-attention in the basic transformer blocks, see `BasicTransformerBlock`.

block_out_channels (`tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`) : The tuple of output channels for each block.

layers_per_block (`int`, *optional*, defaults to 2) : The number of layers per block.

downsample_padding (`int`, *optional*, defaults to 1) : The padding to use for the downsampling convolution.

mid_block_scale_factor (`float`, *optional*, defaults to 1.0) : The scale factor to use for the mid block.

act_fn (`str`, *optional*, defaults to `"silu"`) : The activation function to use.

norm_num_groups (`int`, *optional*, defaults to 32) : The number of groups to use for the normalization. If `None`, normalization and activation layers is skipped in post-processing.

norm_eps (`float`, *optional*, defaults to 1e-5) : The epsilon to use for the normalization.

cross_attention_dim (`int` or `tuple[int]`, *optional*, defaults to 1280) : The dimension of the cross attention features.

transformer_layers_per_block (`int` or `tuple[int]`, *optional*, defaults to 1) : The number of transformer blocks of type `BasicTransformerBlock`. Only relevant for `~models.unet_2d_blocks.CrossAttnDownBlock2D`, `~models.unet_2d_blocks.CrossAttnUpBlock2D`, `~models.unet_2d_blocks.UNetMidBlock2DCrossAttn`.

attention_head_dim (`int`, *optional*, defaults to 8) : The dimension of the attention heads.

num_attention_heads (`int`, *optional*) : The number of attention heads. If not defined, defaults to `attention_head_dim`

resnet_time_scale_shift (`str`, *optional*, defaults to `"default"`) : Time scale shift config for ResNet blocks (see `ResnetBlock2D`). Choose from `default` or `scale_shift`.

class_embed_type (`str`, *optional*, defaults to `None`) : The type of class embedding to use which is ultimately summed with the time embeddings. Choose from `None`, `"timestep"`, `"identity"`, `"projection"`, or `"simple_projection"`.

num_class_embeds (`int`, *optional*, defaults to `None`) : Input dimension of the learnable embedding matrix to be projected to `time_embed_dim`, when performing class conditioning with `class_embed_type` equal to `None`.

time_embedding_type (`str`, *optional*, defaults to `positional`) : The type of position embedding to use for timesteps. Choose from `positional` or `fourier`.

time_embedding_dim (`int`, *optional*, defaults to `None`) : An optional override for the dimension of the projected time embedding.

time_embedding_act_fn (`str`, *optional*, defaults to `None`) : Optional activation function to use only once on the time embeddings before they are passed to the rest of the UNet. Choose from `silu`, `mish`, `gelu`, and `swish`.

timestep_post_act (`str`, *optional*, defaults to `None`) : The second activation function to use in timestep embedding. Choose from `silu`, `mish` and `gelu`.

time_cond_proj_dim (`int`, *optional*, defaults to `None`) : The dimension of `cond_proj` layer in the timestep embedding.

conv_in_kernel (`int`, *optional*, default to `3`) : The kernel size of `conv_in` layer.

conv_out_kernel (`int`, *optional*, default to `3`) : The kernel size of `conv_out` layer.

projection_class_embeddings_input_dim (`int`, *optional*) : The dimension of the `class_labels` input when `class_embed_type="projection"`. Required when `class_embed_type="projection"`.

class_embeddings_concat (`bool`, *optional*, defaults to `False`) : Whether to concatenate the time embeddings with the class embeddings.

A conditional 2D UNet model that takes a noisy sample, conditional state, and a timestep and returns a sample
shaped output. Compared to the vanilla [UNet2DConditionModel](/docs/diffusers/v0.40.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel), this variant optionally includes an additional
self-attention layer in each Transformer block, as well as multiple cross-attention layers. It also allows for up
to two cross-attention embeddings, `encoder_hidden_states` and `encoder_hidden_states_1`.

This model inherits from [ModelMixin](/docs/diffusers/v0.40.0/en/api/models/overview#diffusers.ModelMixin). Check the superclass documentation for it's generic methods implemented
for all models (such as downloading or saving).

#### forward[[diffusers.AudioLDM2UNet2DConditionModel.forward]]

```python
forward(sample: Tensor, timestep: typing.Union[torch.Tensor, float, int], encoder_hidden_states: Tensor, class_labels: typing.Optional[torch.Tensor] = None, timestep_cond: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, cross_attention_kwargs: dict[str, typing.Any] | None = None, encoder_attention_mask: typing.Optional[torch.Tensor] = None, return_dict: bool = True, encoder_hidden_states_1: typing.Optional[torch.Tensor] = None, encoder_attention_mask_1: typing.Optional[torch.Tensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/audioldm2/modeling_audioldm2.py#L612)

**Parameters:**

sample (`torch.Tensor`) : The noisy input tensor with the following shape `(batch, channel, height, width)`.

timestep (`torch.Tensor` or `float` or `int`) : The number of timesteps to denoise an input.

encoder_hidden_states (`torch.Tensor`) : The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.

encoder_attention_mask (`torch.Tensor`) : A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias, which adds large negative values to the attention scores corresponding to "discard" tokens.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [UNet2DConditionOutput](/docs/diffusers/v0.40.0/en/api/models/unet2d-cond#diffusers.models.unets.unet_2d_condition.UNet2DConditionOutput) instead of a plain tuple.

cross_attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttnProcessor`.

encoder_hidden_states_1 (`torch.Tensor`, *optional*) : A second set of encoder hidden states with shape `(batch, sequence_length_2, feature_dim_2)`. Can be used to condition the model on a different set of embeddings to `encoder_hidden_states`.

encoder_attention_mask_1 (`torch.Tensor`, *optional*) : A cross-attention mask of shape `(batch, sequence_length_2)` is applied to `encoder_hidden_states_1`. If `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias, which adds large negative values to the attention scores corresponding to "discard" tokens.

**Returns:** [UNet2DConditionOutput](/docs/diffusers/v0.40.0/en/api/models/unet2d-cond#diffusers.models.unets.unet_2d_condition.UNet2DConditionOutput) or `tuple`

If `return_dict` is True, an [UNet2DConditionOutput](/docs/diffusers/v0.40.0/en/api/models/unet2d-cond#diffusers.models.unets.unet_2d_condition.UNet2DConditionOutput) is returned,
otherwise a `tuple` is returned where the first element is the sample tensor.

The [AudioLDM2UNet2DConditionModel](/docs/diffusers/v0.40.0/en/api/pipelines/audioldm2#diffusers.AudioLDM2UNet2DConditionModel) forward method.

## AudioPipelineOutput[[diffusers.AudioPipelineOutput]]

#### diffusers.AudioPipelineOutput[[diffusers.AudioPipelineOutput]]

```python
diffusers.AudioPipelineOutput(audios: ndarray)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/pipeline_utils.py#L149)

**Parameters:**

audios (`np.ndarray`) : List of denoised audio samples of a NumPy array of shape `(batch_size, num_channels, sample_rate)`.

Output class for audio pipelines.

### Visualcloze
https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/visualcloze.md

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
-->

# VisualCloze

[VisualCloze: A Universal Image Generation Framework via Visual In-Context Learning](https://huggingface.co/papers/2504.07960) is an innovative in-context learning based universal image generation framework that offers key capabilities:
1. Support for various in-domain tasks
2. Generalization to unseen tasks through in-context learning
3. Unify multiple tasks into one step and generate both target image and intermediate results
4. Support reverse-engineering conditions from target images

## Overview

The abstract from the paper is:

*Recent progress in diffusion models significantly advances various image generation tasks. However, the current mainstream approach remains focused on building task-specific models, which have limited efficiency when supporting a wide range of different needs. While universal models attempt to address this limitation, they face critical challenges, including generalizable task instruction, appropriate task distributions, and unified architectural design. To tackle these challenges, we propose VisualCloze, a universal image generation framework, which supports a wide range of in-domain tasks, generalization to unseen ones, unseen unification of multiple tasks, and reverse generation. Unlike existing methods that rely on language-based task instruction, leading to task ambiguity and weak generalization, we integrate visual in-context learning, allowing models to identify tasks from visual demonstrations. Meanwhile, the inherent sparsity of visual task distributions hampers the learning of transferable knowledge across tasks. To this end, we introduce Graph200K, a graph-structured dataset that establishes various interrelated tasks, enhancing task density and transferable knowledge. Furthermore, we uncover that our unified image generation formulation shared a consistent objective with image infilling, enabling us to leverage the strong generative priors of pre-trained infilling models without modifying the architectures. The codes, dataset, and models are available at https://visualcloze.github.io.*

## Inference

### Model loading

VisualCloze is a two-stage cascade pipeline, containing `VisualClozeGenerationPipeline` and `VisualClozeUpsamplingPipeline`.
- In `VisualClozeGenerationPipeline`, each image is downsampled before concatenating images into a grid layout, avoiding excessively high resolutions. VisualCloze releases two models suitable for diffusers, i.e., [VisualClozePipeline-384](https://huggingface.co/VisualCloze/VisualClozePipeline-384) and [VisualClozePipeline-512](https://huggingface.co/VisualCloze/VisualClozePipeline-384), which downsample images to resolutions of 384 and 512, respectively. 
- `VisualClozeUpsamplingPipeline` uses [SDEdit](https://huggingface.co/papers/2108.01073) to enable high-resolution image synthesis.

The `VisualClozePipeline` integrates both stages to support convenient end-to-end sampling, while also allowing users to utilize each pipeline independently as needed.

### Input Specifications

#### Task and Content Prompts
- Task prompt: Required to describe the generation task intention
- Content prompt: Optional description or caption of the target image
- When content prompt is not needed, pass `None`
- For batch inference, pass `List[str|None]`

#### Image Input Format
- Format: `List[List[Image|None]]`
- Structure:
  - All rows except the last represent in-context examples
  - Last row represents the current query (target image set to `None`)
- For batch inference, pass `List[List[List[Image|None]]]`

#### Resolution Control
- Default behavior:
  - Initial generation in the first stage: area of ${pipe.resolution}^2$
  - Upsampling in the second stage: 3x factor
- Custom resolution: Adjust using `upsampling_height` and `upsampling_width` parameters

### Examples

For comprehensive examples covering a wide range of tasks, please refer to the [Online Demo](https://huggingface.co/spaces/VisualCloze/VisualCloze) and [GitHub Repository](https://github.com/lzyhha/VisualCloze). Below are simple examples for three cases: mask-to-image conversion, edge detection, and subject-driven generation.

#### Example for mask2image

```python
import torch
from diffusers import VisualClozePipeline
from diffusers.utils import load_image

pipe = VisualClozePipeline.from_pretrained("VisualCloze/VisualClozePipeline-384", resolution=384, dtype=torch.bfloat16)
pipe.to("cuda")

# Load in-context images (make sure the paths are correct and accessible)
image_paths = [
    # in-context examples
    [
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_incontext-example-1_mask.jpg'),
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_incontext-example-1_image.jpg'),
    ],
    # query with the target image
    [
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_query_mask.jpg'),
        None, # No image needed for the target image
    ],
]

# Task and content prompt
task_prompt = "In each row, a logical task is demonstrated to achieve [IMAGE2] an aesthetically pleasing photograph based on [IMAGE1] sam 2-generated masks with rich color coding."
content_prompt = """Majestic photo of a golden eagle perched on a rocky outcrop in a mountainous landscape. 
The eagle is positioned in the right foreground, facing left, with its sharp beak and keen eyes prominently visible. 
Its plumage is a mix of dark brown and golden hues, with intricate feather details. 
The background features a soft-focus view of snow-capped mountains under a cloudy sky, creating a serene and grandiose atmosphere. 
The foreground includes rugged rocks and patches of green moss. Photorealistic, medium depth of field, 
soft natural lighting, cool color palette, high contrast, sharp focus on the eagle, blurred background, 
tranquil, majestic, wildlife photography."""

# Run the pipeline
image_result = pipe(
    task_prompt=task_prompt,
    content_prompt=content_prompt,
    image=image_paths,
    upsampling_width=1344,
    upsampling_height=768,
    upsampling_strength=0.4,
    guidance_scale=30,
    num_inference_steps=30,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0][0]

# Save the resulting image
image_result.save("visualcloze.png")
```

#### Example for edge-detection

```python
import torch
from diffusers import VisualClozePipeline
from diffusers.utils import load_image

pipe = VisualClozePipeline.from_pretrained("VisualCloze/VisualClozePipeline-384", resolution=384, dtype=torch.bfloat16)
pipe.to("cuda")

# Load in-context images (make sure the paths are correct and accessible)
image_paths = [
    # in-context examples
    [
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_edgedetection_incontext-example-1_image.jpg'),
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_edgedetection_incontext-example-1_edge.jpg'),
    ],
    [
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_edgedetection_incontext-example-2_image.jpg'),
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_edgedetection_incontext-example-2_edge.jpg'),
    ],
    # query with the target image
    [
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_edgedetection_query_image.jpg'),
        None, # No image needed for the target image
    ],
]

# Task and content prompt
task_prompt = "Each row illustrates a pathway from [IMAGE1] a sharp and beautifully composed photograph to [IMAGE2] edge map with natural well-connected outlines using a clear logical task."
content_prompt = ""

# Run the pipeline
image_result = pipe(
    task_prompt=task_prompt,
    content_prompt=content_prompt,
    image=image_paths,
    upsampling_width=864,
    upsampling_height=1152,
    upsampling_strength=0.4,
    guidance_scale=30,
    num_inference_steps=30,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0][0]

# Save the resulting image
image_result.save("visualcloze.png")
```

#### Example for subject-driven generation

```python
import torch
from diffusers import VisualClozePipeline
from diffusers.utils import load_image

pipe = VisualClozePipeline.from_pretrained("VisualCloze/VisualClozePipeline-384", resolution=384, dtype=torch.bfloat16)
pipe.to("cuda")

# Load in-context images (make sure the paths are correct and accessible)
image_paths = [
    # in-context examples
    [
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_subjectdriven_incontext-example-1_reference.jpg'),
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_subjectdriven_incontext-example-1_depth.jpg'),
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_subjectdriven_incontext-example-1_image.jpg'),
    ],
    [
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_subjectdriven_incontext-example-2_reference.jpg'),
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_subjectdriven_incontext-example-2_depth.jpg'),
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_subjectdriven_incontext-example-2_image.jpg'),
    ],
    # query with the target image
    [
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_subjectdriven_query_reference.jpg'),
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_subjectdriven_query_depth.jpg'),
        None, # No image needed for the target image
    ],
]

# Task and content prompt
task_prompt = """Each row describes a process that begins with [IMAGE1] an image containing the key object, 
[IMAGE2] depth map revealing gray-toned spatial layers and results in 
[IMAGE3] an image with artistic qualitya high-quality image with exceptional detail."""
content_prompt = """A vintage porcelain collector's item. Beneath a blossoming cherry tree in early spring, 
this treasure is photographed up close, with soft pink petals drifting through the air and vibrant blossoms framing the scene."""

# Run the pipeline
image_result = pipe(
    task_prompt=task_prompt,
    content_prompt=content_prompt,
    image=image_paths,
    upsampling_width=1024,
    upsampling_height=1024,
    upsampling_strength=0.2,
    guidance_scale=30,
    num_inference_steps=30,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0][0]

# Save the resulting image
image_result.save("visualcloze.png")
```

#### Utilize each pipeline independently 

```python
import torch
from diffusers import VisualClozeGenerationPipeline, FluxFillPipeline as VisualClozeUpsamplingPipeline
from diffusers.utils import load_image
from PIL import Image

pipe = VisualClozeGenerationPipeline.from_pretrained(
    "VisualCloze/VisualClozePipeline-384", resolution=384, dtype=torch.bfloat16
)
pipe.to("cuda")

image_paths = [
    # in-context examples
    [
        load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_incontext-example-1_mask.jpg"
        ),
        load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_incontext-example-1_image.jpg"
        ),
    ],
    # query with the target image
    [
        load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_query_mask.jpg"
        ),
        None,  # No image needed for the target image
    ],
]
task_prompt = "In each row, a logical task is demonstrated to achieve [IMAGE2] an aesthetically pleasing photograph based on [IMAGE1] sam 2-generated masks with rich color coding."
content_prompt = "Majestic photo of a golden eagle perched on a rocky outcrop in a mountainous landscape. The eagle is positioned in the right foreground, facing left, with its sharp beak and keen eyes prominently visible. Its plumage is a mix of dark brown and golden hues, with intricate feather details. The background features a soft-focus view of snow-capped mountains under a cloudy sky, creating a serene and grandiose atmosphere. The foreground includes rugged rocks and patches of green moss. Photorealistic, medium depth of field, soft natural lighting, cool color palette, high contrast, sharp focus on the eagle, blurred background, tranquil, majestic, wildlife photography."

# Stage 1: Generate initial image
image = pipe(
    task_prompt=task_prompt,
    content_prompt=content_prompt,
    image=image_paths,
    guidance_scale=30,
    num_inference_steps=30,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0][0]

# Stage 2 (optional): Upsample the generated image
pipe_upsample = VisualClozeUpsamplingPipeline.from_pipe(pipe)
pipe_upsample.to("cuda")

mask_image = Image.new("RGB", image.size, (255, 255, 255))

image = pipe_upsample(
    image=image,
    mask_image=mask_image,
    prompt=content_prompt,
    width=1344,
    height=768,
    strength=0.4,
    guidance_scale=30,
    num_inference_steps=30,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]

image.save("visualcloze.png")
```

## VisualClozePipeline[[diffusers.VisualClozePipeline]]

#### diffusers.VisualClozePipeline[[diffusers.VisualClozePipeline]]

```python
diffusers.VisualClozePipeline(scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKL, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, text_encoder_2: T5EncoderModel, tokenizer_2: T5Tokenizer, transformer: FluxTransformer2DModel, resolution: int = 384)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/visualcloze/pipeline_visualcloze_combined.py#L89)

**Parameters:**

transformer ([FluxTransformer2DModel](/docs/diffusers/v0.40.0/en/api/models/flux_transformer#diffusers.FluxTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.40.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`CLIPTextModel`) : [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.

text_encoder_2 (`T5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.

tokenizer (`CLIPTokenizer`) : Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).

tokenizer_2 (`T5TokenizerFast`) : Second Tokenizer of class [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).

resolution (`int`, *optional*, defaults to 384) : The resolution of each image when concatenating images from the query and in-context examples.

The VisualCloze pipeline for image generation with visual context. Reference:
https://github.com/lzyhha/VisualCloze/tree/main. This pipeline is designed to generate images based on visual
in-context examples.

#### __call__[[diffusers.VisualClozePipeline.__call__]]

```python
__call__(task_prompt: str | list[str] = None, content_prompt: str | list[str] = None, image: typing.Optional[torch.FloatTensor] = None, upsampling_height: int | None = None, upsampling_width: int | None = None, num_inference_steps: int = 50, sigmas: list[float] | None = None, guidance_scale: float = 30.0, num_images_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.FloatTensor] = None, prompt_embeds: typing.Optional[torch.FloatTensor] = None, pooled_prompt_embeds: typing.Optional[torch.FloatTensor] = None, output_type: str | None = 'pil', return_dict: bool = True, joint_attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'], max_sequence_length: int = 512, upsampling_strength: float = 1.0)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/visualcloze/pipeline_visualcloze_combined.py#L249)

**Parameters:**

task_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to define the task intention.

content_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to define the content or caption of the target image to be generated.

image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) : `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)`.

upsampling_height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The height in pixels of the generated image (i.e., output image) after upsampling via SDEdit. By default, the image is upsampled by a factor of three, and the base resolution is determined by the resolution parameter of the pipeline. When only one of `upsampling_height` or `upsampling_width` is specified, the other will be automatically set based on the aspect ratio.

upsampling_width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The width in pixels of the generated image (i.e., output image) after upsampling via SDEdit. By default, the image is upsampled by a factor of three, and the base resolution is determined by the resolution parameter of the pipeline. When only one of `upsampling_height` or `upsampling_width` is specified, the other will be automatically set based on the aspect ratio.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

guidance_scale (`float`, *optional*, defaults to 30.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.FloatTensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

pooled_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, pooled text embeddings will be generated from `prompt` input argument.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.flux.FluxPipelineOutput` instead of a plain tuple.

joint_attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int` defaults to 512) : Maximum sequence length to use with the `prompt`.

upsampling_strength (`float`, *optional*, defaults to 1.0) : Indicates extent to transform the reference `image` when upsampling the results. Must be between 0 and 1. The generated image is used as a starting point and more noise is added the higher the `upsampling_strength`. The number of denoising steps depends on the amount of noise initially added. When `upsampling_strength` is 1, added noise is maximum and the denoising process runs for the full number of iterations specified in `num_inference_steps`. A value of 0 skips the upsampling step and output the results at the resolution of `self.resolution`.

**Returns:** `~pipelines.flux.FluxPipelineOutput` or `tuple`

`~pipelines.flux.FluxPipelineOutput` if `return_dict`
is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
images.

Function invoked when calling the VisualCloze pipeline for generation.

Examples:
```python
>>> import torch
>>> from diffusers import VisualClozePipeline
>>> from diffusers.utils import load_image

>>> image_paths = [
...     # in-context examples
...     [
...         load_image(
...             "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_incontext-example-1_mask.jpg"
...         ),
...         load_image(
...             "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_incontext-example-1_image.jpg"
...         ),
...     ],
...     # query with the target image
...     [
...         load_image(
...             "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_query_mask.jpg"
...         ),
...         None,  # No image needed for the target image
...     ],
... ]
>>> task_prompt = "In each row, a logical task is demonstrated to achieve [IMAGE2] an aesthetically pleasing photograph based on [IMAGE1] sam 2-generated masks with rich color coding."
>>> content_prompt = "Majestic photo of a golden eagle perched on a rocky outcrop in a mountainous landscape. The eagle is positioned in the right foreground, facing left, with its sharp beak and keen eyes prominently visible. Its plumage is a mix of dark brown and golden hues, with intricate feather details. The background features a soft-focus view of snow-capped mountains under a cloudy sky, creating a serene and grandiose atmosphere. The foreground includes rugged rocks and patches of green moss. Photorealistic, medium depth of field, soft natural lighting, cool color palette, high contrast, sharp focus on the eagle, blurred background, tranquil, majestic, wildlife photography."
>>> pipe = VisualClozePipeline.from_pretrained(
...     "VisualCloze/VisualClozePipeline-384", resolution=384, torch_dtype=torch.bfloat16
... )
>>> pipe.to("cuda")

>>> image = pipe(
...     task_prompt=task_prompt,
...     content_prompt=content_prompt,
...     image=image_paths,
...     upsampling_width=1344,
...     upsampling_height=768,
...     upsampling_strength=0.4,
...     guidance_scale=30,
...     num_inference_steps=30,
...     max_sequence_length=512,
...     generator=torch.Generator("cpu").manual_seed(0),
... ).images[0][0]
>>> image.save("visualcloze.png")
```

## VisualClozeGenerationPipeline[[diffusers.VisualClozeGenerationPipeline]]

#### diffusers.VisualClozeGenerationPipeline[[diffusers.VisualClozeGenerationPipeline]]

```python
diffusers.VisualClozeGenerationPipeline(scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKL, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, text_encoder_2: T5EncoderModel, tokenizer_2: T5Tokenizer, transformer: FluxTransformer2DModel, resolution: int = 384)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/visualcloze/pipeline_visualcloze_generation.py#L118)

**Parameters:**

transformer ([FluxTransformer2DModel](/docs/diffusers/v0.40.0/en/api/models/flux_transformer#diffusers.FluxTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.40.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`CLIPTextModel`) : [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.

text_encoder_2 (`T5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.

tokenizer (`CLIPTokenizer`) : Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).

tokenizer_2 (`T5TokenizerFast`) : Second Tokenizer of class [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).

resolution (`int`, *optional*, defaults to 384) : The resolution of each image when concatenating images from the query and in-context examples.

The VisualCloze pipeline for image generation with visual context. Reference:
https://github.com/lzyhha/VisualCloze/tree/main This pipeline is designed to generate images based on visual
in-context examples.

#### __call__[[diffusers.VisualClozeGenerationPipeline.__call__]]

```python
__call__(task_prompt: str | list[str] = None, content_prompt: str | list[str] = None, image: typing.Optional[torch.FloatTensor] = None, num_inference_steps: int = 50, sigmas: list[float] | None = None, guidance_scale: float = 30.0, num_images_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.FloatTensor] = None, prompt_embeds: typing.Optional[torch.FloatTensor] = None, pooled_prompt_embeds: typing.Optional[torch.FloatTensor] = None, output_type: str | None = 'pil', return_dict: bool = True, joint_attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'], max_sequence_length: int = 512)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/visualcloze/pipeline_visualcloze_generation.py#L654)

**Parameters:**

task_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to define the task intention.

content_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to define the content or caption of the target image to be generated.

image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) : `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)`.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

guidance_scale (`float`, *optional*, defaults to 30.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.FloatTensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

pooled_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, pooled text embeddings will be generated from `prompt` input argument.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.flux.FluxPipelineOutput` instead of a plain tuple.

joint_attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int` defaults to 512) : Maximum sequence length to use with the `prompt`.

**Returns:** `~pipelines.flux.FluxPipelineOutput` or `tuple`

`~pipelines.flux.FluxPipelineOutput` if `return_dict`
is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
images.

Function invoked when calling the VisualCloze pipeline for generation.

Examples:
```python
>>> import torch
>>> from diffusers import VisualClozeGenerationPipeline, FluxFillPipeline as VisualClozeUpsamplingPipeline
>>> from diffusers.utils import load_image
>>> from PIL import Image

>>> image_paths = [
...     # in-context examples
...     [
...         load_image(
...             "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_incontext-example-1_mask.jpg"
...         ),
...         load_image(
...             "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_incontext-example-1_image.jpg"
...         ),
...     ],
...     # query with the target image
...     [
...         load_image(
...             "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_query_mask.jpg"
...         ),
...         None,  # No image needed for the target image
...     ],
... ]
>>> task_prompt = "In each row, a logical task is demonstrated to achieve [IMAGE2] an aesthetically pleasing photograph based on [IMAGE1] sam 2-generated masks with rich color coding."
>>> content_prompt = "Majestic photo of a golden eagle perched on a rocky outcrop in a mountainous landscape. The eagle is positioned in the right foreground, facing left, with its sharp beak and keen eyes prominently visible. Its plumage is a mix of dark brown and golden hues, with intricate feather details. The background features a soft-focus view of snow-capped mountains under a cloudy sky, creating a serene and grandiose atmosphere. The foreground includes rugged rocks and patches of green moss. Photorealistic, medium depth of field, soft natural lighting, cool color palette, high contrast, sharp focus on the eagle, blurred background, tranquil, majestic, wildlife photography."
>>> pipe = VisualClozeGenerationPipeline.from_pretrained(
...     "VisualCloze/VisualClozePipeline-384", resolution=384, torch_dtype=torch.bfloat16
... )
>>> pipe.to("cuda")

>>> image = pipe(
...     task_prompt=task_prompt,
...     content_prompt=content_prompt,
...     image=image_paths,
...     guidance_scale=30,
...     num_inference_steps=30,
...     max_sequence_length=512,
...     generator=torch.Generator("cpu").manual_seed(0),
... ).images[0][0]

>>> # optional, upsampling the generated image
>>> pipe_upsample = VisualClozeUpsamplingPipeline.from_pipe(pipe)
>>> pipe_upsample.to("cuda")

>>> mask_image = Image.new("RGB", image.size, (255, 255, 255))

>>> image = pipe_upsample(
...     image=image,
...     mask_image=mask_image,
...     prompt=content_prompt,
...     width=1344,
...     height=768,
...     strength=0.4,
...     guidance_scale=30,
...     num_inference_steps=30,
...     max_sequence_length=512,
...     generator=torch.Generator("cpu").manual_seed(0),
... ).images[0]

>>> image.save("visualcloze.png")
```

#### encode_prompt[[diffusers.VisualClozeGenerationPipeline.encode_prompt]]

```python
encode_prompt(layout_prompt: str | list[str], task_prompt: str | list[str], content_prompt: str | list[str], device: typing.Optional[torch.device] = None, num_images_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.FloatTensor] = None, pooled_prompt_embeds: typing.Optional[torch.FloatTensor] = None, max_sequence_length: int = 512, lora_scale: float | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/visualcloze/pipeline_visualcloze_generation.py#L287)

**Parameters:**

layout_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to define the number of in-context examples and the number of images involved in the task.

task_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to define the task intention.

content_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to define the content or caption of the target image to be generated.

device : (`torch.device`): torch device

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

pooled_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, pooled text embeddings will be generated from `prompt` input argument.

lora_scale (`float`, *optional*) : A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.

### JoyAI-Image-Edit-Plus
https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/joyimage_edit_plus.md
