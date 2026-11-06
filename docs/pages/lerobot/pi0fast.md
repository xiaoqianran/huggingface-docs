# π₀-FAST (Pi0-FAST)

π₀-FAST is a **Vision-Language-Action model for general robot control** that uses autoregressive next-token prediction to model continuous robot actions.

## Model Overview

π₀-FAST combines the power of Vision-Language Models with a novel action tokenization approach called **FAST (Frequency-space Action Sequence Tokenization)**. This enables training autoregressive VLAs on highly dexterous tasks that are impossible with standard binning-based discretization, while training **up to 5x faster** than diffusion-based approaches like π₀.

<img
  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/lerobot-pifast.png"
  alt="An overview of Pi0-FAST"
  width="85%"
/>

### Why FAST?

Standard approaches for robot action tokenization use simple per-dimension, per-timestep binning schemes. While passable for simple behaviors, this rapidly breaks down for complex and dexterous skills that require precision and high-frequency control.

FAST solves this by compressing action sequences using signal processing techniques, resulting in a dense sequence of action tokens that can be predicted autoregressively—just like language tokens.

### How FAST Tokenization Works

The FAST tokenizer compresses action sequences through the following steps:

1. **Normalize**: Take a continuous action chunk of shape `(H, D)` where `H` is the horizon and `D` is the action dimension. Normalize using one of the supported normalization methods (Quantiles recommended to handle outliers).

2. **Discrete Cosine Transform (DCT)**: Apply DCT (via scipy) to each action dimension separately. DCT is a compression algorithm commonly used in image and audio codecs (JPEG, MP3).

3. **Quantization**: Round and remove insignificant coefficients for each action dimension, producing a sparse frequency matrix.

4. **Flatten**: Flatten the matrix into a 1D vector, with low-frequency components first.

5. **Byte Pair Encoding (BPE)**: Train a BPE tokenizer to compress the DCT coefficients into dense action tokens, typically achieving **10x compression** over prior tokenization approaches.

This approach can transform **any existing VLM** into a VLA by training it to predict these FAST tokens.

## Installation Requirements

1. Install LeRobot by following our [Installation Guide](./installation).
2. Install π₀-FAST dependencies by running:

   ```bash
   pip install -e ".[pi]"
   ```

## Training a Custom FAST Tokenizer

You have two options for the FAST tokenizer:

1. **Use the pre-trained tokenizer**: The `lerobot/fast-action-tokenizer` tokenizer was trained on 1M+ real robot action sequences and works as a general-purpose tokenizer.

2. **Train your own tokenizer**: For maximum performance on your specific dataset, you can finetune the tokenizer on your own data.

### Training Your Own Tokenizer

```bash
lerobot-train-tokenizer \
    --repo_id "user/my-lerobot-dataset" \
    --action_horizon 10 \
    --encoded_dims "0:6" \
    --vocab_size 1024 \
    --scale 10.0 \
    --normalization_mode QUANTILES \
    --output_dir "./my_fast_tokenizer" \
    --push_to_hub true \
    --hub_repo_id "username/my-action-tokenizer"
```

### Key Tokenizer Parameters

| Parameter              | Description                                                                       | Default      |
| ---------------------- | --------------------------------------------------------------------------------- | ------------ |
| `--repo_id`            | LeRobot dataset repository ID                                                     | Required     |
| `--action_horizon`     | Number of future actions in each chunk                                            | `10`         |
| `--encoded_dims`       | Comma-separated dimension ranges to encode (e.g., `"0:6,7:23"`)                   | `"0:6,7:23"` |
| `--vocab_size`         | BPE vocabulary size                                                               | `1024`       |
| `--scale`              | DCT scaling factor for quantization                                               | `10.0`       |
| `--normalization_mode` | Normalization mode (`MEAN_STD`, `MIN_MAX`, `QUANTILES`, `QUANTILE10`, `IDENTITY`) | `QUANTILES`  |
| `--sample_fraction`    | Fraction of chunks to sample per episode                                          | `0.1`        |

## Usage

To use π₀-FAST in LeRobot, specify the policy type as:

```python
policy.type=pi0_fast
```

## Training

For training π₀-FAST, you can use the LeRobot training script:

```bash
lerobot-train \
    --dataset.repo_id=your_dataset \
    --policy.type=pi0_fast \
    --output_dir=./outputs/pi0fast_training \
    --job_name=pi0fast_training \
    --policy.pretrained_path=lerobot/pi0fast-base \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.chunk_size=10 \
    --policy.n_action_steps=10 \
    --policy.max_action_tokens=256 \
    --steps=100000 \
    --batch_size=4 \
    --policy.device=cuda
```

### Key Training Parameters

| Parameter                              | Description                                        | Default                         |
| -------------------------------------- | -------------------------------------------------- | ------------------------------- |
| `--policy.gradient_checkpointing=true` | Reduces memory usage significantly during training | `false`                         |
| `--policy.dtype=bfloat16`              | Use mixed precision training for efficiency        | `float32`                       |
| `--policy.chunk_size`                  | Number of action steps to predict (action horizon) | `50`                            |
| `--policy.n_action_steps`              | Number of action steps to execute                  | `50`                            |
| `--policy.max_action_tokens`           | Maximum number of FAST tokens per action chunk     | `256`                           |
| `--policy.action_tokenizer_name`       | FAST tokenizer to use                              | `lerobot/fast-action-tokenizer` |
| `--policy.compile_model=true`          | Enable torch.compile for faster training           | `false`                         |
| `--policy.optimizer_fused=true`        | Use fused AdamW; ignored on resume                 | `None`                          |

## Inference

### KV-Caching for Fast Inference

π₀-FAST supports **KV-caching**, a widely used optimization in LLM inference. This caches the key-value pairs from the attention mechanism, avoiding redundant computation during autoregressive decoding.

```python
# KV-caching is enabled by default
policy.use_kv_cache=true
```

### Inference Example

```python
from lerobot.policies.pi0_fast import PI0FastPolicy, PI0FastConfig

# Load the policy
policy = PI0FastPolicy.from_pretrained("your-model-path")

# During inference
actions = policy.predict_action_chunk(batch)
```

## Model Architecture

π₀-FAST uses a PaliGemma-based architecture:

- **Vision Encoder**: SigLIP vision tower for image understanding
- **Language Model**: Gemma 2B for processing language instructions and predicting action tokens

The model takes images, text instructions, and robot state as input, and outputs discrete FAST tokens that are decoded back to continuous actions.

## Configuration Options

| Parameter            | Description                                     | Default    |
| -------------------- | ----------------------------------------------- | ---------- |
| `paligemma_variant`  | VLM backbone variant (`gemma_300m`, `gemma_2b`) | `gemma_2b` |
| `max_state_dim`      | Maximum state vector dimension (padded)         | `32`       |
| `max_action_dim`     | Maximum action vector dimension (padded)        | `32`       |
| `temperature`        | Sampling temperature (0.0 for greedy)           | `0.0`      |
| `max_decoding_steps` | Maximum decoding steps                          | `256`      |
| `use_kv_cache`       | Enable KV caching for faster inference          | `true`     |

## Comparison with π₀

| Feature               | π₀                        | π₀-FAST                      |
| --------------------- | ------------------------- | ---------------------------- |
| Action Representation | Flow Matching (Diffusion) | Autoregressive Tokens (FAST) |
| Training Speed        | 1x                        | **5x faster**                |
| Dexterity             | High                      | High                         |
| Inference Method      | Iterative Denoising       | Autoregressive Decoding      |
| KV-Caching            | N/A                       | Supported                    |

## Reproducing π₀Fast results

We reproduce the results of π₀Fast on the LIBERO benchmark using the LeRobot implementation. We take the LeRobot PiFast base model [lerobot/pi0fast-base](https://huggingface.co/lerobot/pi0fast-base) and finetune for an additional 40k steps in bfloat16, with batch size of 256 on 8 H100 GPUs using the [HuggingFace LIBERO dataset](https://huggingface.co/datasets/HuggingFaceVLA/libero).

The finetuned model can be found here:

- **π₀Fast LIBERO**: [lerobot/pi0fast-libero](https://huggingface.co/lerobot/pi0fast-libero)

With the following training command:

```bash
lerobot-train \
  --dataset.repo_id=lerobot/libero \
  --output_dir=outputs/libero_pi0fast \
  --job_name=libero_pi0fast \
  --policy.path=lerobot/pi0fast-base \
  --policy.dtype=bfloat16 \
  --steps=100000 \
  --save_freq=20000 \
  --batch_size=4 \
  --policy.device=cuda \
  --policy.scheduler_warmup_steps=4000 \
  --policy.scheduler_decay_steps=100000 \
  --policy.scheduler_decay_lr=1e-5 \
  --policy.gradient_checkpointing=true \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --policy.max_action_tokens=256 \
  --policy.empty_cameras=1
```

We then evaluate the finetuned model using the LeRobot LIBERO implementation, by running the following command:

```bash
tasks="libero_object,libero_spatial,libero_goal,libero_10"
lerobot-eval \
  --policy.path=lerobot/pi0fast-libero \
  --policy.max_action_tokens=256 \
  --env.type=libero \
  --policy.gradient_checkpointing=false \
  --env.task=${tasks} \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
```

**Note:** We set `n_action_steps=10`, similar to the original OpenPI implementation.

### Results

We obtain the following results on the LIBERO benchmark:

| Model       | LIBERO Spatial | LIBERO Object | LIBERO Goal | LIBERO 10 | Average  |
| ----------- | -------------- | ------------- | ----------- | --------- | -------- |
| **π₀-fast** | 70.0           | 100.0         | 100.0       | 60.0      | **82.5** |

The full evaluation output folder, including videos, is available [here](https://drive.google.com/drive/folders/1HXpwPTRm4hx6g1sF2P7OOqGG0TwPU7LQ?usp=sharing)

## License

This model follows the **Apache 2.0 License**, consistent with the original [OpenPI repository](https://github.com/Physical-Intelligence/openpi).

## References

- [FAST: Efficient Robot Action Tokenization](https://www.physicalintelligence.company/research/fast) - Physical Intelligence Blog
- [OpenPI Repository](https://github.com/Physical-Intelligence/openpi) - Original implementation
- [FAST Tokenizer on Hugging Face](https://huggingface.co/physical-intelligence/fast) - Pre-trained tokenizer

### Research Paper
https://huggingface.co/docs/lerobot/v0.7.0/policy_fastwam_README.md

## Research Paper

Paper: https://arxiv.org/abs/2603.16666

## Repository

Code: https://github.com/yuantianyuan01/FastWAM

Project page: https://yuantianyuan01.github.io/FastWAM/

## Citation

```bibtex
@article{yuan2026fastwam,
  title = {Fast-WAM: Do World Action Models Need Test-time Future Imagination?},
  author = {Tianyuan Yuan and Zibin Dong and Yicheng Liu and Hang Zhao},
  journal = {arXiv preprint arXiv:2603.16666},
  year = {2026},
  url = {https://arxiv.org/abs/2603.16666}
}
```

## Additional Resources

Base video model: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B

Released upstream checkpoints: https://huggingface.co/yuanty/fastwam

## Results

Evaluated on LIBERO with [`lerobot/fastwam_libero_uncond_2cam224`](https://huggingface.co/lerobot/fastwam_libero_uncond_2cam224):

This checkpoint uses `VISUAL: IDENTITY` preprocessing. FastWAM rescales images to `[-1, 1]` inside the model, so applying image normalization in the preprocessor would normalize them twice.

| Suite          | Success rate | n_episodes |
| -------------- | -----------: | ---------: |
| libero_spatial |        97.6% |        500 |
| libero_object  |        99.0% |        500 |
| libero_goal    |        95.0% |        500 |
| libero_10      |        94.0% |        500 |
| **average**    |    **96.4%** |       2000 |

Reproduce: `lerobot-eval --policy.path=lerobot/fastwam_libero_uncond_2cam224 --policy.device=cuda --policy.dtype=float32 --policy.n_action_steps=10 --env.type=libero --env.task=libero_spatial --env.observation_height=256 --env.observation_width=256 --eval.batch_size=1 --eval.n_episodes=50 --seed=0 --env.episode_length=300`.

For LIBERO-10, use `--env.task=libero_10 --env.episode_length=600`:

```bash
lerobot-eval \
    --policy.path=lerobot/fastwam_libero_uncond_2cam224 \
    --policy.device=cuda \
    --policy.dtype=float32 \
    --policy.n_action_steps=10 \
    --env.type=libero \
    --env.task=libero_10 --env.observation_height=256 --env.observation_width=256 \
    --eval.batch_size=1 \
    --eval.n_episodes=50 \
    --seed=0 --env.episode_length=600
```

## Faster-WAM variant (arXiv 2608.02365)

Faster-WAM: Do World Action Models Need Deep Action Modules?
(https://arxiv.org/abs/2608.02365, Huawei Noah's Ark Montreal / McGill / Mila / U Toronto)
is a world-model-centric upgrade of the Fast-WAM architecture above. It is a DIFFERENT
paper from Fast-WAM (arXiv 2603.16666, this policy's default) and from the unrelated
Faster-WAM (arXiv 2608.04404, hustvl). Always cite the arXiv ID, never the name alone.

Three opt-in config flags on `FastWAMConfig` implement it (all default off, which keeps
the Fast-WAM behavior and checkpoint layout exactly):

- `action_expert_layers=1`: a single-layer action expert (paper Sec. 3.1; about 28.6M
  parameters with hidden 1024, FFN 4096, QKV to the video backbone's 3072 attention
  width). Any depth between 1 and the video backbone depth works.
- `lite_kv_fusion=True`: Dock of Transformers Lite KV-Fusion (paper Eq. 1). Each action
  layer learns, per attention head, a softmax mixture over the native K/V cached across
  ALL video layers (fused after the backbone's QK-Norm and RoPE). Adds
  `num_heads x L_v x L_a` parameters (720 for the paper's 24x30x1). Required whenever
  `action_expert_layers` differs from the video depth.
- `retracted_1d_rope=True`: retracted 1D-RoPE (paper Eq. 3). The fused video keys are
  de-rotated with the backbone's native 3D rotary transform and re-encoded on the action
  queries' 1D temporal axis (identity for the observed frame at t=0). Requires
  `lite_kv_fusion`. Optional `world_model_only_conditioning=True` additionally builds the
  action expert without text cross-attention and never passes it text or proprioceptive
  context (paper Sec. 2.2); language and proprioception condition only the video backbone.

The paper's full configuration is
`action_expert_layers=1, lite_kv_fusion=True, retracted_1d_rope=True,
world_model_only_conditioning=True`. Training uses the same joint video + action
flow-matching loss as Fast-WAM; gradients from the action loss reach the video backbone
through the fusion. At inference the video KV cache is prefilled once per chunk and the
fused context is fixed across action denoising steps, so repeated denoising only runs the
shallow expert. No trained Faster-WAM checkpoints are released here. You can initialize
from a default FastWAM backbone and then fine-tune on your own data:

```bash
lerobot-train --policy.path=lerobot/fastwam_base \
  --policy.action_expert_layers=1 --policy.lite_kv_fusion=true \
  --policy.retracted_1d_rope=true --policy.world_model_only_conditioning=true \
  --dataset.repo_id=YOUR_DATASET --policy.push_to_hub=false
```

This retains compatible backbone weights and the leading action layers. It discards
unused action layers and text modules, and initializes fusion logits to zero. This
conversion requires fine-tuning; it does not preserve the trained policy's behavior.
Keep the video backbone depth unchanged.
The Python loader permits it with `strict=False` (the default). When loading an already
converted Faster-WAM checkpoint, keep its saved Faster-WAM options and action depth.

The implementation is derived from the paper and the Apache-2.0 LeRobot fastwam code.
The public repository [TeamFasterwam/FasterWAM](https://github.com/TeamFasterwam/FasterWAM)
licenses its original contributions under the
[PolyForm Noncommercial License 1.0.0](https://github.com/TeamFasterwam/FasterWAM/blob/main/LICENSE).
Its code was not copied, translated, or adapted.

### WALL-OSS
https://huggingface.co/docs/lerobot/v0.7.0/walloss.md
