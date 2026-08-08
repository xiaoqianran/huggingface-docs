# HopeJR

## Prerequisites

- [Hardware Setup](https://github.com/TheRobotStudio/HOPEJr)

## Install LeRobot

Follow the [installation instructions](https://github.com/huggingface/lerobot#installation) to install LeRobot.

Install LeRobot with HopeJR dependencies:

```bash
pip install -e ".[hopejr]"
```

## Device Configuration

Before starting calibration and operation, you need to identify the USB ports for each HopeJR component. Run this script to find the USB ports for the arm, hand, glove, and exoskeleton:

```bash
lerobot-find-port
```

This will display the available USB ports and their associated devices. Make note of the port paths (e.g., `/dev/tty.usbmodem58760433331`, `/dev/tty.usbmodem11301`) as you'll need to specify them in the `--robot.port` and `--teleop.port` parameters when recording data, replaying episodes, or running teleoperation scripts.

## Step 1: Calibration

Before performing teleoperation, HopeJR's limbs need to be calibrated. Calibration files will be saved in `~/.cache/huggingface/lerobot/calibration`

### 1.1 Calibrate Robot Hand

```bash
lerobot-calibrate \
    --robot.type=hope_jr_hand \
    --robot.port=/dev/tty.usbmodem58760432281 \
    --robot.id=blue \
    --robot.side=right
```

When running the calibration script, a calibration GUI will pop up. Finger joints are named as follows:

**Thumb**:

- **CMC**: base joint connecting thumb to hand
- **MCP**: knuckle joint
- **PIP**: first finger joint
- **DIP** : fingertip joint

**Index, Middle, Ring, and Pinky fingers**:

- **Radial flexor**: Moves base of finger towards the thumb
- **Ulnar flexor**: Moves base of finger towards the pinky
- **PIP/DIP**: Flexes the distal and proximal phalanx of the finger

Each one of these will need to be calibrated individually via the GUI.
Note that ulnar and radial flexors should have ranges of the same size (but with different offsets) in order to get symmetric movement.

  <img
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/calibration_gui_1.png"
    alt="Setting boundaries in the hand calibration GUI"
    title="Setting boundaries in the hand calibration GUI"
    width="100%"
  >

Use the calibration interface to set the range boundaries for each joint as shown above.

  <img
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/calibration_gui_2.png"
    alt="Saving calibration values"
    title="Saving calibration values"
    width="100%"
  >

Once you have set the appropriate boundaries for all joints, click "Save" to save the calibration values to the motors.

### 1.2 Calibrate Teleoperator Glove

```bash
lerobot-calibrate \
    --teleop.type=homunculus_glove \
    --teleop.port=/dev/tty.usbmodem11201 \
    --teleop.id=red \
    --teleop.side=right
```

Move each finger through its full range of motion, starting from the thumb.

```
Move thumb through its entire range of motion.
Recording positions. Press ENTER to stop...

-------------------------------------------
NAME      |    MIN |    POS |    MAX
thumb_cmc |   1790 |   1831 |   1853
thumb_mcp |   1497 |   1514 |   1528
thumb_pip |   1466 |   1496 |   1515
thumb_dip |   1463 |   1484 |   1514
```

Continue with each finger:

```
Move middle through its entire range of motion.
Recording positions. Press ENTER to stop...

-------------------------------------------
NAME                 |    MIN |    POS |    MAX
middle_mcp_abduction |   1598 |   1718 |   1820
middle_mcp_flexion   |   1512 |   1658 |   2136
middle_dip           |   1484 |   1500 |   1547
```

Once calibration is complete, the system will save the calibration to `/Users/your_username/.cache/huggingface/lerobot/calibration/teleoperators/homunculus_glove/red.json`

### 1.3 Calibrate Robot Arm

```bash
lerobot-calibrate \
    --robot.type=hope_jr_arm \
    --robot.port=/dev/tty.usbserial-1110 \
    --robot.id=white
```

This will open a calibration GUI where you can set the range limits for each motor. The arm motions are organized as follows:

- **Shoulder**: pitch, yaw, and roll
- **Elbow**: flex
- **Wrist**: pitch, yaw, and roll

  <img
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/calibration_gui_2.png"
    alt="Setting boundaries in the arm calibration GUI"
    title="Setting boundaries in the arm calibration GUI"
    width="100%"
  >

Use the calibration interface to set the range boundaries for each joint. Move each joint through its full range of motion and adjust the minimum and maximum values accordingly. Once you have set the appropriate boundaries for all joints, save the calibration.

### 1.4 Calibrate Teleoperator Exoskeleton

```bash
lerobot-calibrate \
    --teleop.type=homunculus_arm \
    --teleop.port=/dev/tty.usbmodem11201 \
    --teleop.id=black
```

The exoskeleton allows one to control the robot arm. During calibration, you'll be prompted to move all joints through their full range of motion:

```
Move all joints through their entire range of motion.
Recording positions. Press ENTER to stop...

-------------------------------------------
-------------------------------------------
NAME            |    MIN |    POS |    MAX
shoulder_pitch  |    586 |    736 |    895
shoulder_yaw    |   1257 |   1374 |   1390
shoulder_roll   |    449 |   1034 |   2564
elbow_flex      |   3023 |   3117 |   3134
wrist_roll      |   3073 |   3096 |   3147
wrist_yaw       |   2143 |   2171 |   2185
wrist_pitch     |   1975 |   1993 |   2074
Calibration saved to /Users/your_username/.cache/huggingface/lerobot/calibration/teleoperators/homunculus_arm/black.json
```

## Step 2: Teleoperation

Due to global variable conflicts in the Feetech middleware, teleoperation for arm and hand must run in separate shell sessions:

### Hand

```bash
lerobot-teleoperate \
    --robot.type=hope_jr_hand \
    --robot.port=/dev/tty.usbmodem58760432281 \
    --robot.id=blue \
    --robot.side=right \
    --teleop.type=homunculus_glove \
    --teleop.port=/dev/tty.usbmodem11201 \
    --teleop.id=red \
    --teleop.side=right \
    --display_data=true \
    --fps=30
```

### Arm

```bash
lerobot-teleoperate \
    --robot.type=hope_jr_arm \
    --robot.port=/dev/tty.usbserial-1110 \
    --robot.id=white \
    --teleop.type=homunculus_arm \
    --teleop.port=/dev/tty.usbmodem11201 \
    --teleop.id=black \
    --display_data=true \
    --fps=30
```

## Step 3: Record, Replay, Train

Record, Replay and Train with Hope-JR is still experimental.

### Record

This step records the dataset, which can be seen as an example [here](https://huggingface.co/datasets/nepyope/hand_record_test_with_video_data).

```bash
lerobot-record \
    --robot.type=hope_jr_hand \
    --robot.port=/dev/tty.usbmodem58760432281 \
    --robot.id=right \
    --robot.side=right \
    --robot.cameras='{"main": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \
    --teleop.type=homunculus_glove \
    --teleop.port=/dev/tty.usbmodem1201 \
    --teleop.id=right \
    --teleop.side=right \
    --dataset.repo_id=<USER>/hand_record_test_with_video_data \
    --dataset.single_task="Hand recording test with video data" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=5 \
    --dataset.push_to_hub=true \
    --dataset.private=true \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    # --dataset.rgb_encoder.vcodec=auto \
    --display_data=true
```

### Replay

```bash
lerobot-replay \
    --robot.type=hope_jr_hand \
    --robot.port=/dev/tty.usbmodem58760432281 \
    --robot.id=right \
    --robot.side=right \
    --dataset.repo_id=<USER>/hand_record_test_with_camera \
    --dataset.episode=0
```

### Train

```bash
lerobot-train \
  --dataset.repo_id=<USER>/hand_record_test_with_video_data \
  --policy.type=act \
  --output_dir=outputs/train/hopejr_hand \
  --job_name=hopejr \
  --policy.device=mps \
  --wandb.enable=true \
  --policy.repo_id=<USER>/hand_test_policy
```

### Evaluate

This training run can be viewed as an example [here](https://wandb.ai/tino/lerobot/runs/rp0k8zvw?nw=nwusertino).

```bash
lerobot-record \
  --robot.type=hope_jr_hand \
  --robot.port=/dev/tty.usbmodem58760432281 \
  --robot.id=right \
  --robot.side=right \
  --robot.cameras='{"main": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \
  --display_data=false \
  --dataset.repo_id=<USER>/eval_hopejr \
  --dataset.single_task="Evaluate hopejr hand policy" \
  --dataset.num_episodes=10 \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  # --dataset.rgb_encoder.vcodec=auto \
  --policy.path=outputs/train/hopejr_hand/checkpoints/last/pretrained_model
```

### Research Paper
https://huggingface.co/docs/lerobot/policy_groot_README.md

## Research Paper

GR00T N1 technical report (covers the GR00T N1.x family, including N1.7): https://arxiv.org/abs/2503.14734

GR00T N1.7 model card: https://huggingface.co/nvidia/GR00T-N1.7-3B

GR00T N1.5 research page (earlier version): https://research.nvidia.com/labs/gear/gr00t-n1_5/

> GR00T N1.5 support was removed from LeRobot; the last release supporting it is `lerobot==0.5.1`.
> Current releases support GR00T N1.7 only.

## Repository

Code: https://github.com/NVIDIA/Isaac-GR00T

## Citation

```bibtex
@inproceedings{gr00tn1_2025,
  archivePrefix = {arxiv},
  eprint     = {2503.14734},
  title      = {{GR00T} {N1}: An Open Foundation Model for Generalist Humanoid Robots},
  author     = {NVIDIA and Johan Bjorck andFernando Castañeda, Nikita Cherniadev and Xingye Da and Runyu Ding and Linxi "Jim" Fan and Yu Fang and Dieter Fox and Fengyuan Hu and Spencer Huang and Joel Jang and Zhenyu Jiang and Jan Kautz and Kaushil Kundalia and Lawrence Lao and Zhiqi Li and Zongyu Lin and Kevin Lin and Guilin Liu and Edith Llontop and Loic Magne and Ajay Mandlekar and Avnish Narayan and Soroush Nasiriany and Scott Reed and You Liang Tan and Guanzhi Wang and Zu Wang and Jing Wang and Qi Wang and Jiannan Xiang and Yuqi Xie and Yinzhen Xu and Zhenjia Xu and Seonghyeon Ye and Zhiding Yu and Ao Zhang and Hao Zhang and Yizhou Zhao and Ruijie Zheng and Yuke Zhu},
  month      = {March},
  year       = {2025},
  booktitle  = {ArXiv Preprint},
}
```

## Additional Resources

Blog: https://developer.nvidia.com/isaac/gr00t

Hugging Face Models:

- GR00T N1.7: https://huggingface.co/nvidia/GR00T-N1.7-3B
- GR00T N1.7 LIBERO checkpoints: https://huggingface.co/nvidia/GR00T-N1.7-LIBERO

Original-vs-LeRobot parity test

## Original-vs-LeRobot parity test

`tests/policies/groot/test_groot_vs_original.py` verifies this LeRobot
reimplementation of GR00T N1.7 (Qwen3-VL backbone + flow-matching action head)
against NVIDIA's original `gr00t` package with two comparisons, each parametrized
over every embodiment tag present in the checkpoint:

1. **Model parity** — given byte-identical pre-processed inputs and the same
   flow-matching seed (recorded in each artifact), both implementations must produce
   the **same raw model output** (`get_action(...)["action_pred"]`, the normalized
   flow-matching prediction). Output shapes must match exactly; any action-horizon
   or action-dim mismatch fails the test.
2. **Preprocessor parity** — given the identical raw observations (per-camera
   frames, state vectors, language instruction), LeRobot's own preprocessor pipeline
   (real Qwen3-VL chat template / tokenizer / image packing + checkpoint-driven
   state normalization, no mocks) must produce the **same collated model inputs**
   (`input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`, `state`,
   `embodiment_id`) as the original package's processor.

### Why two environments

The original `gr00t` package pins `transformers==4.57.3` (Python 3.10); this
integration requires `transformers>=5.x` (Qwen3-VL). Under 5.x, `PretrainedConfig`
is itself a defaulted dataclass, so the original config dataclasses fail to import
(`non-default argument follows default argument`). The two implementations therefore
**cannot be imported in the same Python process**.

So the test uses a **producer / consumer** split across two venvs:

1. **Producer** — `tests/policies/groot/utils/dump_original_n1_7.py`, run in the _original_
   gr00t venv. For each embodiment it builds dummy inputs generically from the
   checkpoint metadata (state dims from `statistics.json`; camera/language keys from
   the processor modality configs), runs the original model, and saves to one `.npz`
   per tag: the raw observations (`raw::` keys), the exact collated inputs
   (`in::` keys), the seed, and the raw `action_pred`.
2. **Consumer** — the pytest above, run in the _LeRobot_ venv. It discovers every
   `.npz`; the model-parity case replays the byte-identical collated inputs through
   the LeRobot model with the recorded seed and asserts the outputs match, and the
   preprocessor-parity case replays the raw observations through LeRobot's full
   preprocessor pipeline and asserts the collated tensors match.

> Artifacts generated by older versions of the dump script contain no `raw::`
> fields; the preprocessor-parity case then **skips** with a regeneration hint.
> Re-run the producer to refresh them.

### Fairness controls

- **Same pre-processed inputs (model parity)** — the original processor's `input_ids`,
  `pixel_values`, `image_grid_thw`, `attention_mask`, `state`, `embodiment_id` are
  fed verbatim to the LeRobot model (no re-tokenization / re-normalization), so the
  model comparison isolates the model. LeRobot's own tokenization / image packing is
  covered separately by the preprocessor-parity case, which compares its output
  against those same collated tensors from identical raw observations.
- **Same precision + attention kernel** — both sides run **fp32 + SDPA**. The
  original defaults to `use_flash_attention=True` (flash_attention_2 + bf16); the
  producer forces SDPA + fp32. (With the defaults the gap is ~3e-2 — pure
  kernel/rounding noise, not an implementation difference.)
- **Same flow-matching seed** — fixed right before sampling on both sides; the
  producer records it in each artifact (`--seed`, default 42) and the consumer
  replays the recorded value.

### How to run

```bash
<CopyLLMTxtMenu containerStyle="float: right; margin-left: 10px; display: inline-flex; position: relative; z-index: 10;"></CopyLLMTxtMenu>

# Resolve a local checkpoint (GR00T-N1.7-LIBERO / libero_10)
CKPT=$(python - <<'PY'
import os
from huggingface_hub import snapshot_download
print(os.path.join(snapshot_download("nvidia/GR00T-N1.7-LIBERO",
      allow_patterns=["libero_10/*"]), "libero_10"))
PY
)

# 1) Produce the original-side artifacts for all embodiments (original gr00t venv, CUDA)
CUDA_VISIBLE_DEVICES=0 /path/to/Isaac-GR00T/.venv-original/bin/python \
    tests/policies/groot/utils/dump_original_n1_7.py \
    --ckpt "$CKPT" --out-dir tests/policies/groot/artifacts --device cuda --seed 42

# 2) Run the parity test (LeRobot venv) — one parametrized case per embodiment
CUDA_VISIBLE_DEVICES=0 GROOT_PARITY_DEVICE=cuda \
    uv run pytest tests/policies/groot/test_groot_vs_original.py -v -s
```

The `.npz` artifacts are local-only (gitignored, ~6–10 MB each) and are regenerated by
the producer; they are never committed. The tests **skip** (do not fail) on CI or
when the checkpoint / artifacts are absent.

#### Env knobs (all optional)

| Var                                       | Default                          | Purpose                               |
| ----------------------------------------- | -------------------------------- | ------------------------------------- |
| `GROOT_N1_7_PARITY_DIR`                   | `tests/policies/groot/artifacts` | directory of per-tag `.npz` artifacts |
| `GROOT_N1_7_LIBERO_CKPT`                  | auto (HF cache)                  | override checkpoint dir               |
| `GROOT_PARITY_DEVICE`                     | `cuda` if available              | `cpu` or `cuda`                       |
| `GROOT_PARITY_ATOL` / `GROOT_PARITY_RTOL` | `1e-3`                           | comparison tolerance                  |

### π₀ (pi0)
https://huggingface.co/docs/lerobot/policy_pi0_README.md
