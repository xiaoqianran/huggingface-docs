# RoboMME

  The commands below run the simulator locally. To keep its dependencies away
  from your policy's, serve it instead with `lerobot-env-server
  --config_path=docker/sim/configs/robomme.yaml --sim.task= --docker=true`, where ``
  is the command's `--env.task`, and add `--env.server.endpoint=tcp/127.0.0.1:7448` to
  the command. A server runs the task it was started with (see [Simulation: local or served](./simulation)).

[RoboMME](https://robomme.github.io) is a memory-augmented manipulation benchmark built on ManiSkill (SAPIEN). It evaluates a robot's ability to retain and use information across an episode — counting, object permanence, reference, and imitation.

- **16 tasks** across 4 memory-skill suites
- **1,600 training demos** (100 per task, 50 val, 50 test)
- **Dataset**: [`lerobot/robomme`](https://huggingface.co/datasets/lerobot/robomme) — LeRobot v3.0, 768K frames at 10 fps
- **Simulator**: ManiSkill / SAPIEN, Panda arm, Linux only

![RoboMME benchmark tasks overview](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2603.04639/gradient.png)

## Tasks

| Suite                             | Tasks                                                         |
| --------------------------------- | ------------------------------------------------------------- |
| **Counting** (temporal memory)    | BinFill, PickXtimes, SwingXtimes, StopCube                    |
| **Permanence** (spatial memory)   | VideoUnmask, VideoUnmaskSwap, ButtonUnmask, ButtonUnmaskSwap  |
| **Reference** (object memory)     | PickHighlight, VideoRepick, VideoPlaceButton, VideoPlaceOrder |
| **Imitation** (procedural memory) | MoveCube, InsertPeg, PatternLock, RouteStick                  |

## Installation

> RoboMME requires **Linux** (ManiSkill/SAPIEN uses Vulkan rendering). Docker is recommended to isolate dependency conflicts.

### Native (Linux)

```bash
uv pip install --override <(printf 'gymnasium==0.29.1\nnumpy==1.26.4\n') \
  -e '.[smolvla,av-dep]' \
  'robomme @ git+https://github.com/RoboMME/robomme_benchmark.git@main'
```

> **Dependency note**: `mani-skill` (pulled by `robomme`) pins `gymnasium==0.29.1` and `numpy<2.0.0`, which conflict with lerobot's base `numpy>=2.0.0`. That's why `robomme` is not a pyproject extra — use the override install above, or the Docker approach below to avoid conflicts entirely.

### Docker (recommended)

Run RoboMME in its own simulator image, so its `numpy`/`gymnasium` pins never meet the policy's:

```bash
lerobot-env-server --config_path=docker/sim/configs/robomme.yaml --docker=true
```

Then evaluate with `--env.type=robomme --env.server.endpoint=tcp/127.0.0.1:7448` (see [Simulation: local or served](./simulation)).

## Running Evaluation

### Default (single task, single episode)

```bash
lerobot-eval \
    --policy.path=<your_policy_repo> \
    --env.type=robomme \
    --env.task=PickXtimes \
    --env.dataset_split=test \
    --env.task_ids=[0] \
    --eval.batch_size=1 \
    --eval.n_episodes=1
```

### Multi-task evaluation

Evaluate multiple tasks in one run by comma-separating task names. Use `task_ids` to control which episodes are evaluated per task. Recommended: 50 episodes per task for the test split.

```bash
lerobot-eval \
    --policy.path=<your_policy_repo> \
    --env.type=robomme \
    --env.task=PickXtimes,BinFill,StopCube,MoveCube,InsertPeg \
    --env.dataset_split=test \
    --env.task_ids=[0,1,2,3,4,5,6,7,8,9] \
    --eval.batch_size=1 \
    --eval.n_episodes=50
```

### Key CLI options for `env.type=robomme`

| Option               | Default       | Description                                        |
| -------------------- | ------------- | -------------------------------------------------- |
| `env.task`           | `PickXtimes`  | Any of the 16 task names above (comma-separated)   |
| `env.dataset_split`  | `test`        | `train`, `val`, or `test`                          |
| `env.action_space`   | `joint_angle` | `joint_angle` (8-D) or `ee_pose` (7-D)             |
| `env.episode_length` | `300`         | Max steps per episode                              |
| `env.task_ids`       | `null`        | List of episode indices to evaluate (null = `[0]`) |

## Dataset

The dataset [`lerobot/robomme`](https://huggingface.co/datasets/lerobot/robomme) is in **LeRobot v3.0 format** and can be loaded directly:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("lerobot/robomme")
```

### Dataset features

| Feature            | Shape         | Description                     |
| ------------------ | ------------- | ------------------------------- |
| `image`            | (256, 256, 3) | Front camera RGB                |
| `wrist_image`      | (256, 256, 3) | Wrist camera RGB                |
| `actions`          | (8,)          | Joint angles + gripper          |
| `state`            | (8,)          | Joint positions + gripper state |
| `simple_subgoal`   | str           | High-level language annotation  |
| `grounded_subgoal` | str           | Grounded language annotation    |
| `episode_index`    | int           | Episode ID                      |
| `frame_index`      | int           | Frame within episode            |

### Feature key alignment (training)

The env wrapper exposes its cameras under `pixels/<camera_name>`, named by `RoboMMEEnvConfig.front_camera_name` and `RoboMMEEnvConfig.wrist_camera_name` (`camera1` and `camera2` by default). The policy receives them as `observation.images.camera1` and `observation.images.camera2`, and the state (`agent_pos`) as `observation.state`.

The dataset's raw columns do not match those policy keys, so pass a `--rename_map` when fine-tuning:

```bash
--rename_map='{"image": "observation.images.camera1", "wrist_image": "observation.images.camera2", "state": "observation.state", "actions": "action"}'
```

Override `--env.front_camera_name` / `--env.wrist_camera_name` if you train against different key names, so evaluation observations match the keys the policy was trained on.

## Action Spaces

| Type          | Dim | Description                                               |
| ------------- | --- | --------------------------------------------------------- |
| `joint_angle` | 8   | 7 joint angles + 1 gripper (−1 closed, +1 open, absolute) |
| `ee_pose`     | 7   | xyz + roll/pitch/yaw + gripper                            |

Set via `--env.action_space=joint_angle` (default) or `--env.action_space=ee_pose`.

## Platform Notes

- **Linux only**: ManiSkill requires SAPIEN/Vulkan. macOS and Windows are not supported.
- **GPU recommended**: Rendering is CPU-capable but slow; CUDA + Vulkan gives full speed.
- **gymnasium / numpy conflict**: See installation note above. Docker image handles this automatically.
- **ManiSkill fork**: `robomme` depends on a specific ManiSkill fork (`YinpeiDai/ManiSkill`), pulled in automatically via the `robomme` package.

### Paper
https://huggingface.co/docs/lerobot/v0.7.0/policy_lawam_README.md

## Paper

https://arxiv.org/abs/2606.15768

## Project

Project page: https://rlinf.github.io/LaWAM/

Code: https://github.com/RLinf/LaWAM

Released checkpoints: https://huggingface.co/collections/jialei02/lawam-checkpoints

## LIBERO Benchmark

Released LIBERO SFT checkpoint, 50 episodes per task, using MuJoCo 3.3.2:

LIBERO benchmark success rates are sensitive to the MuJoCo simulator version. Newer MuJoCo releases can materially reduce measured LIBERO success rates, including for policies beyond LaWAM. This is an evaluation-environment compatibility issue rather than a LaWAM adapter issue; use `mujoco==3.3.2` to reproduce the released LaWAM LIBERO and paper results exactly.

| Suite            | Successes | Episodes | Success rate |
| ---------------- | --------: | -------: | -----------: |
| `libero_spatial` |       492 |      500 |        98.4% |
| `libero_object`  |       498 |      500 |        99.6% |
| `libero_goal`    |       490 |      500 |        98.0% |
| `libero_10`      |       490 |      500 |        98.0% |
| `overall`        |      1970 |     2000 |        98.5% |

## Citation

```bibtex
@misc{chen2026lawam,
  title = {LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies},
  author = {Chen, Jialei and Wang, Kai and Chen, Kang and Chen, Shuaihang and Gao, Feng and Tang, Wenhao and Li, Zhiyuan and Liu, Weilin and Yao, Zhuyu and Li, Boxun and Xu, Yuanbo and Yu, Chao},
  journal = {arXiv preprint arXiv:2606.15768},
  year = {2026},
  archiveprefix = {arXiv},
  primaryclass = {cs.RO},
}
```

### Paper
https://huggingface.co/docs/lerobot/v0.7.0/policy_act_README.md

## Paper

https://tonyzhaozh.github.io/aloha

## Citation

```bibtex
@article{zhao2023learning,
  title={Learning fine-grained bimanual manipulation with low-cost hardware},
  author={Zhao, Tony Z and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  journal={arXiv preprint arXiv:2304.13705},
  year={2023}
}
```

### Isaac Teleop
https://huggingface.co/docs/lerobot/v0.7.0/isaac_teleop.md
