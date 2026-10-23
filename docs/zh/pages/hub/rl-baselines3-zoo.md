<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在拥抱脸部时使用 RL-Baselines3-Zoo

`rl-baselines3-zoo`是一个使用稳定基线3的强化学习训练框架。

## 在 Hub 中探索 RL-Baselines3-Zoo

您可以通过[models page](https://huggingface.co/models?library=stable-baselines3)左侧的过滤找到RL-Baselines3-Zoo模型。

Stable-Baselines3 团队托管着超过 150 个训练有素的强化学习代理，这些代理具有经过调整的超参数，您可以找到[here](https://huggingface.co/sb3)。

Hub 上的所有型号都具有有用的功能：
1. 自动生成的模型卡，其中包含描述、训练配置等。
2. 有助于发现的元数据标签。
3.与其他模型比较的评估结果。
4. 一个视频小部件，您可以在其中观看代理的表演。

## 使用现有模型
您可以使用 `load_from_hub` 从 Hub 下载模型：

```
# Download ppo SpaceInvadersNoFrameskip-v4 model and save it into the logs/ folder
python -m rl_zoo3.load_from_hub --algo dqn --env SpaceInvadersNoFrameskip-v4 -f logs/ -orga sb3
python enjoy.py --algo dqn --env SpaceInvadersNoFrameskip-v4  -f logs/
```

您可以定义三个参数：
- `--repo-name`：存储库的名称。
- `-orga`：Hugging Face 用户名或组织。
- `-f`：目标文件夹。

## 分享你的模型
您可以使用`push_to_hub`轻松上传您的模型。这将保存模型、对其进行评估、生成模型卡并录制代理的重播视频，然后将完整的存储库推送到集线器。

```
python -m rl_zoo3.push_to_hub  --algo dqn  --env SpaceInvadersNoFrameskip-v4  --repo-name dqn-SpaceInvadersNoFrameskip-v4  -orga ThomasSimonini  -f logs/
```您可以定义三个参数：
- `--repo-name`：存储库的名称。
- `-orga`：您的 Hugging Face 用户名。
- `-f`：保存模型的文件夹。

## 其他资源

* RL-Baselines3-Zoo [official trained models](https://huggingface.co/sb3)
* RL-Baselines3-Zoo [documentation](https://github.com/DLR-RM/rl-baselines3-zoo)

### 图书馆
https://huggingface.co/docs/hub/models-libraries.md