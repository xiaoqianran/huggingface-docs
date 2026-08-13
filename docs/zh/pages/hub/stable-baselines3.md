<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在拥抱面上使用稳定基线3

`stable-baselines3` 是 PyTorch 中强化学习算法的一组可靠实现。

## 在 Hub 中探索稳定基线3

您可以通过[models page](https://huggingface.co/models?library=stable-baselines3)左侧的过滤找到Stable-Baselines3模型。

Hub 上的所有型号都具有有用的功能：
1. 自动生成的模型卡，其中包含描述、训练配置等。
2. 有助于发现的元数据标签。
3.与其他模型比较的评估结果。
4. 一个视频小部件，您可以在其中观看代理的表演。

## 安装库

要安装`stable-baselines3`库，您需要安装两个包：
- `stable-baselines3`：Stable-Baselines3 库。
- `huggingface-sb3`：从 Hub 加载和上传 Stable-baselines3 模型的附加代码。

```
pip install stable-baselines3
pip install huggingface-sb3
```

## 使用现有模型
您只需使用 `load_from_hub` 功能从 Hub 下载模型即可

```
checkpoint = load_from_hub(
    repo_id="sb3/demo-hf-CartPole-v1",
    filename="ppo-CartPole-v1.zip",
)
```

您需要定义两个参数：
- `--repo-id`：您要下载的 Hugging Face 存储库的名称。
- `--filename`：您要下载的文件。

## 分享你的模型
您可以使用两种不同的功能轻松上传模型：1. `package_to_hub()`：保存模型，对其进行评估，生成模型卡并录制代理的重播视频，然后将完整的存储库推送到 Hub。

```
package_to_hub(model=model, 
               model_name="ppo-LunarLander-v2",
               model_architecture="PPO",
               env_id=env_id,
               eval_env=eval_env,
               repo_id="ThomasSimonini/ppo-LunarLander-v2",
               commit_message="Test commit")
```

您需要定义七个参数：
- `--model`：您经过训练的模型。
- `--model_architecture`：模型架构的名称（DQN、PPO、A2C、SAC...）。
- `--env_id`：环境名称。
- `--eval_env`：用于评估代理的环境。
- `--repo-id`：您要创建或更新的 Hugging Face 存储库的名称。是`<your huggingface username>/<the repo name>`。
- `--commit-message`。
- `--filename`：您要推送到 Hub 的文件。

2. `push_to_hub()`：只需将文件推送到 Hub

```
push_to_hub(
    repo_id="ThomasSimonini/ppo-LunarLander-v2",
    filename="ppo-LunarLander-v2.zip",
    commit_message="Added LunarLander-v2 model trained with PPO",
)
```
您需要定义三个参数：
- `--repo-id`：您要创建或更新的 Hugging Face 存储库的名称。这是`<your huggingface username>/<the repo name>`。
- `--filename`：您要推送到 Hub 的文件。
- `--commit-message`。

## 其他资源

* 拥抱脸部稳定-基线3 [documentation](https://github.com/huggingface/huggingface_sb3#hugging-face--x-stable-baselines3-v20)
* 稳定基线3 [documentation](https://stable-baselines3.readthedocs.io/en/master/)

### 示例和教程
https://huggingface.co/docs/hub/jobs-examples.md