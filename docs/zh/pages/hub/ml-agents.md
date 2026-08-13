<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在拥抱脸部时使用 ML-Agents

`ml-agents` 是一个开源工具包，可以使用 Unity 制作的游戏和模拟作为训练智能代理的环境。

## 探索 Hub 中的 ML-Agent

您可以通过[models page](https://huggingface.co/models?library=ml-agents)左侧筛选找到`ml-agents`型号。

Hub 上的所有型号都具有有用的功能：
1. 自动生成的模型卡，其中包含描述、训练配置等。
2. 有助于发现的元数据标签。
3. Tensorboard 摘要文件，用于可视化训练指标。
4. Spaces Web 演示的链接，您可以在其中可视化您的代理在浏览器中的运行情况。

## 安装库

要安装 `ml-agents` 库，您需要克隆存储库：

```
# Clone the repository
git clone https://github.com/Unity-Technologies/ml-agents

# Go inside the repository and install the package
cd ml-agents
pip3 install -e ./ml-agents-envs
pip3 install -e ./ml-agents
```

## 使用现有模型

您只需使用 `mlagents-load-from-hf` 从 Hub 下载模型即可。

```
mlagents-load-from-hf --repo-id="ThomasSimonini/MLAgents-Pyramids" --local-dir="./downloads"
```

您需要定义两个参数：
- `--repo-id`：您要下载的 Hugging Face 存储库的名称。
- `--local-dir`：模型下载路径。

## 想象一个智能体正在玩

您可以轻松观看直接在浏览器中播放的任何模型：1. 转到您的模型存储库。
2. 在`Watch Your Agent Play`部分中，单击链接。
3. 在演示中的第 1 步中，选择您的模型存储库，即模型 ID。
4. 在步骤 2 中，选择您要重播的模型。

## 分享你的模型

您可以使用`mlagents-push-to-hf`轻松上传模型：

```
mlagents-push-to-hf --run-id="First Training" --local-dir="results/First Training" --repo-id="ThomasSimonini/MLAgents-Pyramids" --commit-message="Pyramids"
```

您需要定义四个参数：
- `--run-id`：训练运行id的名称。
- `--local-dir`：保存模型的位置。
- `--repo-id`：您要创建或更新的 Hugging Face 存储库的名称。是`<your huggingface username>/<the repo name>`。
- `--commit-message`。

## 其他资源

* ML-代理[documentation](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Hugging-Face-Integration.md)
* 官方 Unity ML-Agents 空间 [demos](https://huggingface.co/unity)

### 下载数据集
https://huggingface.co/docs/hub/datasets-downloading.md