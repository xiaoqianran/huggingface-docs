<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在 Hugging Face 中使用示例工厂

[⟦T5⟧](https://github.com/alex-petrenko/sample-factory) 是高吞吐量异步强化学习的代码库。它与 Hugging Face Hub 集成，可共享具有评估结果和训练指标的模型。

## 探索 Hub 中的示例工厂

您可以通过[models page](https://huggingface.co/models?library=sample-factory)左侧筛选找到`sample-factory`型号。

Hub 上的所有型号都具有有用的功能：
1. 自动生成的模型卡，其中包含描述、训练配置等。
2. 有助于发现的元数据标签。
3.与其他模型比较的评估结果。
4. 一个视频小部件，您可以在其中观看代理的表演。

## 安装库
要安装`sample-factory`库，您需要安装该软件包：

`pip install sample-factory`

众所周知，SF 可在 Linux 和 MacOS 上运行。目前没有 Windows 支持。

## 从 Hub 加载模型
### 使用 load_from_hub

要从 Hugging Face Hub 下载模型以与 Sample-Factory 一起使用，请使用 `load_from_hub` 脚本：

```
python -m sample_factory.huggingface.load_from_hub -r <HuggingFace_repo_id> -d <train_dir_path>
```

命令行参数是：- `-r`：要下载的 HF 存储库的存储库 ID。存储库 ID 的格式应为 `<username>/<repo_name>`
- `-d`：可选参数，用于指定保存实验的目录。默认为 `./train_dir` ，这会将存储库保存到 `./train_dir/<repo_name>`

### 直接下载模型库

Hugging Face 存储库可以直接使用 `git clone` 下载：

```
git clone git@hf.co:<Name of HuggingFace Repo> # example: git clone git@hf.co:bigscience/bloom
```

## 将下载的模型与 Sample-Factory 一起使用

下载模型后，您可以使用与您的环境相对应的enjoy脚本来运行存储库中的模型。例如，如果您正在下载 `mujoco-ant` 模型，则可以使用以下命令运行它：

```
python -m sf_examples.mujoco.enjoy_mujoco --algo=APPO --env=mujoco_ant --experiment=<repo_name> --train_dir=./train_dir
```

请注意，如果您的本地 train_dir 的路径与 `cfg.json` 中的路径不同，您可能需要指定 `--train_dir`

## 分享你的模型
### 使用push_to_hub

如果您想上传而不生成评估指标或重播视频，可以使用 `push_to_hub` 脚本：

```
python -m sample_factory.huggingface.push_to_hub -r <hf_username>/<hf_repo_name> -d <experiment_dir_path>
```

命令行参数是：

- `-r`：保存在 HF Hub 上的 repo_id。这与享受脚本中的`hf_repository`相同，并且必须采用`<hf_username>/<hf_repo_name>`的形式
- `-d`：要上传的实验目录的完整路径

### 使用enjoy.py您可以使用环境的 `enjoy` 脚本和 `--push_to_hub` 标志将模型上传到 Hub。使用`enjoy`上传还可以生成评估指标和重播视频。

评估指标是通过在指定环境中运行模型多次并报告这些运行的平均值和标准奖励来生成的。

其他相关的命令行参数是：

- `--hf_repository`：要推送到的存储库。格式必须为`<username>/<repo_name>`。模型将保存到`https://huggingface.co/<username>/<repo_name>`
- `--max_num_episodes`：上传前要评估的集数。用于生成评估指标。建议使用多个片段来生成准确的平均值和标准差。
- `--max_num_frames`：上传前评估的帧数。 `max_num_episodes` 的替代品
- `--no_render`：禁用渲染和显示环境步骤的标志。建议设置此标志以加快评估过程。

您还可以在评估期间保存模型的视频，以使用 `--save_video` 标志上传到中心

- `--video_frames`：视频中要渲染的帧数。默认为 -1，渲染整个剧集
- `--video_name`：要另存为的视频的名称。如果`None`，将保存到实验目录中的`replay.mp4`例如：

```
python -m sf_examples.mujoco_examples.enjoy_mujoco --algo=APPO --env=mujoco_ant --experiment=<repo_name> --train_dir=./train_dir --max_num_episodes=10 --push_to_hub --hf_username=<username> --hf_repository=<hf_repo_name> --save_video --no_render
```

### 数据集
https://huggingface.co/docs/hub/enterprise-datasets.md