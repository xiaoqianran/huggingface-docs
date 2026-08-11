<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 亚马逊 SageMaker

Hugging Face 和 Amazon 推出新款 [Hugging Face Deep Learning Containers (DLCs)](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#huggingface-training-containers)
使在 [Amazon SageMaker](https://aws.amazon.com/sagemaker/) 中训练 Hugging Face Transformer 模型变得比以往更容易。

## 开始使用

### 设置和安装

您需要注册一个 AWS 账户，然后才能在 Amazon SageMaker 上运行 Accelerate 脚本。如果你不这样做
已有 AWS 账户，但了解更多信息 [here](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-set-up.html)。

拥有 AWS 账户后，您需要安装用于 Accelerate 的 `sagemaker` sdk：

```bash
pip install "accelerate[sagemaker]" --upgrade
```

Accelerate 目前使用 DLC，预装了`transformers`、`datasets` 和 `tokenizers`。 Accelerate 尚未包含在 DLC 中（很快就会添加！），因此要在 Amazon SageMaker 中使用它，您需要创建一个
`requirements.txt` 位于训练脚本所在的同一目录中，并将其添加为依赖项：

```
accelerate
```

您还应该添加对此 `requirements.txt` 的任何其他依赖项。

### 配置加速

您可以像配置非 SageMaker 训练作业一样配置 Amazon SageMaker 的启动配置：
加速 CLI：

```bash
accelerate config
# In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): 1
```

Accelerate 将完成有关您的 Amazon SageMaker 设置的调查问卷，并创建一个您可以编辑的配置文件。

    加速不会保存您的任何凭据。

### 准备一个 Accelerate 微调脚本训练脚本与您可能在 SageMaker 外部运行的训练脚本非常相似，但要保存模型
训练后，您需要指定 `/opt/ml/model` 或使用 `os.environ["SM_MODEL_DIR"]` 作为保存
目录。训练完成后，该目录中的工件将上传到 S3：

```diff
- torch.save('/opt/ml/model`)
+ accelerator.save('/opt/ml/model')
```

    SageMaker 不支持 argparse 操作。例如，如果您想使用布尔超参数，则需要
    在脚本中将类型指定为 bool 并为此超参数提供明确的 True 或 False 值。 [[参考]](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#prepare-a-pytorch-training-script)。

### 启动培训

您可以通过以下方式启动 Accelerate CLI 培训：

```
accelerate launch path_to_script.py --args_to_the_script
```

这将使用您的配置启动您的训练脚本。您唯一需要做的就是提供所有
训练脚本所需的参数作为命名参数。

**示例**

    如果您运行示例脚本之一，请不要忘记向其中添加 `accelerator.save('/opt/ml/model')`。

```bash
accelerate launch ./examples/sagemaker_example.py
```

输出：

```
Configuring Amazon SageMaker environment
Converting Arguments to Hyperparameters
Creating Estimator
2021-04-08 11:56:50 Starting - Starting the training job...
2021-04-08 11:57:13 Starting - Launching requested ML instancesProfilerReport-1617883008: InProgress
.........
2021-04-08 11:58:54 Starting - Preparing the instances for training.........
2021-04-08 12:00:24 Downloading - Downloading input data
2021-04-08 12:00:24 Training - Downloading the training image..................
2021-04-08 12:03:39 Training - Training image download completed. Training in progress..
........
epoch 0: {'accuracy': 0.7598039215686274, 'f1': 0.8178438661710037}
epoch 1: {'accuracy': 0.8357843137254902, 'f1': 0.882249560632689}
epoch 2: {'accuracy': 0.8406862745098039, 'f1': 0.8869565217391304}
........
2021-04-08 12:05:40 Uploading - Uploading generated training model
2021-04-08 12:05:40 Completed - Training job completed
Training seconds: 331
Billable seconds: 331
You can find your model data at: s3://your-bucket/accelerate-sagemaker-1-2021-04-08-11-56-47-108/output/model.tar.gz
```

## 高级功能

### 分布式训练：数据并行通过运行 `accelerate config` 设置加速配置并回答 SageMaker 问题并进行设置。
要使用 SageMaker DDP，请在询问时选择它 
`What is the distributed mode? ([0] No distributed training, [1] data parallelism):`。
下面的示例配置：
```yaml
base_job_name: accelerate-sagemaker-1
compute_environment: AMAZON_SAGEMAKER
distributed_type: DATA_PARALLEL
ec2_instance_type: ml.p3.16xlarge
iam_role_name: xxxxx
image_uri: null
mixed_precision: fp16
num_machines: 1
profile: xxxxx
py_version: py10
pytorch_version: 2.5.0
region: us-east-1
transformers_version: 4.17.0
use_cpu: false
```

### 分布式训练：模型并行性

*目前正在开发中，很快就会得到支持。*

### Python 包和依赖项

Accelerate 目前使用 DLC，预装了 `transformers`、`datasets` 和 `tokenizers`。如果你
想要使用不同/其他 Python 包，您可以通过将它们添加到 `requirements.txt` 来实现。这些包
将在您的训练脚本开始之前安装。

### 本地训练：SageMaker 本地模式

SageMaker SDK 中的本地模式允许您在 HuggingFace DLC（深度学习容器）内本地运行训练脚本 
或使用您的自定义容器映像。这对于在最终容器环境中调试和测试训练脚本非常有用。
本地模式使用 Docker compose（*注意：尚不支持 Docker Compose V2*）。 SDK将处理针对ECR的身份验证
将 DLC 拉至您的本地环境。您可以模拟 CPU（单实例和多实例）和 GPU（单实例）SageMaker 训练作业。要使用本地模式，您需要将`ec2_instance_type`设置为`local`。

```yaml
ec2_instance_type: local
```

### 高级配置

该配置允许您覆盖 [Estimator](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html) 的参数。
这些设置必须在配置文件中应用，并且不是`accelerate config`的一部分。您可以控制训练作业的许多其他方面，例如使用 Spot 实例、启用网络隔离等等。

```yaml
additional_args:
  # enable network isolation to restrict internet access for containers
  enable_network_isolation: True
```

您可以找到所有可用的配置[here](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html)。

### 使用 Spot 实例

您可以使用 Spot 实例，例如使用（参见[Advanced configuration](#advanced-configuration)）：
```yaml
additional_args:
  use_spot_instances: True
  max_wait: 86400
```

*注意：Spot 实例可能会被终止，并且训练将从检查点继续。开箱即用的加速中未处理此问题。如果您需要此功能，请联系我们。*

### 远程脚本：使用位于 Github 上的脚本

*尚未决定是否需要该功能。如果您需要此功能，请联系我们。*

### 威震天-LM
https://huggingface.co/docs/accelerate/v1.14.0/usage_guides/megatron_lm.md