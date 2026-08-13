<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在拥抱脸部时使用 🧨 `diffusers`

Diffusers 是最先进的预训练扩散模型的首选库，用于生成图像、音频甚至分子的 3D 结构。无论您是在寻找简单的推理解决方案还是想要训练自己的扩散模型，Diffusers 都是一个支持两者的模块化工具箱。该库的设计重点是可用性而非性能、简单而非简单、可定制性而非抽象。

## 探索中心的扩散器

Hub上有超过10,000个`diffusers`兼容管道，您可以通过[the models page](https://huggingface.co/models?library=diffusers&sort=downloads)左侧的过滤找到它们。扩散系统通常由文本编码器、UNet、VAE 和调度器等多个组件组成。尽管它们不是独立的模型，但管道抽象使得可以轻松地将它们用于推理或训练。

您可以找到用于许多不同任务的扩散管道：

* 根据自然语言文本提示生成图像（[text-to-image](https://huggingface.co/models?library=diffusers&pipeline_tag=text-to-image&sort=downloads)）。
* 使用自然语言文本提示转换图像 ([image-to-image](https://huggingface.co/models?library=diffusers&pipeline_tag=image-to-image&sort=downloads))。
* 根据自然语言描述生成视频 ([text-to-video](https://huggingface.co/models?library=diffusers&pipeline_tag=text-to-video&sort=downloads))。如果您想测试模型而不下载它们，您可以直接在浏览器中试用这些模型，这要归功于浏览器内的小部件！ 

## Diffusers 存储库文件

[Diffusers](https://hf.co/docs/diffusers/index) 模型存储库包含所有必需的模型子组件，例如用于编码图像和解码潜在变量的变分自动编码器、文本编码器、变压器模型等。这些子组件被组织成多文件夹布局。

  

每个子文件夹包含类似于 [Transformers](./transformers) 模型的每个组件的权重和配置（如果适用）。

权重通常存储为安全张量文件，配置通常是包含模型架构信息的 json 文件。

## 使用现有管道

所有`diffusers`管道距离使用仅一线之遥！要运行生成，我们建议始终从 `DiffusionPipeline` 开始： 

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
```

如果您想加载特定的管道组件（例如 UNet），可以通过以下方式执行：

```py
from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet")
```

## 共享您的管道和模型

所有[pipeline classes](https://huggingface.co/docs/diffusers/main/api/pipelines/overview)、[model classes](https://huggingface.co/docs/diffusers/main/api/models/overview)和[scheduler classes](https://huggingface.co/docs/diffusers/main/api/schedulers/overview)均与集线器完全兼容。更具体地说，它们可以使用`from_pretrained()`方法轻松地从Hub加载，并可以使用`push_to_hub()`方法与其他人共享。欲了解更多详情，请查看[documentation](https://huggingface.co/docs/diffusers/main/en/using-diffusers/push_to_hub)。 

## 其他资源

* 扩散器[library](https://github.com/huggingface/diffusers)。
* 扩散器[docs](https://huggingface.co/docs/diffusers/index)。

### 在空间中使用 OpenCV
https://huggingface.co/docs/hub/spaces-using-opencv.md