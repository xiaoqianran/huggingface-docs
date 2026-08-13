<!-- huggingface-docs: machine-translated zh-CN from English source -->

# Spaces ZeroGPU：空间的动态 GPU 分配

ZeroGPU 是一个共享基础设施，可优化 AI 模型和 Hugging Face Spaces 上演示的 GPU 使用。它根据需要动态分配和释放 NVIDIA RTX Pro 6000 Blackwell GPU，提供：

1. **免费 GPU 访问**：为空间提供经济高效的 GPU 使用。
2. **多 GPU 支持**：允许 Spaces 在单个应用程序上同时利用多个 GPU。

与传统的单 GPU 分配不同，ZeroGPU 的高效系统通过最大限度地提高资源利用率和能效，降低了开发人员、研究人员和组织部署 AI 模型的障碍。

## 使用和托管 ZeroGPU 空间- **使用现有的 ZeroGPU 空间**
  - ZeroGPU 空间可供所有用户免费使用。 （访问[the curated list](https://huggingface.co/spaces/enzostvs/zero-gpu-spaces)）。
  - [PRO users](https://huggingface.co/subscribe/pro) 获得额外 8 倍的每日使用配额、GPU 队列中的最高优先级，并且在使用任何 ZeroGPU 空间时可以使用预付费积分超出每日配额。
- **托管您自己的 ZeroGPU 空间**
  - 免费个人帐户：信誉良好的帐户（经过验证的电子邮件、超过 30 天的帐户）可以免费托管最多 2 个 ZeroGPU 空间。
  - PRO 帐户：[Subscribe to PRO](https://huggingface.co/settings/billing/subscription) 在您的帐户下托管最多 10 个 ZeroGPU 空间。
  - 组织：[Subscribe to a Team or Enterprise plan](https://huggingface.co/enterprise) 为所有组织成员启用 ZeroGPU 空间。

## 技术规格

ZeroGPU 支持两种 GPU 尺寸

| GPU 大小 |支持硬件|内存 |配额成本|
|--------------------------------|------------------------------------|------|------------|
| `large` *（默认）* |一半 NVIDIA RTX Pro 6000 Blackwell | 48GB | 1× |
| `xlarge` |完整版 NVIDIA RTX Pro 6000 Blackwell | 96GB | 2×|

> [!注意]
> 请参阅 [GPU size selection](#gpu-size-selection) 了解如何使用尺寸

## 兼容性ZeroGPU 空间设计为与大多数基于 PyTorch 的 GPU 空间兼容。虽然`transformers`和`diffusers`等高级Hugging Face库的兼容性得到了增强，但用户应该注意：

- 目前，ZeroGPU Spaces 与 **Gradio SDK** 完全兼容。
- 与标准 GPU 空间相比，ZeroGPU 空间的兼容性可能有限。
- 在某些情况下可能会出现意外问题。

### 支持的版本

- **收音机**：4+
- **PyTorch**：支持从 **2.8.0** 到 **最新** 的几乎所有版本
  
    查看完整列表

    - 2.8.0  
    - 2.9.1  
    - 2.10.0  
    - 2.11.0  

  
- **Python**：
  - 2012年12月3日  
  - 2013年10月3日  

## ZeroGPU 入门

要在您的空间中使用 ZeroGPU，请按照以下步骤操作：

1. 确保在您的空间设置中选择 ZeroGPU 硬件。
2. 导入`spaces`模块。
3. 用`@spaces.GPU`装饰GPU相关函数。

这个装饰过程允许Space在调用函数时请求GPU并在完成后释放它。

> [!注意]
> `@spaces.GPU` 装饰器设计为在非 ZeroGPU 环境中无影响，确保不同设置之间的兼容性。

### 用法示例

```python
import spaces
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(...)
pipe.to('cuda')

@spaces.GPU
def generate(prompt):
    return pipe(prompt).images

gr.Interface(
    fn=generate,
    inputs=gr.Text(),
    outputs=gr.Gallery(),
).launch()
```

### 模型加载尽管真正的 GPU 仅在 `@spaces.GPU` 函数中可用，但模型必须放置在根模块级别的 `cuda` 上（如上例所示）。

不鼓励在`@spaces.GPU`内延迟加载或移动模型到 CUDA，因为它的效率明显较低（CUDA 传输针对启动期间完成的放置进行了优化）。

> [!注意]
> 在模块级别加载`cuda`上的模型是有效的，因为在`@spaces.GPU`功能之外启用了PyTorch CUDA模拟模式，允许在没有真正GPU的情况下进行CUDA操作。在`@spaces.GPU`内部，使用的是真正的CUDA。

## GPU 大小选择

`@spaces.GPU`使用的默认大小是`large`（NVIDIA RTX Pro 6000 Blackwell的一半）。

您可以通过指定 `size="xlarge"` 明确请求完整的 NVIDIA RTX Pro 6000 Blackwell：

``` python
@spaces.GPU(size="xlarge")
def generate(prompt):
    return pipe(prompt).images
```

> [!注意]
> - `xlarge`比`large`多消耗**2×**每日配额（例如45秒**有效**任务持续时间消耗90秒配额）
> - `xlarge`通常意味着更高的排队概率和更长的等待时间
> - 仅当您的工作负载真正受益于额外的计算或内存时才使用`xlarge`

## 持续时间管理

对于预计超过默认 60 秒 GPU 运行时间的函数，您可以指定自定义持续时间：

```python
@spaces.GPU(duration=120)
def generate(prompt):
   return pipe(prompt).images
```这会将最大函数运行时间设置为 120 秒。为更快的功能指定更短的持续时间将提高空间访客的队列优先级。

### 动态持续时间

`@spaces.GPU`还支持动态持续时间。

不要直接传递持续时间，只需传递一个可调用函数，该可调用函数采用与装饰函数相同的输入并返回持续时间值：

```python
def get_duration(prompt, steps):
    step_duration = 3.75
    return steps * step_duration

@spaces.GPU(duration=get_duration)
def generate(prompt, steps):
   return pipe(prompt, num_inference_steps=steps).images
```

## 编译

ZeroGPU不支持`torch.compile`，但您可以使用PyTorch**提前**编译（需要torch`2.8+`）

查看此[blogpost](https://huggingface.co/blog/zerogpu-aoti)，获取有关 ZeroGPU 上提前编译的完整指南。

## 使用等级

GPU 使用量受每个帐户层的**每日**配额限制：|账户类型 |包含每日 GPU 配额 |队列优先级|
| ------------------------------------------ | ------------------------ | ---------------- |
|未经验证 | 2 分钟 |低|
|免费帐户 | 5 分钟 |中等|
|专业帐户 | 40 分钟（可延长）|最高|
|团队组织成员| 40 分钟（可延长）|最高|
|企业组织会员| 60 分钟（可延长）|最高|

包含的每日配额将在您首次使用 GPU 后 24 小时内重置。

> [!注意]
> 剩余配额直接影响 ZeroGPU 队列中的优先级。

### 通过积分扩大配额

PRO、团队和企业用户可以通过以每 10 分钟 GPU 时间 **1 美元**的速度消耗预付费积分，继续使用超出每日配额的 ZeroGPU Spaces。
一旦您的每日配额用完，任何额外的 GPU 使用量都会自动从您的信用余额中扣除。

您可以从您的[billing settings](https://huggingface.co/settings/billing)添加积分。

## 托管限制- **免费个人帐户**：最多 2 个 ZeroGPU 空间，适用于信誉良好的帐户（经过验证的电子邮件、超过 30 天的帐户）。
- **个人帐户 ([PRO subscribers](https://huggingface.co/subscribe/pro))**：最多 10 个 ZeroGPU 空间。
- **组织帐户 ([Team & Enterprise](https://huggingface.co/enterprise))**：最多 50 个 ZeroGPU 空间。

通过利用 ZeroGPU，开发人员可以创建更高效​​、可扩展的空间，最大限度地提高 GPU 利用率，同时最大限度地降低成本。

## 建议

如果您的演示使用大型模型，我们建议使用提前编译和 flash-attention 3 等优化。您可以了解如何利用这些优化
[this post](https://huggingface.co/blog/zerogpu-aoti) 中的 ZeroGPU。这些优化将帮助您最大限度地发挥 ZeroGPU 时间的优势并提供
更好的用户体验。

## 反馈

您可以直接在 HF Hub 上分享您对 Spaces ZeroGPU 的反馈：https://huggingface.co/spaces/zero-gpu-explorers/README/discussions

### 小部件
https://huggingface.co/docs/hub/models-widgets.md