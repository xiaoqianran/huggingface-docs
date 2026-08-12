<!-- huggingface-docs: machine-translated zh-CN from English source -->

# LM Studio 中的 GGUF 用法

![cover](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/gguf-lmstudio-coverimage.png)

[LM Studio](https://lmstudio.ai) 是一款桌面应用程序，用于直接在计算机上使用本地 AI 模型进行实验和开发。 LM Studio 基于 llama.cpp 构建，可在 Mac (Apple Silicon)、Windows 和 Linux 上运行！ 

## 将模型从 Hugging Face 导入 LM Studio

首先，在Hugging Face中的[Local Apps Settings](https://huggingface.co/settings/local-apps)下启用LM Studio。  

  <img
    class="block"
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/gguf-lmstudio-hfsettings.gif"
  />

### 选项 1：使用 Hugging Face 中的“使用此模型”按钮

对于任何 GGUF 或 MLX LLM，请单击“使用此模型”下拉列表并选择 LM Studio。如果您已经有 LM Studio，这将直接在 LM Studio 中运行该模型；如果没有，则会显示下载选项。  

  <img
    class="block"
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/gguf-lmstudio-modelcard.gif"
  />

要尝试使用热门模型的 LM Studio，请在此处找到它们：[https://huggingface.co/models?library=gguf\&sort=trending](https://huggingface.co/models?library=gguf&sort=trending)

### 选项 2：使用 LM Studio 的应用内下载器

打开 LM Studio 应用程序，然后按 Mac 上的 ⌘ + Shift + M 或 PC 上的 Ctrl + Shift + M 搜索任何模型（M 代表模型）。您甚至可以将整个拥抱脸 URL 粘贴到搜索栏中！

  <img
    class="block"
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/gguf-lmstudio-modelsearch.gif"
  />

对于每个模型，您可以展开下拉列表以查看多个量化选项。 LM Studio 会突出显示针对您的硬件的推荐选择，并指出支持哪些选项。

### 选项 3：使用 lms，LM Studio 的 CLI：如果您更喜欢基于终端的工作流程，请使用 lms（LM Studio 的 CLI）。

  <img
    class="block"
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/gguf-lmstudio-lms.gif"
  />

 

#### **从终端搜索模型：**

使用关键字搜索  
```bash
lms get qwen
```

按 MLX 或 GGUF 结果过滤搜索
```bash
lms get qwen \--mlx  # or \--gguf
```

#### **从 Hugging Face 下载任何模型：**

使用完整的拥抱脸 URL
```bash  
lms get https://huggingface.co/lmstudio-community/Ministral-3-8B-Reasoning-2512-GGUF
```

#### **选择模型量化**

您可以选择平衡性能、内存使用和准确性的模型量化级别。 

这是通过 @ 限定符完成的，例如：
```bash  
lms get https://huggingface.co/lmstudio-community/Ministral-3-8B-Reasoning-2512-GGUF@Q6\_K
```

## 您下载了模型 – 现在怎么办？

您已经按照上述选项之一下载了模型，现在让我们开始在 LM Studio 中吧！

### LM Studio 应用程序入门

在 LM Studio 应用程序中，前往模型加载器查看下载模型的列表并选择一个进行加载。您可以自定义模型加载参数，但 LM Studio 默认情况下会选择可优化硬件上模型性能的加载参数。

  <img
    class="block"
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/gguf-lmstudio-modelload.gif"
  />

模型完成加载后（如进度条所示），您可以开始使用我们应用程序的聊天界面聊天！

### 或者，使用 LM Studio 的 CLI 与您的模型进行交互查看命令列表[here](https://lmstudio.ai/docs/cli)。请注意，您需要运行 LM Studio ***至少一次***才能使用 lms

## **跟上最新型号**

关注 Hugging Face 上的 [LM Studio Community](https://huggingface.co/lmstudio-community) 页面，第一时间了解最新、最好的本地法学硕士。

### 渐变空间
https://huggingface.co/docs/hub/spaces-sdks-gradio.md