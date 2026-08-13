<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 硬件

告诉集线器您拥有哪些计算硬件（GPU、CPU 或 Apple Silicon），它将帮助您找到在您的设置上运行的模型。它还会显示在您的公开个人资料中，因此您可以将您的设置与社区进行比较，并且可以随时将其设为私有。

随时通过您的 [Hardware settings](https://huggingface.co/settings/hardware) 进行管理。

## 添加您的硬件

在您的 [Hardware settings](https://huggingface.co/settings/hardware) 页面上，添加您拥有的每件硬件：

1. 选择**类型**：GPU、CPU 或 Apple Silicon。
2. 选择**提供商**和**型号**（例如 NVIDIA RTX 4090）。
3. 设置您拥有的**内存**（VRAM、RAM 或统一内存）和**单元数量**。

添加任意数量的项目。如果您拥有多个，请将其中一个标记为您的**主要**硬件，以便它首先出现在型号页面上。

> [!提示]
> 默认情况下，您的硬件是公开的。关闭 **公开可见** 开关以保持其私密性。

## 查看哪些型号适合您的硬件

在提供 [GGUF](./gguf) 或 MLX 文件的模型页面上，**硬件兼容性**面板会估计每个量化是否将在您保存的硬件上运行，因此您可以在下载之前选择适合的大小。将其与 [Local Apps](./local-apps) 配对，只需点击几下即可启动并运行。

## 分享和比较

当您的硬件是公开的时：- 您的个人资料上会出现 **TFLOPS** 徽章，总结您的估计总计算能力。
- 您可以浏览 [what the community is running](https://huggingface.co/hardware) 并查看您的设置在 GPU、CPU 和 Apple Silicon 之间的比较情况。

## 从API读取硬件

如果你的硬件是公共的，它也是[available programmatically](https://huggingface-openapi.hf.space/#tag/users/GET/api/users/{username}/overview)：

- 当您的应用程序被授予 `profile` 范围时，`GET https://huggingface.co/oauth/userinfo` 包含 `hardwareItems` 声明，因此 [Sign in with Hugging Face](./oauth) 应用程序无需额外调用即可读取登录用户的设置。
- `GET https://huggingface.co/api/users/{username}/overview` 为任何硬件公开的用户返回一个 `hardwareItems` 数组。

```bash
curl https://huggingface.co/api/users/{username}/overview
```

```json
{
  "hardwareItems": [
    { "sku": ["GPU", "NVIDIA", "RTX 4090"], "mem": 24, "num": 2, "isPrimary": true }
  ]
}
```

每个项目包含：

- `sku`：`[type, provider, model]` 三元组。
- `mem`：以 GB 为单位的内存（VRAM、RAM 或统一内存）。
- `num`：单位数量。
- `isPrimary`：仅出现在标记为主要的项目上。

如果用户关闭 **公开可见** 开关，则两个响应中都会省略 `hardwareItems`。有关完整响应模式，请参阅[Hub API documentation](./api)。

## 后续步骤

- [Use AI Models Locally](./local-apps) — 使用您最喜欢的本地应用程序运行模型。
- [Local Agents with llama.cpp](./agents-local) — 在您自己的硬件上构建编码代理。

### 网络钩子
https://huggingface.co/docs/hub/webhooks.md