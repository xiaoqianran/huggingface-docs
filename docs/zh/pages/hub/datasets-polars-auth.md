<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 认证

为了访问私有或门控数据集，您需要首先进行身份验证。身份验证的工作原理是提供访问令牌，该令牌将用于验证和授权您访问门控和私有数据集。第一步是为您的帐户创建访问令牌。这可以通过访问[Hugging Face Settings - Tokens](https://huggingface.co/settings/tokens)来完成。

可通过三种方式提供令牌：设置环境变量、将参数传递给阅读器或使用 Hugging Face CLI。

## 环境变量

如果您设置环境变量`HF_TOKEN`，Polars 在向 Hugging Face 请求数据集时会自动使用它。

```bash
export HF_TOKEN="REDACTED"
```

## 参数

您还可以通过 `storage_options` 参数显式向读取器提供访问令牌（例如 `read_parquet`）。有关所有参数的完整概述，请查看[API reference guide](https://docs.pola.rs/api/python/stable/reference/api/polars.read_parquet.html)。

```python
pl.read_parquet(
    "hf://datasets/roneneldan/TinyStories/data/train-*.parquet",
    storage_options={"token": ACCESS_TOKEN},
)
```

## 命令行界面

或者，您可以使用 [Hugging Face CLI](/docs/huggingface_hub/en/guides/cli) 进行身份验证。使用`hf auth login`成功登录后，访问令牌将存储在`HF_HOME`目录中，默认为`~/.cache/huggingface`。然后，Polars 将使用此令牌进行身份验证。

如果指定了多个方法，则它们按以下顺序优先：

- 参数（`storage_options`）
- 环境变量（`HF_TOKEN`）
- 命令行界面### 在拥抱脸部时使用 SpanMarker
https://huggingface.co/docs/hub/span_marker.md