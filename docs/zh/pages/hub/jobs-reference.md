<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 参考

# 作业命令行界面 (CLI)

`huggingface_hub` Python 包附带一个名为 `hf` 的内置 CLI。该工具允许您直接从终端与 Hugging Face Hub 进行交互。例如，您可以登录帐户、创建存储库、上传和下载文件等。它还具有配置计算机或管理缓存以及启动和管理作业的便捷功能。

在此处的 `huggingface_hub` 文档中查找 `hf jobs` 安装步骤、指南和参考：

* [Installation](https://huggingface.co/docs/huggingface_hub/en/guides/cli#getting-started)
* [Run and manage Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/cli#hf-jobs)
* [CLI reference for Jobs](https://huggingface.co/docs/huggingface_hub/en/package_reference/cli#hf-jobs)

## Python 客户端

`huggingface_hub` Python 包附带了一个名为 `HfApi` 的客户端。该客户端允许您直接使用 Python 与 Hugging Face Hub 进行交互。例如，您可以登录帐户、创建存储库、上传和下载文件等。它还具有配置计算机或管理缓存以及启动和管理作业的便捷功能。

在 `huggingface_hub` 文档中查找安装步骤和指南：

* [Installation](https://huggingface.co/docs/huggingface_hub/en/installation)
* [Run and manage Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs)

## HTTP API

作业 HTTP API 端点在 `https://huggingface.co/api/jobs` 下可用。使用 Hugging 面部令牌进行身份验证，并有权在您的命名空间（您的帐户或组织）下启动和管理作业。
将令牌作为不记名令牌传递，其标头为：`"Authorization: Bearer {token}"`。

以下是可用端点和参数的列表：

* [View Jobs OpenAPI](https://huggingface-openapi.hf.space/#tag/jobs)

### 型号发布清单
https://huggingface.co/docs/hub/model-release-checklist.md