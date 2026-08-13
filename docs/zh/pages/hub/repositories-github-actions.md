<!-- huggingface-docs: machine-translated zh-CN from English source -->

# GitHub 操作

您可以使用 [GitHub Actions](https://docs.github.com/en/actions) 自动将 GitHub 存储库同步到 Hugging Face Hub。官方[⟦T2⟧](https://github.com/marketplace/actions/sync-github-to-hugging-face-hub)操作支持同步**模型**、**数据集**和**空间**。

> [!提示]
> 对于无密钥发布 - 无需存储或轮换 `HF_TOKEN` 秘密 - 请参阅 [Trusted Publishers](./trusted-publishers)，它在每次运行开始时将 GitHub Actions 的内置 OIDC 令牌交换为短期的、存储库范围的 Hub 令牌。

## 设置

1. 创建一个拥抱脸[access token](https://huggingface.co/settings/tokens)，并对目标存储库具有**写入**权限。为了获得更好的安全性，请使用范围仅限于您要同步到的存储库的[fine-grained token](https://huggingface.co/settings/tokens)。
2. 在存储库设置中添加名为 `HF_TOKEN` 的令牌作为 [GitHub secret](https://docs.github.com/en/actions/security-guides/encrypted-secrets#creating-encrypted-secrets-for-an-environment)。
3. 将工作流程文件（例如`.github/workflows/sync-to-hub.yml`）添加到您的存储库。

## 基本用法

```yaml
name: Sync to Hugging Face Hub
on:
  push:
    branches: [main]

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: huggingface/hub-sync@v0.1.0
        with:
          github_repo_id: ${{ github.repository }}
          huggingface_repo_id: username/repo-name
          hf_token: ${{ secrets.HF_TOKEN }}
```

默认情况下，这会同步到 **Space**。要同步模型或数据集，请设置 `repo_type` 参数：

```yaml
      - uses: huggingface/hub-sync@v0.1.0
        with:
          github_repo_id: ${{ github.repository }}
          huggingface_repo_id: username/my-dataset
          hf_token: ${{ secrets.HF_TOKEN }}
          repo_type: dataset
```

## 参数|参数|必填 |默认|描述 |
|---|---|---|---|
| `github_repo_id` |是的 | — | GitHub 存储库（使用`${{ github.repository }}`）|
| `huggingface_repo_id` |是的 | — | Hub 上的目标存储库 (`username/repo-name`) |
| `hf_token` |是的 | — |拥抱脸部访问令牌 |
| `repo_type` |没有 | `space` | `space`、`model` 或 `dataset` |
| `space_sdk` |没有 | `gradio` | `gradio`、`streamlit`、`docker` 或 `static` |
| `private` |没有 | `false` |是否将存储库创建为私有 |
| `subdirectory` |没有 | `.` |同步特定子目录（对于 monorepos 有用）|

该操作使用 `hf` CLI 将您的文件镜像到 Hub — 它不是 git 到 git 同步。它会自动排除 `.github/` 和 `.git/` 目录和镜像删除（从 GitHub 删除的文件也将从 Hub 中删除）。

对于更复杂的工作流程（例如构建步骤、自定义上传逻辑），您可以直接在工作流程中安装和使用 [⟦T30⟧ CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)。

有关特定于空间的指南（文件大小限制、LFS 处理），请参阅 [Managing Spaces with GitHub Actions](./spaces-github-actions)。

### 快速入门
https://huggingface.co/docs/hub/jobs-quickstart.md