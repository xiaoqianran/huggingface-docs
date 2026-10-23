<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 使用 GitHub Actions 管理空间

您可以使用官方 [⟦T2⟧](https://github.com/marketplace/actions/sync-github-to-hugging-face-hub) GitHub Action 使您的空间与 GitHub 存储库保持同步。

`hub-sync` 也适用于模型和数据集。一般用法请参见[GitHub Actions](./repositories-github-actions)。

## 设置

1. 创建一个名为 `HF_TOKEN` 的 [GitHub secret](https://docs.github.com/en/actions/security-guides/encrypted-secrets#creating-encrypted-secrets-for-an-environment)，并带有拥抱脸 [access token](https://huggingface.co/settings/tokens)。
2. 将工作流程文件（例如 `.github/workflows/sync-to-hub.yml`）添加到您的存储库：

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
      - uses: huggingface/hub-sync@v0.3.0
        with:
          github_repo_id: ${{ github.repository }}
          huggingface_repo_id: username/my-space
          hf_token: ${{ secrets.HF_TOKEN }}
```

您可以使用`space_sdk`（默认为`gradio`）配置Space SDK。参见[all parameters](./repositories-github-actions#parameters)。

## 它是如何工作的

该操作使用 `hf` CLI (`hf repos create` + `hf upload`) 将您的文件镜像到集线器。它不是 git-to-git 同步 - 它上传文件内容并自动排除 `.github/` 和 `.git/` 目录。从 GitHub 存储库中删除的文件也将从 Hub 中删除。

对于更复杂的工作流程（例如构建步骤、自定义逻辑），您可以直接在工作流程中安装和使用 [⟦T13⟧ CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)。

## 文件大小注意事项

该操作使用 `hf upload` 上传文件内容，因此您无需为 Hub 设置 Git LFS 或 git-xet：大文件会自动存储在 Xet 中。如果您的 GitHub 存储库使用 Git LFS 存储文件（GitHub 需要使用它来存储超过 100MB 的文件），请在 `actions/checkout` 步骤上设置 `lfs: true`，以便该操作上传文件而不是 LFS 指针。## 替代方案：手动 git 推送

如果您更喜欢直接 git 到 git 同步而不是文件镜像，您可以直接推送到空间的 git 远程。使用 git Push 时，必须使用 [git-xet](./xet/using-xet-storage#git-xet)（推荐）或 Git LFS 跟踪大于 10MB 的文件：

```yaml
name: Sync to Hugging Face hub
on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  sync-to-hub:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0
          lfs: true
      - name: Push to hub
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: git push https://HF_USERNAME:$HF_TOKEN@huggingface.co/spaces/HF_USERNAME/SPACE_NAME main
```

将 `HF_USERNAME` 替换为您的用户名，将 `SPACE_NAME` 替换为您的空间名称。

### Hugging Face Hub 文档
https://huggingface.co/docs/hub/index.md