<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 使用 CircleCI 工作流程管理空间

您可以使用 **CircleCI 工作流程** 使您的应用程序与 GitHub 存储库保持同步。 

[CircleCI](https://circleci.com) 是一个持续集成和持续交付 (CI/CD) 平台，有助于自动化软件开发过程。 [CircleCI workflow](https://circleci.com/docs/workflows/) 是在配置文件中定义的一组自动化任务，由 CircleCI 编排，用于简化构建、测试和部署软件应用程序的过程。

*注意：对于大于 10MB 的文件，Spaces 需要 Git-LFS。如果您不想使用 Git-LFS，您可能需要查看您的文件并检查您的历史记录。使用像 [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/) 这样的工具从历史记录中删除任何大文件。 BFG Repo-Cleaner 将保留您的存储库的本地副本作为备份。*

首先，一起设置 GitHub 存储库和 Spaces 应用程序。将您的 Spaces 应用程序作为附加遥控器添加到现有 Git 存储库。

```bash
git remote add space https://huggingface.co/spaces/HF_USERNAME/SPACE_NAME
```

然后强制推送以第一次同步所有内容：

```bash
git push --force space main
```

接下来，设置一个 [CircleCI workflow](https://circleci.com/docs/workflows/) 将您的 `main` git 分支推送到 Spaces。 

在下面的例子中：* 将 `HF_USERNAME` 替换为您的用户名，将 `SPACE_NAME` 替换为您的空间名称。 
* [Create a context in CircleCI](https://circleci.com/docs/contexts/) 并向其中添加一个名为 *HF_PERSONAL_TOKEN* 的环境变量（您可以给它任何名称，使用您创建的密钥代替 HF_PERSONAL_TOKEN），并将值作为您的 Hugging Face API 令牌。您可以在 [your Hugging Face profile](https://huggingface.co/settings/tokens) 上的 **API 令牌** 下找到您的 Hugging Face API 令牌。

```yaml
version: 2.1

workflows:
  main:
    jobs:
      - sync-to-huggingface:
          context:
            - HuggingFace
          filters:
            branches:
              only:
                - main

jobs:
  sync-to-huggingface:
    docker:
      - image: alpine
    resource_class: small
    steps:
      - run: 
          name: install git
          command: apk update && apk add openssh-client git
      - checkout
      - run:
          name: push to Huggingface hub
          command: |
                  git config user.email "<your-email@here>" 
                  git config user.name "<your-identifier>" 
                  git push -f https://HF_USERNAME:${HF_PERSONAL_TOKEN}@huggingface.co/spaces/HF_USERNAME/SPACE_NAME main
```

### 转换你的数据集
https://huggingface.co/docs/hub/datasets-polars-operations.md