<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 存储库入门

这个适合初学者的指南将帮助您获得在 Hub 上创建和管理存储库所需的基本技能。每个部分都建立在前一个部分的基础上，因此请随意选择从哪里开始！

## 要求

本文档展示了如何通过 Web 界面以及终端处理存储库。如果使用 UI，则没有任何要求。如果您想使用终端，请遵循这些安装说明。

如果您还没有 `git` 可用作 CLI 命令，则需要为您的平台提供 [install Git](https://git-scm.com/downloads)。您还需要[install Git-Xet](./xet/using-xet-storage#git-xet)，它将用于处理大文件，例如图像和模型权重。

> [!提示]
> 为了能够从 Git 下载和上传大文件，您需要安装 [Git Xet](./xet/using-xet-storage#git) 扩展。

为了能够将您的代码推送到中心，您需要以某种方式进行身份验证。最简单的方法是安装 [⟦T8⟧ CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) 并运行登录命令：

```bash
# Install hf:
#   brew install hf
#   or
#   pip install hf
hf auth login
```

**本文档“入门”部分中的内容也以视频形式提供！**

## 创建存储库使用 Hub 的 Web 界面，您可以轻松创建存储库、添加文件（甚至是大文件！）、探索模型、可视化差异等等。 Hub 上有三种存储库，在本指南中，您将创建一个**模型存储库**用于演示目的。有关创建和管理模型、数据集和空间的信息，请参阅各自的文档。

1. 要创建新存储库，请访问[huggingface.co/new](http://huggingface.co/new)：

2. 指定存储库的所有者：这可以是您或您所属的任何组织。 

3. 输入您的型号名称。这也将是存储库的名称。 

4. 指定您希望模型是公开的还是私有的。

5. 指定许可证。您现在可以将*许可证*字段留空。要了解许可证，请访问 [**Licenses**](repositories-licenses) 文档。

创建模型存储库后，您应该看到如下页面：

请注意，中心会提示您创建*模型卡*，您可以在[**Model Cards documentation**](./model-cards)中了解它。在模型存储库中包含模型卡是最佳实践，但由于我们目前仅制作测试存储库，因此我们可以跳过此操作。

## 将文件添加到存储库（Web UI）要通过 Web UI 将文件添加到存储库，请首先选择“**文件**”选项卡，导航到所需的目录，然后单击“**添加文件**”。您可以选择创建新文件或直接从计算机上传文件。 

### 创建一个新文件

选择创建新文件将带您进入以下编辑器屏幕，您可以在其中选择文件名称、添加内容并保存文件并显示一条总结更改的消息。您可以选择 `Open as a pull request` 创建一个 [Pull Request](./repositories-pull-requests-discussions)，而不是直接将新文件提交到存储库的 `main` 分支。

### 上传文件

如果您选择_上传文件_，您将能够选择要上传的本地文件，以及一条总结您对存储库所做更改的消息。

与创建新文件一样，您可以选择 `Open as a pull request` 创建 [Pull Request](./repositories-pull-requests-discussions)，而不是将更改直接添加到存储库的 `main` 分支。

## 将文件添加到存储库 (CLI)[[cli]]

您可以使用 [⟦T13⟧ CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) 直接从终端将文件上传到存储库。使用`hf upload`命令推送本地文件或整个文件夹：

```bash
# Upload a single file to your model repo
hf upload your-username/your-model-name model.safetensors

# Upload an entire directory
hf upload your-username/your-model-name ./my-model-directory

# Upload to a dataset repo
hf upload your-username/your-dataset-name ./data --repo-type dataset
```

`hf` CLI 自动处理大文件 — 无需额外设置。## 将文件添加到存储库 (git)[[terminal]]

### 克隆存储库

将存储库下载到本地计算机称为*克隆*。您可以使用以下命令加载存储库并导航到它：

```bash
git clone https://huggingface.co/<your-username>/<your-model-name>
cd <your-model-name>
```

或者对于数据集存储库：

```bash
git clone https://huggingface.co/datasets/<your-username>/<your-dataset-name>
cd <your-dataset-name>
```

您可以使用以下命令通过 SSH 进行克隆：
```bash
git clone git@hf.co:<your-username>/<your-model-name>
cd <your-model-name>
```

您需要将 SSH 公钥添加到 [your user settings](https://huggingface.co/settings/keys) 才能推送更改或访问私有存储库。

### 设置

现在，您可以将任何想要的文件添加到存储库中！ 🔥

您有大于 10MB 的文件吗？这些文件应该使用 [⟦T16⟧](./xet/using-xet-storage#git-xet) 进行跟踪，您可以使用以下命令进行初始化：

```bash
git xet install
```

当您使用 Hugging Face 创建存储库时，Hugging Face 会自动在 `.gitattributes` 文件中提供常见机器学习大文件的常见文件扩展名列表，`git-xet` 使用该列表来有效跟踪大文件的更改。但是，如果您的文件类型尚未处理，您可能需要添加新的扩展名。您可以使用 `git xet track "*.your_extension"` 来实现。

### 推送文件您可以使用 Git 将新文件以及对现有文件的任何更改保存为一组称为“提交”的更改，这可以被视为对项目的“修订”。要创建提交，您必须 `add` 文件让 Git 知道我们计划保存更改，然后 `commit` 这些更改。为了将新提交与 Hugging Face Hub 同步，您需要 `push` 将提交提交到 Hub。

```bash
# Create any files you like! Then...
git add .
git commit -m "First model version"  # You can choose any descriptive message
git push
```

你就完成了！您可以在 Hugging Face 上检查您的存储库以及所有最近添加的文件。例如，在下面的屏幕截图中，用户添加了许多文件。请注意，此示例中的某些文件的大小为 `1.04 GB`，因此存储库使用 Xet 来跟踪它。

> [!提示]
> 如果您使用 HTTP 克隆存储库，则可能会要求您在每次推送操作时填写用户名和密码。避免重复的最简单方法是使用[switch to SSH](#cloning-repositories)，而不是 HTTP。或者，如果您必须使用 HTTP，您可能会发现设置 [git credential helper](https://git-scm.com/docs/gitcredentials#_avoiding_repetition) 自动填充您的用户名和密码很有帮助。

## 查看存储库的历史记录每次您经历 `add`-`commit`-`push` 循环时，存储库都会跟踪您对文件所做的每项更改。 UI 允许您探索模型文件和提交，并查看每个提交引入的差异（也称为 *diff*）。要查看历史记录，您可以单击 **历史记录：X 提交** 链接。

您可以单击单个提交来查看该提交引入了哪些更改：

### 使用 GPU 空间
https://huggingface.co/docs/hub/spaces-gpus.md