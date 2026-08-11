<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 为 PEFT 做出贡献

我们很高兴接受对 PEFT 的贡献。如果您打算做出贡献，请阅读本文，以使过程尽可能顺利。

## 安装

请按照以下步骤开始贡献：

1. 通过单击存储库页面上的“分叉”按钮来分叉 [repository](https://github.com/huggingface/peft)。这将在您的 GitHub 用户帐户下创建代码的副本。

2. 将您的分支克隆到本地磁盘，并将基本存储库添加为远程存储库。以下命令假设您已将 SSH 公钥上传到 GitHub。请参阅以下指南了解更多[information](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)。

   ```bash
   git clone git@github.com:<your Github handle>/peft.git
   cd peft
   git remote add upstream https://github.com/huggingface/peft.git
   ```

3. 创建一个新分支来保存您的开发更改，并对您处理的每个新 PR 执行此操作。

   首先将 `main` 分支与 `upstream/main` 分支同步（更多详细信息请参见 [GitHub Docs](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork)）：

   ```bash
   git checkout main
   git fetch upstream
   git merge upstream/main
   ```

   同步您的 `main` 分支后，从中创建一个新分支：

   ```bash
   git checkout -b a-descriptive-name-for-my-changes
   ```

   **不要**在 `main` 分支上工作。

4. 通过在 conda 或您为处理此库而创建的虚拟环境中运行以下命令来设置开发环境：

   ```bash
   pip install -e ".[test]"
   ```

   （如果虚拟环境中已经安装了PEFT，请先使用`pip uninstall peft`将其删除，然后再重新安装。）如果您不熟悉创建拉取请求，请遵循 GitHub 的 [Creating a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) 指南。

## 测试和代码质量检查

无论贡献类型如何（除非仅涉及文档），您都应该在创建 PR 之前运行测试和代码质量检查，以确保您的贡献不会破坏任何内容并遵循项目标准。

我们提供一个 Makefile 来执行必要的测试。运行以下代码进行单元测试：

```sh
make test
```

运行以下命令之一以仅检查或检查并修复代码质量和样式：

```sh
make quality  # just check
make style  # check and fix
```

您还可以设置 [⟦T13⟧](https://pre-commit.com/) 来运行这些修复
自动作为 Git 提交挂钩。

```bash
$ pip install pre-commit
$ pre-commit install
```

运行所有测试可能需要一段时间，因此在开发过程中，仅使用 [run tests specific to your change](https://docs.pytest.org/en/6.2.x/usage.html#specifying-tests-selecting-tests) 会更高效，例如通过：

```sh
pytest tests/<test-file-name> -k <name-of-test>
```

这应该更快地完成并允许更快的迭代。

如果您的更改特定于硬件设置（例如，它需要 CUDA），请查看 [⟦T14⟧](https://github.com/huggingface/peft/blob/1c1c7fdaa6e6abaa53939b865dee1eded82ad032/tests/test_gpu_examples.py) 和 [⟦T15⟧](https://github.com/huggingface/peft/blob/1c1c7fdaa6e6abaa53939b865dee1eded82ad032/tests/test_common_gpu.py)，看看在那里添加测试是否有意义。如果您的更改可能会影响保存和加载模型，请使用 `--regression` 标志运行测试以触发回归测试。当您处理 PR 时，底层代码库可能会由于合并其他更改而发生更改。如果发生这种情况 - 特别是当存在合并冲突时 - 请使用最新更改更新您的分支。这可以是合并或变基，一旦准备好，我们将压缩并合并 PR。如果可能的话，**避免强行推动**以使审核更容易。

## 公关说明

当打开 PR 时，请提供您所提议的更改的详细描述。如果涉及其他问题或 PR，请参考。提供良好的描述不仅可以帮助审阅者更好更快地审阅您的代码，还可以在以后（作为基础）用于提交消息，从而有助于项目的长期维护。

保持 PR 描述的长度和复杂性与变化保持一致。我们不需要对一行微不足道的更改进行十段解释。不要重述从差异中显而易见的内容（例如“修复了‘foobaar’中的拼写错误）。如果您的代码进行了一些重要的更改，那么向代码添加注释来解释这些更改可能也是一个好主意。例如，如果由于最明显的方法不起作用而必须多次迭代实现，那么这就很好地表明需要代码注释。

如果相关，请说明您如何测试更改，例如通过显示 `pytest` 测试命令或重现器代码。

## 审稿人反馈

提交 PR 后，维护人员通常会在几天内提供反馈。如果您在两周内没有收到任何反馈，您的 PR 可能已经忽视了；届时请随时联系维护人员，但不要更早。

如果审阅者提供内联评论，并且您已经解决了这些评论，则不要将其标记为已解决。让这些评论保持打开状态，因为它们正在帮助审稿人继续他们的工作。

在处理完审阅者的反馈后，请联系审阅者，以便他们知道 PR 已准备好进行审阅。

## 错误修复

请描述导致该错误的情况。如果存在问题，请链接到该问题（例如“解决#12345”）。理想情况下，当提供错误修复时，应该附带对该错误的测试。当前代码的测试应该失败，错误修复后测试应该通过。向测试添加引用问题或 PR 的评论。如果没有测试，将来就更难以防止回归。

## 文档改进

我们很高兴修复损坏的链接以及丢失或不清楚的文档。照顾示例，确保它们是最新的并且在这个快速变化的环境中运行良好也受到高度赞赏。

请避免发送“仅”纠正输入错误的拉取请求，因为这些请求通常会产生比安全更多的工作。此类更改最好​​与更实质性的修复（例如修复损坏的链接或扩展/更新文档）结合起来。

## 添加新的 PEFT 微调方法

新的参数有效的微调方法一直在开发。如果您想向 PEFT 添加一种新的、有前景的方法，请按照以下步骤操作。1. 如果您_不是_原始论文的作者，请检查现有的实现，并与作者仔细检查他们不打算自己提交 PR。
2. 从下面列出的核心集成工作开始。
3. 检查最近提交的新 PEFT 方法是否被添加作为灵感。
4. 一旦方法基本有效并且首先测试通过，然后寻求反馈，尽早打开 PR 草案会很有用。

### 新 PEFT 方法的核心集成

- [ ] 在投入太多工作之前，在`huggingface/peft`上打开一个提案问题。
- [ ] 链接方法的来源，通常是最终论文或其他稳定的主要参考文献。我们希望避免仍在审查中的工作，因为实施应该是稳定的。
- [ ] 在`src/peft/utils/peft_types.py`中添加新的`PeftType`条目。
- [ ] 在 `src/peft/tuners/` 下使用您的方法所需的文件创建一个新的调谐器包（通常：`config.py`、`model.py`、`layer.py` 和 `__init__.py`）。
- [ ] 将方法注册到调谐器`__init__.py`和`register_peft_method(...)`中。
- [ ] 从`src/peft/tuners/__init__.py`和`src/peft/__init__.py`导出新的配置/模型。
- [ ] 如果该方法需要Transformers模型的默认目标模块，请在`src/peft/utils/constants.py`中添加映射。- [ ] 将方法添加到`tests/test_custom_models.py` 中的测试矩阵中，因为这些是最广泛和最快的测试。检查测试是否通过`pytest tests/test_custom_models.py -k <method-name> -v`，修复失败（如果有）。
- [ ] 在推送之前使用 `make style` 运行样式/质量检查。
- [ ] 在 PR 描述中，解释方法、链接论文、总结权衡并列出添加的内容。

### 完整 PR 添加新的 PEFT 方法

- [ ] 确保特定于该方法的配置参数有良好的命名和解释，不要假设用户对本文了如指掌。
- [ ] 遵循 PEFT 的命名和编码约定。
- [ ] 确保您没有意外签入不相关的更改，例如代码格式化程序更改不相关的文件。
- [ ] 如果某些实现选择很重要，请使用代码注释来记录它们。
- [ ] 通过将 PEFT 方法添加到测试矩阵来完成完整的测试套件（`test_config.py`、`test_decoder_models.py` 等）。确保测试通过。
- [ ] 在 `docs/source/package_reference/` 中添加文档，其中包含简短说明、论文链接、用法片段和自动文档块。解释与 LoRA 等其他方法相比的优缺点。在 `docs/source/_toctree.yml` 中注册该文档页面。- [ ] 在`examples/`下添加一个可运行的示例（可以是现有示例的副本），带有简短的`README.md`。
- [ ] 检查`method_comparison/`中的基准测试并为您的新方法添加实验设置。这是一个健全性检查 PEFT 方法是否按预期进行训练的好地方。包括一到两个合理的基准配置（一种默认配置，一种针对基准优化）。
- [ ] 推荐：添加通用量化支持。不必为每种量化方法显式添加量化层类型，而是支持通用量化。作为示例，检查它在[BOFT](https://github.com/huggingface/peft/tree/main/src/peft/tuners/boft)中的实现方式。通过在其中添加 PEFT 方法来扩展 https://github.com/huggingface/peft/blob/main/tests/test_quantization.py 。如果需要，请向维护人员寻求帮助。

## 添加其他功能

最好先在 GitHub 上提出一个问题，并提出添加新功能的建议。这样，您可以在花费太多时间实现该功能之前与维护人员讨论添加该功能是否有意义。

新功能通常应附有测试和文档或示例。如果没有后者，用户将很难发现你很酷的新功能。对代码的更改应该以向后兼容的方式实现。例如，合并功能后，现有代码应继续以相同的方式工作。

### 故障排除
https://huggingface.co/docs/peft/v0.20.0/developer_guides/troubleshooting.md