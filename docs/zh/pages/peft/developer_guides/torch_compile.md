<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 火炬.编译

在 PEFT 中，[torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) 适用于某些但不是所有功能。它并不总是有效的原因是 PEFT 在某些地方是高度动态的（例如在多个适配器之间加载和切换），这可能会给 `torch.compile` 带来麻烦。在其他地方，`torch.compile` 可能有效，但由于图形中断而不会像预期的那么快。

如果您没有看到错误，并不一定意味着 `torch.compile` 工作正常。它可能会给您一个输出，但输出不正确。本指南描述了哪些内容适用于 `torch.compile`，哪些内容不适用。对于您自己的测试，我们建议使用最新的 PyTorch 版本，因为 `torch.compile` 正在不断改进。

> [!提示]
> 除非另有说明，否则使用默认的 `torch.compile` 设置。

## 使用 `torch.compile` 进行训练和推理

这些功能**适用于**`torch.compile`。下面列出的所有内容均已使用因果 LM 进行了测试：

- 使用来自🤗变形金刚的`Trainer`进行训练
- 使用自定义 PyTorch 循环进行训练
- 推理
- 一代

以下适配器已成功测试：

-阿达洛拉
——英国广播电视公司
- IA3
- 层规范调整
- 乐哈
- 洛克尔
- 洛拉
- 洛拉+多拉
- LoRA应用于嵌入层
- 奥夫特
- 维拉
- 人力资源管理局

## `torch.compile` 的高级 PEFT 功能以下是一些**有效**的更高级的 PEFT 功能。它们都经过了 LoRA 测试。

- `modules_to_save`（即`config = LoraConfig(..., modules_to_save=...)`）
- 合并适配器（一个或多个）
- 将多个适配器合并为一个适配器（即调用`model.add_weighted_adapter(...)`）
- 使用具有量化功能的 PEFT 适配器（位和字节）
- 禁用适配器（即使用`with model.disable_adapter()`）
- 卸载（即调用`model.merge_and_unload()`）
- 混合适配器批次（即调用`model(batch, adapter_names=["__base__", "default", "other", ...])`）
- 使用多个适配器进行推理（即使用`model.add_adapter`或`model.load_adapter`加载超过1个适配器）；为此，仅在加载所有适配器之后调用 `torch.compile`

一般来说，我们可以预期，如果某个功能可以与 LoRA 一起正常工作并且也受到其他适配器类型的支持，那么它也应该适用于该适配器类型。

## 测试用例

上面列出的所有用例都在 [⟦T19⟧](https://github.com/huggingface/peft/blob/main/tests/test_torch_compile.py) 内进行了测试。如果您想更详细地检查我们如何测试某个功能，请转到该文件并检查与您的用例相对应的测试。

> [!提示]
> 如果您有另一个用例，您知道 `torch.compile` 是否可以与 PEFT 配合使用，请通过告知我们或打开 PR 将此用例添加到涵盖的测试用例中来做出贡献。### 量化
https://huggingface.co/docs/peft/v0.20.0/developer_guides/quantization.md