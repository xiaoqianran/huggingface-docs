<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 任务

## 什么是任务？

任务或管道类型描述每个模型的 API（输入和输出）的“形状”，并用于确定我们要为任何给定模型显示哪个推理 API 和小部件。 

这种分类相对粗粒度（您始终可以在模型标签中添加更细粒度的任务名称），因此**您应该很少需要创建新任务**。如果您想添加对新任务的支持，本文档说明了所需的步骤。

## 概述

将新任务集成到 Hub 中意味着：
* 用户可以搜索给定任务的所有模型和数据集。
* Inference API 支持该任务。
* 用户可以直接使用小部件尝试模型。 🏆

请注意，您不需要自己实施所有步骤。添加新任务是社区的努力，多人可以做出贡献。 🧑‍🤝‍🧑

要开始此过程，请在 [huggingface_hub](https://github.com/huggingface/huggingface_hub/issues) 存储库中打开一个新问题。请使用“添加新任务”模板。 ⚠️在进行任何编码之前，建议先阅读此文档。 ⚠️第一步是上传您建议的任务的模型。在 Hub 中拥有用于新任务的模型后，下一步就是在 Inference API 中启用它。您可以选择三种类型的支持：

* 🤗 使用 `transformers` 模型
* 🐳 使用 [officially supported library](./models-libraries) 的模型
* 🖨️ 使用带有自定义推理代码的模型。此实验选项有缺点，因此我们建议使用其他方法之一。

最后，您可以添加几个 UI 元素，例如任务图标和小部件，以完成 Hub 中的集成。 📷 

有些步骤是正交的；你不需要按顺序做它们。 **您不需要 Inference API 来添加图标。** 这意味着，即使尚未完全集成，用户仍然可以搜索给定任务的模型。

## 添加新任务到 Hub

### 使用 Hugging Face 变压器库

如果您的模型是基于 `transformers` 的模型，则 Inference API 任务和 `pipeline` 类之间存在 1:1 映射。以下是来自 `transformers` 库的一些 PR 示例：
* [Adding ImageClassificationPipeline](https://github.com/huggingface/transformers/pull/11598)
* [Adding AudioClassificationPipeline](https://github.com/huggingface/transformers/pull/13342)

提交并部署管道后，您应该能够为您的模型使用 Inference API。

### 将社区推理 API 与受支持的库结合使用该中心还支持 [Community Inference API](https://github.com/huggingface/api-inference-community) 中的 10 多个开源库。 

**添加新任务相对简单，需要 2 个 PR：**
* PR 1：将新任务添加到 API [validation](https://github.com/huggingface/api-inference-community/blob/main/api_inference_community/validation.py)。此代码确保推理输入对于给定任务有效。一些公关示例：
    * [Add text-to-image](https://github.com/huggingface/huggingface_hub/commit/5f040a117cf2a44d704621012eb41c01b103cfca#diff-db8bbac95c077540d79900384cfd524d451e629275cbb5de7a31fc1cd5d6c189)
    * [Add audio-classification](https://github.com/huggingface/huggingface_hub/commit/141e30588a2031d4d5798eaa2c1250d1d1b75905#diff-db8bbac95c077540d79900384cfd524d451e629275cbb5de7a31fc1cd5d6c189)
    * [Add tabular-classification](https://github.com/huggingface/huggingface_hub/commit/dbea604a45df163d3f0b4b1d897e4b0fb951c650#diff-db8bbac95c077540d79900384cfd524d451e629275cbb5de7a31fc1cd5d6c189)
* PR 2：将新任务添加到库 docker 镜像中。您还应该向[⟦T4⟧](https://github.com/huggingface/api-inference-community/tree/main/docker_images/common/app/pipelines)添加一个模板，以方便将任务集成到其他库中。这是一个 PR 示例：
    * [Add text-classification to spaCy](https://github.com/huggingface/huggingface_hub/commit/6926fd9bec23cb963ce3f58ec53496083997f0fa#diff-3f1083a92ca0047b50f9ad2d04f0fe8dfaeee0e26ab71eb8835e365359a1d0dc)

### 添加社区推理 API 以实现快速原型

**我的模型不受任何库支持。我注定了吗？ 😱**

对于这些用例，我们建议使用 [Hugging Face Spaces](./spaces)。

### 用户界面元素

该中心允许用户按给定任务过滤模型。为此，您需要将任务添加到多个位置。您还可以为任务选择一个图标！

1. `Types.ts`添加任务类型

在[huggingface.js/packages/tasks/src/pipelines.ts](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/pipelines.ts)中，你需要做几件事

* 将类型添加到`PIPELINE_DATA`。请注意，管道类型分为不同的类别（NLP、音频、计算机视觉等）。
* 您还需要在[huggingface.js/packages/tasks/src/tasks/index.ts](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/tasks/index.ts)中填写较小的更改

2. 选择一个图标您可以在[lib/Icons](https://github.com/huggingface/huggingface.js/tree/main/packages/widgets/src/lib/components/Icons)目录中添加图标。我们通常从 https://icones.js.org/collection/carbon 中选择碳图标。还将图标添加到[PipelineIcon](https://github.com/huggingface/huggingface.js/blob/main/packages/widgets/src/lib/components/PipelineIcon/PipelineIcon.svelte)。

### 小部件

一旦任务投入生产，还有什么比实现某种方式让用户直接在浏览器中使用模型更令人兴奋的呢？ 🤩 你可以找到所有的小部件[here](https://huggingface.co/spaces/huggingfacejs/inference-widgets)。 

如果您有兴趣为小部件做出贡献，您可以查看所有小部件的[implementation](https://github.com/huggingface/huggingface.js/tree/main/packages/widgets)。

### 模型
https://huggingface.co/docs/hub/models.md