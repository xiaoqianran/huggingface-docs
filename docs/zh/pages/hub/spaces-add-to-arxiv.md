<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 如何向 ArXiv 添加空间

Hugging Face Spaces 的演示让广大观众可以尝试最先进的机器
无需编写任何代码即可学习研究。 [Hugging Face and ArXiv have collaborated](https://huggingface.co/blog/arxiv) 
将这些演示直接嵌入到 ArXiv 的副论文中！

由于这种集成，用户现在可以在其 arXiv 摘要页面上找到最流行的论文演示。例如，如果您想尝试 LayoutLM 文档分类模型的演示，您可以转到 [the LayoutLM paper's arXiv page](https://arxiv.org/abs/1912.13318)，然后导航到演示选项卡。您将看到机器学习社区为此模型构建的开源演示，您可以立即在浏览器中试用：

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/layout-lm-space-arxiv.gif)

我们将介绍两种不同的方法将您的 Space 添加到 ArXiv 并将其显示在“演示”选项卡中。 

**先决条件**

* ArXiv 上已有一篇论文，您想为其创建一个演示
* 您已在 Spaces 上构建或（可以构建）模型的演示

**方法 1（推荐）：从 Space README 链接**

将 Space 添加到 ArXiv 论文的最简单方法是将论文链接包含在 Space README 文件 (`README.md`) 中。最好也包含完整的引文。您可以在此 [Echocardiogram Segmentation Space README](https://huggingface.co/spaces/abidlabs/echocardiogram-arxiv/blob/main/README.md) 上查看链接和引用的示例。就是这样！几分钟后，您的空间就会出现在 ArXiv 论文旁边的“演示”选项卡中 🤗

**方法2：链接相关模型**

另一种方法可以通过将中间模型链接到空间来将空间链接到论文。这要求论文**与 Hugging Face Hub 上的模型相关联**（或者可以上传到那里）

1. 首先，将与 ArXiv 论文相关的模型上传到 Hugging Face Hub（如果尚未存在）。 ([Detailed instructions are here](./models-uploading))

2. 为模型编写模型卡 (README.md) 时，请包含 ArXiv 论文的链接。最好也包含完整的引文。您可以在 [LayoutLM model card](https://huggingface.co/microsoft/layoutlm-base-uncased) 上查看链接和引用的示例

    *注意*：您可以通过查看型号卡上方是否出现 ArXiv 按钮来验证此步骤是否已成功执行。对于 LayoutLM，按钮显示：“arxiv:1912.13318”并链接到 ArXiv 上的 LayoutLM 论文。

    ![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/arxiv-button.png)

3. 然后，在 Spaces 上创建一个加载此模型的演示。在代码中的某个位置，必须包含模型名称，以便 Hugging Face 检测到与其关联的 Space。例如，[docformer_for_document_classification](https://huggingface.co/spaces/iakarshu/docformer_for_document_classification) Space 加载 LayoutLM [like this](https://huggingface.co/spaces/iakarshu/docformer_for_document_classification/blob/main/modeling.py#L484) 并包含字符串 `"microsoft/layoutlm-base-uncased"`：

    ```py
    from transformers import LayoutLMForTokenClassification
    
    layoutlm_dummy = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=1)
    ```

    *注意*：这是[overview on building demos on Hugging Face Spaces](./spaces-overview)，这里是[Gradio](./spaces-sdks-gradio)和[Streamlit](./spaces-sdks-streamlit)的更具体说明。 

4. 一旦您的空间建成，Hugging Face 就会检测到它与模型关联。 “链接模型”按钮应出现在空间的右上角，如下所示： 

    ![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/linked-models.png)
    
    *注意*：您还可以通过在 [README metadata for the Space, as described here](https://huggingface.co/docs/hub/spaces-config-reference) 中显式更新链接模型来手动添加链接模型。

几分钟后，您的空间就会出现在 ArXiv 论文旁边的“演示”选项卡中 🤗

### 使用小行星拥抱脸部
https://huggingface.co/docs/hub/asteroid.md