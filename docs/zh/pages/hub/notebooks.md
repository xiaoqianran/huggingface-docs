<!-- huggingface-docs: machine-translated zh-CN from English source -->

# Hugging Face Hub 上的 Jupyter Notebooks

[Jupyter notebooks](https://jupyter.org/) 是一种非常流行的共享机器学习和数据科学代码和数据分析的格式。它们是交互式文档，可以包含代码、可视化和文本。

## 在 Google Colab 和 Kaggle 中打开模型

当您访问 Hugging Face Hub 上的模型页面时，您会在“使用此模型”下拉列表中看到一个新的“Google Colab”/“Kaggle”按钮。单击此按钮将生成一个可立即运行的笔记本，其中包含用于加载和测试模型的基本代码。这非常适合快速原型设计、推理测试或微调实验 - 所有这些都无需离开浏览器。

![Google Colab and Kaggle option for models on the Hub](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hf-google-colab/gemma3-4b-it-dark.png)

用户还可以通过将 /colab 附加到模型卡的 URL 来访问可立即运行的笔记本。以最新的 Gemma 3 4B IT 型号为例，通过型号卡 URL 即可访问对应的 Colab 笔记本：
https://huggingface.co/google/gemma-3-4b-it

然后将 `/colab` 附加到它上面：
https://huggingface.co/google/gemma-3-4b-it/colab

对于 Kaggle 来说也是如此：
https://huggingface.co/google/gemma-3-4b-it/kaggle如果模型存储库包含名为 `notebook.ipynb` 的文件，我们将把它用于 Colab 和 Kaggle，而不是自动生成的笔记本内容。模型作者可以提供定制示例、详细演练或高级用例，同时仍然受益于一键式 Colab 集成。 [NousResearch/Genstruct-7B](https://huggingface.co/NousResearch/Genstruct-7B) 就是这样的一个例子。

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hf-google-colab/genstruct-notebook-dark.png)

## 在 Hub 上渲染 .ipynb Jupyter 笔记本

在底层，Jupyter Notebook 文件（通常以 `.ipynb` 扩展名共享）是 JSON 文件。虽然可以直接查看这些文件，但它不是供人类阅读的格式。 Hub 为 Hub 上托管的笔记本提供渲染支持。这意味着笔记本以人类可读的格式显示。

![Before and after notebook rendering](https://huggingface.co/blog/assets/135_notebooks-hub/before_after_notebook_rendering.png)

当笔记本包含在 Hub 上的任何类型的存储库中时，都会呈现笔记本。这包括模型、数据集和空间。

### 在 Google Colab 中启动

[Google Colab](https://colab.google/) 是一个免费的 Jupyter Notebook 环境，无需设置，完全在云端运行。这是运行 Jupyter Notebooks 的好方法，无需在本地计算机上安装任何内容。 

Hub 上托管的所有 .ipynb 文件都会自动获得“在 Colab 中打开”按钮。这样您只需单击一下即可在 Colab 中打开笔记本。### 存储桶集成
https://huggingface.co/docs/hub/storage-buckets-integrations.md