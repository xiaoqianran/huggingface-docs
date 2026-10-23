<!-- huggingface-docs: machine-translated zh-CN from English source -->

# GGUF 与 GPT4All 的使用

[GPT4All](https://gpt4all.io/)是[Nomic](https://nomic.ai/)开发的开源LLM应用程序。 2.7.2 版本引入了一个全新的实验性功能，称为 `Model Discovery`。

`Model Discovery` 提供了一种从 Hub 搜索和下载 GGUF 模型的内置方法。首先，打开 GPT4All 并单击 `Download Models`。从这里，您可以使用搜索栏查找模型。

选择并下载模型后，您可以转到`Settings`并以GPT4All格式提供适当的提示模板（`%1`和`%2`占位符）。

然后，在主页上，您可以从已安装模型列表中选择模型并开始对话。

### 在拥抱脸部时使用句子转换器
https://huggingface.co/docs/hub/sentence-transformers.md