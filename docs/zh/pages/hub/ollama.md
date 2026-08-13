<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在 Hugging Face Hub 上将 Ollama 与任何 GGUF 模型一起使用

![cover](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/ollama/cover.png)

您还可以从 Hugging Face Hub 运行私有 GGUF。

Ollama 是一个基于 llama.cpp 的应用程序，可通过您的计算机直接与 LLM 交互。您可以直接通过 Ollama 在 Hugging Face 上使用社区创建的任何 GGUF 量化（[bartowski](https://huggingface.co/bartowski)、[MaziyarPanahi](https://huggingface.co/MaziyarPanahi) 和 [many more](https://huggingface.co/models?pipeline_tag=text-generation&library=gguf&sort=trending)），而无需创建新的 `Modelfile`。截至撰写本文时，Hub 上有 45K 个公共 GGUF 检查点，您可以使用单个 `ollama run` 命令运行其中任何一个。我们还提供选择量化类型、系统提示等自定义功能，以改善您的整体体验。 

入门非常简单：

1. 启用[Local Apps settings](https://huggingface.co/settings/local-apps)下的`ollama`。
2. 在模型页面上，从 `Use this model` 下拉列表中选择 `ollama`。例如：[bartowski/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF)。

该片段的格式为：

```sh
ollama run hf.co/{username}/{repository}
```

请注意，您可以同时使用`hf.co`和`huggingface.co`作为域名。

您可以尝试以下一些模型：

```sh
ollama run hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
ollama run hf.co/mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF
ollama run hf.co/arcee-ai/SuperNova-Medius-GGUF
ollama run hf.co/bartowski/Humanish-LLama3-8B-Instruct-GGUF
```

## 自定义量化

默认情况下，当模型存储库中存在 `Q4_K_M` 量化方案时，会使用该方案。如果没有，我们默认选择存储库中存在的一种合理的量化类型。

要选择不同的方案，只需：1. 从模型页面上的`Files and versions`选项卡，打开特定 GGUF 文件的 GGUF 查看器。
2. 从`Use this model`下拉列表中选择`ollama`。

该片段的格式为（添加量化标签）：

```sh
ollama run hf.co/{username}/{repository}:{quantization}
```

例如：

```sh
ollama run hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:IQ3_M
ollama run hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0

# the quantization name is case-insensitive, this will also work
ollama run hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:iq3_m

# you can also directly use the full filename as a tag
ollama run hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-IQ3_M.gguf
```

## 自定义聊天模板和参数

默认情况下，将从常用模板列表中自动选择一个模板。它将根据 GGUF 文件中存储的内置 `tokenizer.chat_template` 元数据进行选择。

如果您的 GGUF 文件没有内置模板，或者您想自定义聊天模板，您可以在存储库中创建一个名为 `template` 的新文件。该模板必须是 Go 模板，而不是 Jinja 模板。这是一个例子：

```
{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>
```

了解更多Go模板格式请参考[this documentation](https://github.com/ollama/ollama/blob/main/docs/template.mdx)

您可以选择配置系统提示符，将其放入存储库中名为 `system` 的新文件中。

要更改采样参数，请在存储库中创建一个名为 `params` 的文件。该文件必须是 JSON 格式。所有可用参数列表，请参阅[this documentation](https://github.com/ollama/ollama/blob/main/docs/modelfile.mdx#parameter)。

## 从 Hugging Face Hub 运行私有 GGUF

您可以通过两个简单的步骤从您的个人帐户或关联的组织帐户运行私有 GGUF：1. 复制您的 Ollama SSH 密钥，您可以通过以下方式进行操作：`cat ~/.ollama/id_ed25519.pub | pbcopy`
2. 前往[your account settings](https://huggingface.co/settings/keys)并点击`Add new SSH key`，将相应的密钥添加到您的Hugging Face帐户中。
3. 就是这样！您现在可以从 Hugging Face Hub 运行私有 GGUF：`ollama run hf.co/{username}/{repository}`。

## 参考文献

- https://github.com/ollama/ollama/blob/main/docs/README.md
- https://huggingface.co/docs/hub/en/gguf

### 组织博客文章
https://huggingface.co/docs/hub/enterprise-blog-articles.md