<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在拥抱脸部时使用 MLX

[MLX](https://github.com/ml-explore/mlx)是苹果机器学习研究中心为苹果芯片打造的模型训练和服务框架。

它附带了各种示例：

- [Generate text with MLX-LM](https://github.com/ml-explore/mlx-lm/tree/main) 和 [generating text with MLX-LM for models in GGUF format](https://github.com/ml-explore/mlx-examples/tree/main/llms/gguf_llm)。
- 使用[LLaMA](https://github.com/ml-explore/mlx-examples/tree/main/llms/llama)进行大规模文本生成。
- 使用[LoRA](https://github.com/ml-explore/mlx-examples/tree/main/lora)进行微调。
- 使用[Stable Diffusion](https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion)生成图像。
- 使用[OpenAI's Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)进行语音识别。

## 在 Hub 上探索 MLX

您可以通过[models page](https://huggingface.co/models?library=mlx&sort=trending)左侧筛选找到MLX型号。
还有一个开放的 [MLX community](https://huggingface.co/mlx-community) 贡献者转换和发布 MLX 格式的权重。

由于 MLX Hugging Face Hub 集成，您可以使用几行代码加载 MLX 模型。 

## 安装

MLX 作为一个独立的包提供，还有一个名为 MLX-LM 的子包，其中包含用于大型语言模型的 Hugging Face 集成。
要安装MLX-LM，您可以通过`pip`使用以下一行安装：

```bash
pip install mlx-lm
```

您可以获取更多相关信息[here](https://github.com/ml-explore/mlx-lm/tree/main)。

如果安装`mlx-lm`，则无需安装`mlx`。如果您不想使用`mlx-lm`而只想使用MLX，您可以按如下方式安装MLX本身。

对于 `pip`：

```bash
pip install mlx
```

对于 `conda`：

```bash
conda install -c conda-forge mlx
```

## 使用现有模型MLX-LM 具有生成文本的有用实用程序。以下行直接下载并加载模型并开始生成文本。

```bash
python -m mlx_lm.generate --model mistralai/Mistral-7B-Instruct-v0.2 --prompt "hello"
```

要获取生成选项的完整列表，请运行

```bash
python -m mlx_lm.generate --help
```

您还可以加载模型并开始通过 Python 生成文本，如下所示：

```python
from mlx_lm import load, generate

model, tokenizer = load("mistralai/Mistral-7B-Instruct-v0.2")

response = generate(model, tokenizer, prompt="hello", verbose=True)
```

MLX-LM 支持流行的 LLM 架构，包括 LLaMA、Phi-2、Mistral 和 Qwen。除受支持的型号外，您可以轻松下载以下型号：

设置 `HF_XET_HIGH_PERFORMANCE=1` 会提高具有高带宽和至少 64 GB RAM 的计算机的并发范围和缓冲区大小：

```bash
pip install -U huggingface_hub

export HF_XET_HIGH_PERFORMANCE=1
hf download --local-dir <LOCAL FOLDER PATH> <USER_ID>/<MODEL_NAME>
```

## 转换和共享模型

您可以从 Hugging Face Hub 转换并可选择量化 LLM，如下所示： 

```bash
python -m mlx_lm.convert --hf-path mistralai/Mistral-7B-v0.1 -q 
```

如果你想在转换后直接推送模型，你可以像下面这样做。 

```bash
python -m mlx_lm.convert \
    --hf-path mistralai/Mistral-7B-v0.1 \
    -q \
    --upload-repo <USER_ID>/<MODEL_NAME>
```

## 其他资源

* [MLX Repository](https://github.com/ml-explore/mlx)
* [MLX Docs](https://ml-explore.github.io/mlx/)
* [MLX-LM](https://github.com/ml-explore/mlx-lm/tree/main)
* [MLX Examples](https://github.com/ml-explore/mlx-examples/tree/main)
* [All MLX models on the Hub](https://huggingface.co/models?library=mlx&sort=trending)

### 渐变空间
https://huggingface.co/docs/hub/spaces-sdks-gradio.md