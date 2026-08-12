<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 古夫

Hugging Face Hub 支持所有文件格式，但具有[GGUF format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)的内置功能，这是一种针对快速加载和保存模型进行了优化的二进制格式，使其在推理方面非常高效。 GGUF 设计用于与 GGML 和其他执行器一起使用。 GGUF由[@ggerganov](https://huggingface.co/ggerganov)开发，他也是流行的C/C++ LLM推理框架[llama.cpp](https://github.com/ggerganov/llama.cpp)的开发者。最初在 PyTorch 等框架中开发的模型可以转换为 GGUF 格式，以便与这些引擎一起使用。

正如我们在此图中看到的，与仅张量的文件格式（如 [safetensors](https://huggingface.co/docs/safetensors)）（这也是 Hub 的推荐模型格式）不同，GGUF 对张量和一组标准化元数据进行编码。

## 查找 GGUF 文件

您可以浏览所有带有 GGUF 文件并通过 GGUF 标签过滤的模型：[hf.co/models?library=gguf](https://huggingface.co/models?library=gguf)。此外，您可以使用[ggml-org/gguf-my-repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo)工具将模型权重转换/量化为GGUF权重。

例如，您可以查看 [TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF) 查看运行中的 GGUF 文件。

## 元数据和张量信息查看器

Hub 有一个 GGUF 文件查看器，可让用户查看元数据和张量信息（名称、形状、精度）。查看器可在模型页面（[example](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF?show_tensors=mixtral-8x7b-instruct-v0.1.Q4_0.gguf)）和文件页面（[example](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/tree/main?show_tensors=mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf)）上使用。

## 与开源工具一起使用* [llama.cpp](./gguf-llamacpp)
* [LM Studio](./lmstudio)
* [GPT4All](./gguf-gpt4all)
* [Ollama](./ollama)

## 使用 @huggingface/gguf 解析元数据

我们还创建了一个 javascript GGUF 解析器，可以处理远程托管文件（例如 Hugging Face Hub）。

```bash
npm install @huggingface/gguf
```

```ts
import { gguf } from "@huggingface/gguf";
// remote GGUF file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
const URL_LLAMA = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/191239b/llama-2-7b-chat.Q2_K.gguf";
const { metadata, tensorInfos } = await gguf(URL_LLAMA);
```

查找更多信息[here](https://github.com/huggingface/huggingface.js/tree/main/packages/gguf)。

## 量化类型

|类型 |来源 |描述 |
|----------------------------|--------|-------------|
| F64| [Wikipedia](https://en.wikipedia.org/wiki/Double-precision_floating-point_format) | 64 位标准 IEEE 754 双精度浮点数。 |
| I64 | [GH](https://github.com/ggerganov/llama.cpp/pull/6062) | 64 位定宽整数。 |
| F32| [Wikipedia](https://en.wikipedia.org/wiki/Single-precision_floating-point_format) | 32 位标准 IEEE 754 单精度浮点数。 |
| I32 | [GH](https://github.com/ggerganov/llama.cpp/pull/6045) | 32 位定宽整数。 |
| F16| [Wikipedia](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) | 16 位标准 IEEE 754 半精度浮点数。 |
| BF16 | [Wikipedia](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) | 32 位 IEEE 754 单精度浮点数的 16 位缩短版本。 |
| I16 | [GH](https://github.com/ggerganov/llama.cpp/pull/6045) | 16 位定宽整数。 |
| Q8_K | [GH](https://github.com/ggerganov/llama.cpp/pull/1684#issue-1739619305) | 8 位量化 (`q`)。每个块有 256 个权重。仅用于量化中间结果。所有 2-6 位点积都是为此量化类型实现的。重量公式：`w = q * block_scale`。 |
| I8 | [GH](https://github.com/ggerganov/llama.cpp/pull/6045) | 8 位定宽整数。 || Q6_K | [GH](https://github.com/ggerganov/llama.cpp/pull/1684#issue-1739619305) | 6 位量化 (`q`)。超级块有 16 个块，每个块有 16 个权重。权重公式：`w = q * block_scale(8-bit)`，得出每个权重 6.5625 位。 |
| Q5_K | [GH](https://github.com/ggerganov/llama.cpp/pull/1684#issue-1739619305) | 5 位量化 (`q`)。超级块有8个块，每个块有32个权重。权重公式：`w = q * block_scale(6-bit) + block_min(6-bit)`，得出每个权重 5.5 位。 |
| Q4_K | [GH](https://github.com/ggerganov/llama.cpp/pull/1684#issue-1739619305) | 4 位量化 (`q`)。超级块有8个块，每个块有32个权重。权重公式：`w = q * block_scale(6-bit) + block_min(6-bit)`，得出每个权重 4.5 位。 |
| Q3_K | [GH](https://github.com/ggerganov/llama.cpp/pull/1684#issue-1739619305) | 3 位量化 (`q`)。超级块有 16 个块，每个块有 16 个权重。权重公式：`w = q * block_scale(6-bit)`，得出每个权重 3.4375 位。 |
| Q2_K | [GH](https://github.com/ggerganov/llama.cpp/pull/1684#issue-1739619305) | 2 位量化 (`q`)。超级块有 16 个块，每个块有 16 个权重。权重公式：`w = q * block_scale(4-bit) + block_min(4-bit)`，得出每个权重 2.625 位。 |
| IQ4_NL | [GH](https://github.com/ggerganov/llama.cpp/pull/5590) | 4 位量化 (`q`)。具有 256 个权重的超级块。使用`super_block_scale` & `importance matrix` 获得权重`w`。 |
| IQ4_XS | [HF](https://huggingface.co/CISCai/OpenCodeInterpreter-DS-6.7B-SOTA-GGUF/blob/main/README.md?code=true#L59-L70) | 4 位量化 (`q`)。具有 256 个权重的超级块。权重`w`是使用`super_block_scale`和`importance matrix`获得的，结果是每个权重4.25位。 || IQ3_S | [HF](https://huggingface.co/CISCai/OpenCodeInterpreter-DS-6.7B-SOTA-GGUF/blob/main/README.md?code=true#L59-L70) | 3 位量化 (`q`)。具有 256 个权重的超级块。权重`w`是使用`super_block_scale`和`importance matrix`获得的，结果是每个权重为3.44位。 |
| IQ3_XXS | [HF](https://huggingface.co/CISCai/OpenCodeInterpreter-DS-6.7B-SOTA-GGUF/blob/main/README.md?code=true#L59-L70) | 3 位量化 (`q`)。具有 256 个权重的超级块。权重`w`是使用`super_block_scale`和`importance matrix`获得的，每个权重为3.06位。 |
| IQ2_XXS | [HF](https://huggingface.co/CISCai/OpenCodeInterpreter-DS-6.7B-SOTA-GGUF/blob/main/README.md?code=true#L59-L70) | 2 位量化 (`q`)。具有 256 个权重的超级块。权重`w`是使用`super_block_scale`和`importance matrix`获得的，结果是每个权重为2.06位。 |
| IQ2_S | [HF](https://huggingface.co/CISCai/OpenCodeInterpreter-DS-6.7B-SOTA-GGUF/blob/main/README.md?code=true#L59-L70) | 2 位量化 (`q`)。具有 256 个权重的超级块。权重`w`是使用`super_block_scale`和`importance matrix`获得的，每个权重为2.5位。 |
| IQ2_XS | [HF](https://huggingface.co/CISCai/OpenCodeInterpreter-DS-6.7B-SOTA-GGUF/blob/main/README.md?code=true#L59-L70) | 2 位量化 (`q`)。具有 256 个权重的超级块。权重`w`是使用`super_block_scale`和`importance matrix`获得的，结果是每个权重为2.31位。 |
| IQ1_S | [HF](https://huggingface.co/CISCai/OpenCodeInterpreter-DS-6.7B-SOTA-GGUF/blob/main/README.md?code=true#L59-L70) | 1 位量化 (`q`)。具有 256 个权重的超级块。权重`w`是使用`super_block_scale`和`importance matrix`获得的，结果是每个权重为1.56位。 |
| IQ1_M | [GH](https://github.com/ggerganov/llama.cpp/pull/6302) | 1 位量化 (`q`)。具有 256 个权重的超级块。权重`w`是使用`super_block_scale`和`importance matrix`获得的，结果是每个权重1.75位。 |
| TQ1_0 | [GH](https://github.com/ggml-org/llama.cpp/pull/8151) |三元量化。 || TQ2_0 | [GH](https://github.com/ggml-org/llama.cpp/pull/8151) |三元量化。 |
| MXFP4 | [GH](https://github.com/ggml-org/llama.cpp/pull/15091) | 4 位微缩放块浮点。 |
| **旧类型** | | |
| Q8_0 | [GH](https://github.com/huggingface/huggingface.js/pull/615#discussion_r1557654249) | 8 位舍入到最接近的量化 (`q`)。每个块有 32 个权重。重量公式：`w = q * block_scale`。传统的量化方法（目前尚未广泛使用）。 |
| Q8_1 | [GH](https://github.com/huggingface/huggingface.js/pull/615#discussion_r1557682290) | 8 位舍入到最接近的量化 (`q`)。每个块有 32 个权重。重量公式：`w = q * block_scale + block_minimum`。传统的量化方法（目前尚未广泛使用）。 |
| Q5_0 | [GH](https://github.com/huggingface/huggingface.js/pull/615#discussion_r1557654249) | 5 位舍入到最接近的量化 (`q`)。每个块有 32 个权重。重量公式：`w = q * block_scale`。传统的量化方法（目前尚未广泛使用）。 |
| Q5_1 | [GH](https://github.com/huggingface/huggingface.js/pull/615#discussion_r1557682290) | 5 位舍入到最接近的量化 (`q`)。每个块有 32 个权重。重量公式：`w = q * block_scale + block_minimum`。传统的量化方法（目前尚未广泛使用）。 |
| Q4_0 | [GH](https://github.com/huggingface/huggingface.js/pull/615#discussion_r1557654249) | 4 位舍入到最接近的量化 (`q`)。每个块有 32 个权重。重量公式：`w = q * block_scale`。传统的量化方法（目前尚未广泛使用）。 |
| Q4_1 | [GH](https://github.com/huggingface/huggingface.js/pull/615#discussion_r1557682290) | 4 位舍入到最接近的量化 (`q`)。每个块有 32 个权重。重量公式：`w = q * block_scale + block_minimum`。传统的量化方法（目前尚未广泛使用）。 |*如果上表有任何不准确的地方，请在[this file](https://github.com/huggingface/huggingface.js/blob/main/packages/gguf/src/quant-descriptions.ts)上打开PR。*

### 网络钩子
https://huggingface.co/docs/hub/webhooks.md