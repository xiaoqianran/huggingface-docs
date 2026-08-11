<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 墨盒

Cartridges 是一种快速学习方法，它将压缩的长上下文表示存储为参数化的 KV 缓存
前缀。核心思想来自论文
[Cartridges: Lightweight and general-purpose long context representations via self-study](https://huggingface.co/papers/2506.06266)。

有关高级概述和动机，请参阅博客文章
[Cartridges: Storing long contexts in tiny caches with self-study](https://hazyresearch.stanford.edu/blog/2025-06-08-cartridges)。

## 唱头与前缀调音有何不同

前缀调优和 Cartridge 都是通过将 `past_key_values`（前缀 KV 缓存）注入到基础模型中来提供服务的。

- 前缀调整学习虚拟令牌嵌入（以及可选的 MLP 投影）并生成 KV 前缀。
- Cartridges 直接学习 KV 前缀本身（`p`虚拟令牌的每层键/值向量），并且是
  设计为从真实的预填充 KV 初始化（例如，语料库/系统提示的第一个 `p` 标记）。

该论文还建议冻结第一个令牌作为稳定性的注意力池（`num_frozen_tokens=1` 是
默认）。

## 用法（推论）

加载经过训练的 CARTRIDGE 适配器并运行生成：

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_path = "path/to/cartridge_adapter"

base = AutoModelForCausalLM.from_pretrained(model_id)
model = PeftModel.from_pretrained(base, adapter_path)

tok = AutoTokenizer.from_pretrained(model_id)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

out = model.generate(**tok("Question about the corpus:", return_tensors="pt"), max_new_tokens=64)
print(tok.decode(out[0], skip_special_tokens=True))
```

如果您需要在训练前创建并初始化卡盒，请参阅下面的初始化选项。

## 初始化选项

论文讨论了一些实用的初始化策略：- 随机KV（默认）：创建一个`CartridgeConfig`并开始训练。这会随机初始化 KV 前缀。
- 提示/语料库的第一个标记的 KV：使用 `initialize_kv_prefix_from_text(model, tokenizer, text=...)`。这个
  在 `text` 上运行预填充，并将第一个 `num_virtual_tokens` 的结果 KV 缓存复制到适配器中。
- 来自现有缓存的 KV：如果您已经使用 `initialize_kv_prefix_from_past_key_values(model, past_key_values=...)`
  有一个来自基本模型预填充的 `past_key_values` 对象。

## 培训

Cartridges 论文提出了一个自学蒸馏目标（一个冻结的基础模型提供教师逻辑；
CARTRIDGE 适配器经过训练，以便学生与教师在目标段上的下一个标记分布相匹配）。
PEFT 将训练逻辑保留在核心库之外；看到
`https://github.com/huggingface/peft/tree/main/examples/cartridge_self_study` 参考工作流程。
示例脚本使用冻结的基本模型作为教师，使用适应的模型作为学生，因此两者共享
相同的底层检查点。

## 成分

要将独立训练的磁带连接到单个适配器中，请使用`compose_cartridge_adapters(...)`。

# API

## CartridgeConfig[[peft.CartridgeConfig]]

#### peft.CartridgeConfig[[peft.CartridgeConfig]]

```python
peft.CartridgeConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, num_virtual_tokens: int = None, token_dim: int = None, num_transformer_submodules: Optional[int] = None, num_attention_heads: Optional[int] = None, num_layers: Optional[int] = None, modules_to_save: Optional[list[str]] = None, num_frozen_tokens: int = 1)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/cartridge/config.py#L22)

**参数：**num_frozen_tokens（`int`，默认为 1）：盒带开头保持冻结的 *前缀* 令牌数（无梯度）。 Cartridges 论文建议冻结第一个令牌作为注意力池以保持稳定性（将其设置为`1`），因为许多 LLM 使用早期令牌作为注意力池，更改它们可能会损害训练。

CARTRIDGE（KV 缓存参数化前缀适配器）的配置。

这与前缀调整的服务方式类似（如`past_key_values`），但它直接将 KV 缓存存储为
可训练的参数，而不是通过 MLP 投影来学习。

初始化：
Cartridges 论文讨论了多种初始化选项。在 PEFT 中，初始化是一个*单独的*步骤
从构建适配器配置：

- **随机KV初始化（论文选项2）**：通过`get_peft_model(...)`创建适配器。墨盒
  提示编码器参数由 PyTorch 随机初始化。

- **从提示/语料库的第一个标记派生的 KV（论文选项 3）**：在 *base 上运行无等级预填充
  model* 并将第一个 `num_virtual_tokens` 缓存的 KV 令牌复制到适配器中。 PEFT 提供实用程序
  这个（可从`peft`或`peft.tuners.cartridge.utils`导入）：

  - `initialize_kv_prefix_from_text(model, tokenizer, text=...)`
  - `initialize_kv_prefix_from_past_key_values(model, past_key_values=...)`如果您已经有一个扁平的 KV 前缀张量，您可以直接通过提示编码器加载它
  `load_prompt_embeddings(...)`方法。

## CartridgeEncoder[[peft.CartridgeEncoder]]

#### peft.CartridgeEncoder[[peft.CartridgeEncoder]]

```python
peft.CartridgeEncoder(config)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/cartridge/model.py#L20)

参数化前缀KV缓存。

参数存储在与 `PrefixEncoder` 输出相同的扁平布局中：`[num_virtual_tokens, num_layers
* 2 * token_dim]`, where `token_dim` 是每个头隐藏大小乘以头数（在任何 GQA 调整之后）
由`_prepare_prompt_learning_config`执行）。

如果`num_frozen_tokens > 0`，则第一个`num_frozen_tokens`虚拟令牌存储为不可训练的参数，
其余的代币是可训练的。

#### load_prompt_embeddings[[peft.CartridgeEncoder.load_prompt_embeddings]]

```python
load_prompt_embeddings(prompt_embeddings: torch.Tensor)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/cartridge/model.py#L89)

加载 PEFT (`prompt_embeddings`) 保存的扁平化提示嵌入。

PEFT 将即时学习适配器保存为单个 `prompt_embeddings` 张量。对于 CARTRIDGE，我们分割该张量
根据 `self.num_frozen_tokens` 分为冻结部分和可训练部分。

### 模型合并[[peft.utils.merge_utils.prune]]
https://huggingface.co/docs/peft/v0.20.0/package_reference/merge_utils.md