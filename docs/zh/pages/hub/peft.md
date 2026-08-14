<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在拥抱脸部时使用 PEFT

🤗 [Parameter-Efficient Fine-Tuning (PEFT)](https://huggingface.co/docs/peft/index) 是一个库，用于有效地将预训练的语言模型适应各种下游应用程序，而无需微调所有模型的参数。 

## 探索 Hub 上的 PEFT

您可以通过[models page](https://huggingface.co/models?library=peft&sort=trending)左侧筛选来查找PEFT型号。

## 安装

首先，您可以查看[Quick Tour in the PEFT docs](https://huggingface.co/docs/peft/quicktour)。安装时请遵循[PEFT installation guide](https://huggingface.co/docs/peft/install)。
您还可以通过 pip 使用以下一行安装：

```
$ pip install peft
```

## 使用现有模型

所有 PEFT 模型都可以从 Hub 加载。要使用 PEFT 模型，您还需要加载经过微调的基础模型，如下所示。每个微调模型的模型卡中都有基本模型。

```py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

base_model = "mistralai/Mistral-7B-v0.1"
adapter_model = "dfurman/Mistral-7B-Instruct-v0.2"

model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, adapter_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)

model = model.to("cuda")
model.eval()
```

加载后，您可以将输入传递给分词器来准备它们，并以常规 `transformers` 方式调用 `model.generate()`。

```py
inputs = tokenizer("Tell me the recipe for chocolate chip cookie", return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=10)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])
```

它输出以下内容：

```text
Tell me the recipe for chocolate chip cookie dough.

1. Preheat oven to 375 degrees F (190 degrees C).
2. In a large bowl, cream together 1/2 cup (1 stick) of butter or margarine, 1/2 cup granulated sugar, and 1/2 cup packed brown sugar.
3. Beat in 1 egg and 1 teaspoon vanilla extract.
4. Mix in 1 1/4 cups all-purpose flour.
5. Stir in 1/2 teaspoon baking soda and 1/2 teaspoon salt.
6. Fold in 3/4 cup semisweet chocolate chips.
7. Drop by
```

如果您想加载特定的PEFT模型，您可以单击模型卡中的`Use in PEFT`，您将获得一个工作片段！

## 其他资源

* PEFT [repository](https://github.com/huggingface/peft)
* PEFT [docs](https://huggingface.co/docs/peft/index)
* PEFT [models](https://huggingface.co/models?library=peft&sort=trending)

### 附录
https://huggingface.co/docs/hub/model-card-appendix.md