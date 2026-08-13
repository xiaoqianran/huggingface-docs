<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在拥抱脸部时使用 OpenCLIP

[OpenCLIP](https://github.com/mlfoundations/open_clip) 是 OpenAI 的 CLIP 的开源实现。

## 在 Hub 上探索 OpenCLIP

您可以通过[models page](https://huggingface.co/models?library=open_clip&sort=trending)左侧的筛选来查找OpenCLIP模型。

Hub 上托管的 OpenCLIP 模型有一个模型卡，其中包含有关模型的有用信息。由于 OpenCLIP Hugging Face Hub 集成，您可以使用几行代码加载 OpenCLIP 模型。您还可以使用[Inference Endpoints](https://huggingface.co/inference-endpoints)部署这些模型。

## 安装

首先，您可以按照[OpenCLIP installation guide](https://github.com/mlfoundations/open_clip#usage)进行操作。
您还可以通过 pip 使用以下一行安装：

```
$ pip install open_clip_torch
```

## 使用现有模型

所有 OpenCLIP 模型都可以轻松地从 Hub 加载：

```py
import open_clip

model, preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
```

加载后，您可以对图像和文本进行编码以执行[zero-shot image classification](https://huggingface.co/tasks/zero-shot-image-classification)：

```py
import torch
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
image = preprocess(image).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs) 
```

它输出每个可能类别的概率：

```text
Label probs: tensor([[0.0020, 0.0034, 0.9946]])
```

如果您想加载特定的 OpenCLIP 模型，您可以单击模型卡中的`Use in OpenCLIP`，您将获得一个工作片段！

## 其他资源

* OpenCLIP [repository](https://github.com/mlfoundations/open_clip)
* OpenCLIP [docs](https://github.com/mlfoundations/open_clip/tree/main/docs)
* OpenCLIP [models in the Hub](https://huggingface.co/models?library=open_clip&sort=trending)

### 冲向空格
https://huggingface.co/docs/hub/spaces-sdks-docker-dash.md