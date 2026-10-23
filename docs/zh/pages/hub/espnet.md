<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在拥抱脸部时使用 ESPnet

`espnet`是一个用于语音处理的端到端工具包，包括自动语音识别、文本转语音、语音增强、二维化等任务。

## 在 Hub 中探索 ESPnet

通过[models page](https://huggingface.co/models?library=espnet&sort=downloads)左侧筛选，您可以找到数百个`espnet`型号。 

Hub 上的所有型号都具有有用的功能：
1. 自动生成的模型卡，其中包含描述、训练配置、许可证等。
2. 元数据标签有助于发现并包含许可证、语言和数据集等信息。
3. 一个交互式小部件，您可以使用它直接在浏览器中玩模型。
4. 推理提供程序小部件，允许发出推理请求。

## 使用现有模型

有关加载预训练模型的完整指南，我们建议查看[official guide](https://github.com/espnet/espnet_model_zoo)）。 

如果您对推理感兴趣，不同任务的不同类都有一个 `from_pretrained` 方法，允许从 Hub 加载模型。例如：
* `Speech2Text` 用于自动语音识别。
* `Text2Speech` 用于文本转语音。
* `SeparateSpeech` 用于音频源分离。

这是一个推理示例：

```py
import soundfile
from espnet2.bin.tts_inference import Text2Speech

text2speech = Text2Speech.from_pretrained("model_name")
speech = text2speech("foobar")["wav"]
soundfile.write("out.wav", speech.numpy(), text2speech.fs, "PCM_16")
```如果您想了解如何加载特定模型，可以单击`Use in ESPnet`，您将获得一个可以加载它的工作片段！ 

## 分享你的模型

`ESPnet` 输出`zip` 文件，可以轻松上传到 Hugging Face。有关共享模型的完整指南，我们建议您查看[official guide](https://github.com/espnet/espnet_model_zoo#register-your-model)）。

`run.sh` 脚本允许将给定模型上传到 Hugging Face 存储库。

```bash
./run.sh --stage 15 --skip_upload_hf false --hf_repo username/model_repo
```

## 其他资源

* ESPnet [docs](https://espnet.github.io/espnet/index.html)。
* ESPnet 模型动物园[repository](https://github.com/espnet/espnet_model_zoo)。
* 集成[docs](https://github.com/asteroid-team/asteroid/blob/master/docs/source/readmes/pretrained_models.md)。

### 集线器 API 端点
https://huggingface.co/docs/hub/api.md