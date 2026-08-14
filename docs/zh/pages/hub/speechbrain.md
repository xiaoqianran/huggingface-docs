<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在拥抱脸部时使用 SpeechBrain

`speechbrain` 是一款开源、一体化的音频/语音对话工具包。目标是创建一个单一、灵活且用户友好的工具包，可用于轻松开发最先进的语音技术，包括语音识别、说话人识别、语音增强、语音分离、语言识别、多麦克风信号处理等系统。

## 探索 Hub 中的 SpeechBrain

您可以通过[models page](https://huggingface.co/models?library=speechbrain)左侧筛选找到`speechbrain`型号。

Hub 上的所有型号均具有以下功能：
1. 自动生成的模型卡，附有简要说明。
2. 元数据标签有助于发现语言、许可证、论文等信息。
3. 一个交互式小部件，您可以使用它直接在浏览器中玩模型。
4. 推理提供程序小部件，允许发出推理请求。

## 使用现有模型

`speechbrain`提供了不同的接口来管理不同任务的预训练模型，例如`EncoderClassifier`、`EncoderClassifier`、`SepformerSeparation`和`SpectralMaskEnhancement`。这些类有一个 `from_hparams` 方法，您可以使用它从 Hub 加载模型以下是在城市声音中运行声音识别推理的示例。

```py
import torchaudio
from speechbrain.pretrained import EncoderClassifier

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/urbansound8k_ecapa"
)
out_prob, score, index, text_lab = classifier.classify_file('speechbrain/urbansound8k_ecapa/dog_bark.wav')
```

如果您想了解如何加载特定模型，可以单击`Use in speechbrain`，您将获得一个可以加载它的工作片段！

## 其他资源

* SpeechBrain [website](https://speechbrain.github.io/)。
* SpeechBrain [docs](https://speechbrain.readthedocs.io/en/latest/index.html)。

### 服务帐户
https://huggingface.co/docs/hub/enterprise-service-accounts.md