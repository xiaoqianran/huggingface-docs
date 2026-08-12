<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 图书馆

该中心支持开源生态系统中的数十个库。借助 `huggingface_hub` Python 库，可以轻松在 Hub 上共享模型。该中心支持许多库，我们正在努力扩展这种支持。我们很高兴欢迎来到该中心的一组开源库，这些库正在推动机器学习的发展。

下表总结了支持的库及其集成级别。在 [the model-libraries.ts file](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/model-libraries.ts) 中查找我们所有支持的库。|图书馆 |描述 |推理提供商 |小部件 |从集线器下载 |推送到集线器 |
|----------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|---|---:|---|---|
| [Adapters](./adapters) |统一的 Transformers 附加组件，用于参数高效和模块化微调。                                                | ✅ | ✅ | ✅ | ✅ |
| [AllenNLP](./allennlp) |一个基于 PyTorch 构建的开源 NLP 研究库。                               | ✅ | ✅ | ✅ | ❌ |
| [Asteroid](./asteroid) |基于PyTorch的音频源分离工具包 | ✅ | ✅ | ✅ | ❌ |
| [BERTopic](./bertopic) | BERTopic 是一个文本和图像主题建模库 | ✅ | ✅ | ✅ | ✅ | 
| [Diffusers](./diffusers) |用于扩散模型推理和训练的模块化工具箱 | ✅ | ✅ | ✅ | ✅ || [docTR](https://github.com/mindee/doctr) | PyTorch 和 TensorFlow 中 OCR 相关任务的模型和数据集 | ✅ | ✅ | ✅ | ❌ |
| [ESPnet](./espnet) |端到端语音处理工具包（例如 TTS）| ✅ | ✅ | ✅ | ❌ |
| [fastai](./fastai) |用于训练具有最先进输出的快速、准确模型的库。             | ✅ | ✅ | ✅ | ✅ |
| [Keras](./keras) |开源多后端深度学习框架，支持 JAX、TensorFlow 和 PyTorch。 | ❌ | ❌ | ✅ | ✅ |
| [KerasNLP](https://keras.io/guides/keras_nlp/upload/) |自然语言处理库构建在 Keras 之上，可与 TensorFlow、JAX 或 PyTorch 原生配合使用。 | ❌ | ❌ | ✅ | ✅ |
| [TF-Keras](./tf-keras)（旧版）|旧库使用一致且简单的 API 来构建利用 TensorFlow 及其生态系统的模型。 | ❌ | ❌ | ✅ | ✅ |
| [Flair](./flair) |最先进的 NLP 的非常简单的框架。                                      | ✅ | ✅ | ✅ | ✅ || [MBRL-Lib](https://github.com/facebookresearch/mbrl-lib) | MBRL 算法的 PyTorch 实现。                                          | ❌ | ❌ | ✅ | ✅ |
| [MidiTok](https://github.com/Natooz/MidiTok) |符号音乐/MIDI 文件的标记器。                                          | ❌ | ❌ | ✅ | ✅ |
| [ML-Agents](./ml-agents) |使使用 Unity 制作的游戏和模拟能够作为训练智能代理的环境。 | ❌ | ❌ | ✅ | ✅ |
| [MLX](./mlx) |苹果公司在苹果芯片上进行模型训练和服务框架。 | ❌ | ❌ | ✅ | ✅ |
| [NeMo](https://github.com/NVIDIA/NeMo) |为研究人员构建的对话式人工智能工具包 | ✅ | ✅ | ✅ | ❌ |
| [OpenCLIP](./open_clip) |用于 OpenAI 的 CLIP 开源实现的库 | ❌ | ❌ | ✅ | ✅ |  
| [PaddleNLP](./paddlenlp) |基于 PaddlePaddle 构建的易于使用且功能强大的 NLP 库 | ✅ | ✅ | ✅ | ✅ |
| [PEFT](./peft) |尖端参数高效微调库 | ✅ | ✅ | ✅ | ✅ || [Pyannote](https://github.com/pyannote/pyannote-audio) |用于说话人二值化的神经构建模块。                                      | ❌ | ❌ | ✅ | ❌ |
| [PyCTCDecode](https://github.com/kensho-technologies/pyctcdecode) |语言模型支持语音识别的 CTC 解码 | ❌ | ❌ | ✅ | ❌ |
| [Pythae](https://github.com/clementchadebec/benchmark_VAE) | Python 生成自动编码器的统一框架 | ❌ | ❌ | ✅ | ✅ |
| [RL-Baselines3-Zoo](./rl-baselines3-zoo) |强化学习的训练框架，使用[Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)。| ❌ | ✅ | ✅ | ✅ |
| [Sample Factory](./sample-factory) |高吞吐量异步强化学习的代码库。                    | ❌ | ✅ | ✅ | ✅ |
| [Sentence Transformers](./sentence-transformers) |计算句子、段落和图像的密集向量表示。          | ✅ | ✅ | ✅ | ✅ |
| [SetFit](./setfit) |使用 Sentence Transformers 进行高效的少量文本分类 | ✅ | ✅ | ✅ | ✅ |
| [spaCy](./spacy) | Python 和 Cython 中的高级自然语言处理。                           | ✅ | ✅ | ✅ | ✅ |
| [SpanMarker](./span_marker) |熟悉、简单且最先进的命名实体识别。                      | ✅ | ✅ | ✅ | ✅ || [Scikit Learn (using skops)](https://skops.readthedocs.io/en/stable/) | Python 中的机器学习。                                                          | ✅ | ✅ | ✅ | ✅ |
| [Speechbrain](./speechbrain) | PyTorch 支持的语音工具包。                                                    | ✅ | ✅ | ✅ | ❌ |
| [Stable-Baselines3](./stable-baselines3) | PyTorch 中深度强化学习算法的一组可靠实现 | ❌ | ✅ | ✅ | ✅ |
| [TensorFlowTTS](https://github.com/TensorSpeech/TensorFlowTTS) |最先进的实时语音合成架构。                           | ❌ | ❌ | ✅ | ❌ |
| [Timm](./timm) |图像模型、脚本、预训练权重等集合 | ✅ | ✅ | ✅ | ✅ |
| [Transformers](./transformers) |适用于 PyTorch、TensorFlow 和 JAX 的最先进自然语言处理 | ✅ | ✅ | ✅ | ✅ |
| [Transformers.js](./transformers-js) |最先进的网络机器学习。直接在浏览器中运行 🤗 Transformers，无需服务器！ | ❌ | ❌ | ✅ | ❌ |
| [Unity Sentis](./unity-sentis) | Unity 3D 游戏引擎的推理引擎 | ❌ | ❌ | ❌ | ❌ |

### 如何添加对新库的支持？

如果您有兴趣添加您的图书馆，请联系我们！请阅读[Adding a Library Guide](./models-adding-libraries)了解相关内容。### Docker 空间
https://huggingface.co/docs/hub/spaces-sdks-docker.md