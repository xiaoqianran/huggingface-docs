<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 使用 Hugging Face 中的 Unity Sentis 模型

[Unity 3D](https://unity.com/)是世界上最受欢迎的游戏引擎之一。 [Unity Sentis](https://unity.com/products/sentis)是在Unity 2023或更高版本上运行的推理引擎。它是一个 API，可让您利用硬件加速在游戏或应用程序中轻松集成和运行神经网络模型。由于 Unity 可以导出到许多不同的外形尺寸，包括 PC、移动设备和控制台，这意味着这是在许多不同类型的硬件上运行神经网络模型的简单方法。

## 探索 Hub 中的 Sentis 模型
通过[models page](https://huggingface.co/models?library=unity-sentis)左侧筛选，您将找到`unity-sentis`型号。

Hub 中的所有 Sentis 模型都附带代码和说明，可帮助您轻松开始在 Unity 中使用模型。 `unity`命名空间下的所有Sentis模型（例如，[unity/sentis-yolotinyv7](https://huggingface.co/unity/sentis-yolotinyv7)）都经过验证可以工作，因此您可以确定它们将在Unity中运行。

要了解有关使用 Sentis 的更多详细信息，您可以阅读其[documentation](https://docs.unity3d.com/Packages/com.unity.sentis@latest)。要获得其他使用 Sentis 的帮助，您可以在其[discussion forum](https://discussions.unity.com/c/ai-beta/sentis) 中询问

## 文件类型
每个存储库将包含多种类型的文件：* ``sentis`` 文件：这些是包含在 Unity 上运行的神经网络的主要模型文件。
* ``ONNX`` 文件：这是一种替代格式，您可以添加到 Sentis 文件中，或者代替 Sentis 文件。它对于使用第三方工具（例如 [Netron](https://github.com/lutzroeder/netron)）进行可视化非常有用。
* ``cs`` 文件：这些是 C# 文件，包含在 Unity 上运行模型的代码。
* ``info.json``：该文件包含有关存储库中文件的信息。
* 数据文件。这些是运行模型所需的其他文件。它们可以包括词汇文件、类名列表等。一些典型的文件将具有扩展名``json`` or ``txt``。
* ``README.md``。这是型号卡。它包含有关如何使用模型的说明和其他相关信息。

## 运行模型
请务必参阅型号卡上的说明。希望您具备一些 Unity 知识和一些 C# 基础知识。

1.打开Unity 2023或以上版本并创建一个新场景。

2. 从 [package manager](https://docs.unity3d.com/Manual/upm-ui-quick.html) 安装 ``com.unity.sentis`` 软件包。3. 下载模型文件（``*.sentis``）和数据文件，并将它们放入 StreamingAssets 文件夹中，该文件夹是 Assets 文件夹内的子文件夹。 （如果该文件夹不存在，您可以创建它）。

4. 将 C# 文件放置在场景中的对象上，例如主摄像机。 

5. 请参阅模型卡，查看场景中是否还有需要创建的其他对象。

在大多数情况下，我们只提供基本的实现来帮助您启动和运行。找到创造性的用途取决于您。例如，您可能想要组合两个或多个模型来完成有趣的事情。

## 分享您自己的 Sentis 模型
我们鼓励您在 Hugging Face 上分享您自己的 Sentis 模型。这些可能是您自己训练的模型，也可能是您已转换为 [Sentis format](https://docs.unity3d.com/Packages/com.unity.sentis@1.3/manual/serialize-a-model.html) 并经过测试可在 Unity 中运行的模型。 

请为您上传的每个存储库提供 Sentis 格式的模型。这提供了额外的检查，确保它们将在 Unity 中运行，也是大型模型的首选格式。您还可以包含模型文件的原始 ONNX 版本。提供具有最小实现的 C# 文件。例如，图像处理模型应该具有显示如何为输入准备图像并从输出构建图像的代码。或者，您可以链接到一些外部示例代码。这将使其他人可以轻松下载并在 Unity 中使用该模型。

提供运行模型所需的任何数据文件。例如，词汇文件。

最后，请提供``info.json`` file, which lists your project's files. This helps in counting the downloads. Some examples of the contents of ``info.json``分别是：

```
{
   "code": [ "mycode.cs"], 
   "models": [ "model1.sentis", "model2.sentis"],
   "data": [ "vocab.txt" ]
}
```

或者，如果您的代码示例是外部的：

```
{
   "sampleURL": [ "http://sampleunityproject"], 
   "models": [ "model1.sentis", "model2.sentis"]
}
```

## 附加信息
我们还有一些完整的 [sample projects](https://github.com/Unity-Technologies/sentis-samples) 来帮助您开始使用 Sentis。

### 在拥抱脸部使用 PEFT
https://huggingface.co/docs/hub/peft.md