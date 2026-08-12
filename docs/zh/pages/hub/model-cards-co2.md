<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 显示您模型的碳排放量

## 为什么计算我的模型的碳排放量是有益的？

训练 ML 模型通常是能源密集型的，并且会产生大量碳足迹，如[Strubell et al.](https://arxiv.org/abs/1906.02243)所述。因此，“跟踪”和“报告”模型的排放量以更好地了解我们领域的环境影响非常重要。

## 关于我的模型的碳足迹，我应该包含哪些信息？

如果可以的话，您应该包含以下信息：
- 模型的训练地点（就位置而言）
- 使用的硬件——例如GPU、TPU 或 CPU，以及多少个
- 训练类型：预训练或微调
- 模型的估计碳足迹，使用[Code Carbon](https://github.com/mlco2/codecarbon)包实时计算或使用[ML CO2 Calculator](https://mlco2.github.io/impact/)训练后计算。

## 碳足迹元数据

您可以将碳足迹数据添加到模型卡元数据（在 README.md 文件中）。元数据的结构应该是：

```yaml
---
co2_eq_emissions:
  emissions: number (in grams of CO2)
  source: "source of the information, either directly from AutoTrain, code carbon or from a scientific article documenting the model"
  training_type: "pre-training or fine-tuning"
  geographical_location: "as granular as possible, for instance Quebec, Canada or Brooklyn, NY, USA. To check your compute's electricity grid, you can check out https://app.electricitymap.org."
  hardware_used: "how much compute and what kind, e.g. 8 v100 GPUs"
---
```

## 我的模型的碳足迹是如何计算的？ 🌎

考虑到计算硬件、位置、使用情况和训练时间，您可以估算模型产生的二氧化碳量。

数学很简单！ ➕首先，您获取用于培训的电网的“碳强度”——这是每千瓦时电力产生的二氧化碳量。碳强度取决于硬件的位置以及该位置使用的[energy mix](https://electricitymap.org/)——无论是太阳能🌞、风能🌬️和水力💧等可再生能源，还是煤炭⚫和天然气💨等不可再生能源。用于训练的可再生能源越多，碳强度就越低！
 
然后，您可以使用 `pynvml` 库获取训练期间 GPU 的功耗。

最后，将功耗和碳强度乘以模型的训练时间，即可估算出二氧化碳排放量。

请记住，这不是一个确切的数字，因为其他因素也会发挥作用，例如用于数据中心加热和冷却的能源，这会增加碳排放。但这会让您很好地了解您的模型产生的二氧化碳排放规模！

要将 **碳排放** 元数据添加到您的模型中：1. 如果您使用 **AutoTrain**，我们会为您跟踪 🔥
2. 否则，在训练代码中使用 Code Carbon 等跟踪器，然后指定
```yaml
co2_eq_emissions: 
  emissions: 1.2345
```
在您的模型卡元数据中，其中 `1.2345` 是以 **克** 为单位的排放值。 

要了解有关变形金刚碳足迹的更多信息，请查看[video](https://www.youtube.com/watch?v=ftWlj4FBHTg)，拥抱脸部课程的一部分！

### 值得信赖的出版商
https://huggingface.co/docs/hub/trusted-publishers.md