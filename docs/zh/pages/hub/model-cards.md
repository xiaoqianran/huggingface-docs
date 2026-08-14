<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 模型卡

## 什么是模型卡？

模型卡是模型随附的文件，可提供方便的信息。在底层，模型卡是带有附加元数据的简单 Markdown 文件。模型卡对于可发现性、可重复性和共享至关重要！您可以在任何模型存储库中找到模型卡作为 `README.md` 文件。

型号卡应描述：
- 模型
- 其预期用途和潜在限制，包括偏见和道德考虑，详见[Mitchell, 2018](https://arxiv.org/abs/1810.03993)
- 训练参数和实验信息（您可以嵌入或链接到实验跟踪平台以供参考）
- 使用哪些数据集来训练您的模型
- 模型的评估结果

型号卡模板可用[here](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md)。

型号卡各部分的填写方法详见[the Annotated Model Card](https://huggingface.co/docs/hub/model-card-annotated)。

集线器上的模型卡有两个关键部分，并且信息重叠：
- [Metadata](#model-card-metadata)
- [Text descriptions](#model-card-text)

## 模型卡元数据

模型存储库会将其 `README.md` 渲染为模型卡。模型卡是一个 [Markdown](https://en.wikipedia.org/wiki/Markdown) 文件，顶部有一个 [YAML](https://en.wikipedia.org/wiki/YAML) 部分，其中包含有关模型的元数据。 

添加到模型卡的元数据支持发现和更轻松地使用模型。例如：* 允许用户在 https://huggingface.co/models 上过滤模型。
* 显示模型的许可证。
* 将数据集添加到元数据将向您的模型页面添加一条消息“`Datasets used to train:`”，并链接相关数据集（如果 Hub 上提供）。

数据集和语言标识符在[Datasets](https://huggingface.co/datasets)和[Languages](https://huggingface.co/languages)页面上列出。

### 将元数据添加到模型卡

有几种不同的方法可以将元数据添加到模型卡，包括：
- 使用元数据用户界面
- 直接编辑`README.md`文件的YAML部分
- 通过 [⟦T16⟧](https://huggingface.co/docs/huggingface_hub) Python 库，请参阅 [docs](https://huggingface.co/docs/huggingface_hub/guides/model-cards#update-metadata) 了解更多详细信息。

当您上传模型时，许多具有 [Hub integration](./models-libraries) 的库会自动将元数据添加到模型卡中。 

#### 使用元数据 UI

您可以使用元数据 UI 将元数据添加到模型卡。要访问元数据 UI，请转到模型页面并单击模型卡右上角的 `Edit model card` 按钮。这将打开一个显示模型卡 `README.md` 文件的编辑器，以及用于编辑元数据的 UI。此 UI 将允许您将关键元数据添加到模型卡中，并且许多字段将根据您提供的信息自动完成。使用 UI 是将元数据添加到模型卡的最简单方法，但它不支持所有元数据字段。如果要添加 UI 不支持的元数据，可以直接编辑 `README.md` 文件的 YAML 部分。

#### 编辑 `README.md` 文件的 YAML 部分

您还可以直接编辑 `README.md` 文件的 YAML 部分。如果模型卡还没有 YAML 部分，您可以通过在文件顶部添加三个 `---` 来添加一个 YAML 部分，然后包含所有相关元数据，并使用另一组 `---` 关闭该部分，如下例所示：

```yaml
---
language: 
  - "List of ISO 639-1 code for your language"
  - lang1
  - lang2
thumbnail: "url to a thumbnail used in social sharing"
tags:
- tag1
- tag2
license: "any valid license identifier"
datasets:
- dataset1
- dataset2
base_model: "base model Hub identifier"
---
```

您可以在此处找到详细的型号卡元数据规范。

### 指定一个库

您可以在模型卡元数据部分指定支持的库。了解有关我们支持的库[here](./models-libraries)的更多信息。库将按以下优先级顺序指定：

1. 在型号卡中指定`library_name`（如果您的型号不是`transformers`型号，建议使用）。此信息可以通过元数据 UI 添加或直接在模型卡 YAML 部分中添加：```yaml
library_name: flair
```

2. 拥有一个带有支持的库名称的标签

```yaml
tags:
- flair
```

如果未指定，Hub 将尝试自动检测库类型。但是，不鼓励这种方法，并且存储库创建者应尽可能使用显式的 `library_name`。 

1. 通过查看`*.nemo`或`*.mlmodel`等文件的存在，Hub可以确定模型是否来自NeMo或CoreML。
2. 以前，如果没有检测到任何内容并且存在`config.json`文件，则认为该库是`transformers`。对于 2024 年 8 月之后创建的模型存储库，情况不再如此，因此您需要显式设置 `library_name: transformers`。

### 指定基本模型

如果您的模型是基本模型的微调、适配器或量化版本，您可以在模型卡元数据部分指定基本模型。此信息还可用于指示您的模型是否是多个现有模型的合并。因此，`base_model`字段可以是单个模型ID，也可以是一个或多个base_models的列表（由它们的Hub标识符指定）。 

```yaml
base_model: HuggingFaceH4/zephyr-7b-beta
```此元数据将用于在模型页面上显示基本模型。用户还可以使用此信息按基本模型过滤模型或查找从特定基本模型派生的模型：

   
      对于微调模型：
      
         
         
      
   
   
      对于适配器（LoRA、PEFT 等）：
      
         
         
      
   

   
      对于另一个模型的量化版本：
      
        
        
      
   
   
      对于两个或多个模型的合并：
      
        
        
      
   

在合并情况下，您指定两个或多个基本模型的列表：

```yaml
base_model:
- Endevor/InfinityRP-v1-7B
- l3utterfly/mistral-7b-v0.1-layla-v4
```

Hub 将推断从当前模型到基本模型 (`"adapter", "merge", "quantized", "finetune"`) 的关系类型，但您也可以根据需要显式设置它：例如 `base_model_relation: quantized`。

### 指定新版本

如果中心中有新版本的模型可用，您可以在 `new_version` 字段中指定它。  

例如，在 `l3utterfly/mistral-7b-v0.1-layla-v3` 上：

```yaml
new_version: l3utterfly/mistral-7b-v0.1-layla-v4
```

此元数据将用于在模型页面上显示模型最新版本的链接。如果`new_version`中链接的模型也有`new_version`字段，则将始终显示最新版本。 

   
   

### 指定数据集您可以在模型卡元数据部分指定用于训练模型的数据集。数据集将显示在模型页面上，用户将能够按数据集过滤模型。您应该使用 Hub 数据集标识符，它与数据集的存储库名称相同作为标识符：

```yaml
datasets:
- stanfordnlp/imdb
- HuggingFaceFW/fineweb
```

### 指定一个桶

您可以在模型卡元数据部分指定链接到您的模型的 [storage buckets](./storage-buckets)。这些存储桶将在模型页面上显示为标签，并且链接的存储桶页面将反过来显示该模型。您应该使用 Hub 存储桶标识符，它与存储桶的存储库名称相同：

```yaml
buckets:
- my-org/my-bucket
- my-org/another-bucket
```

### 指定任务 (`pipeline_tag`)

您可以在模型卡元数据中指定 `pipeline_tag`。 `pipeline_tag` 表示模型适用的任务类型。该标签将显示在模型页面上，用户可以按任务过滤 Hub 上的模型。此标签还用于确定模型使用哪个 [widget](./models-widgets#enabling-a-widget) 以及在后台使用哪些 API。对于 `transformers` 模型，管道标记是从模型的 `config.json` 文件中自动推断的，但如果需要，您可以在模型卡元数据中覆盖它。在元数据 UI 中编辑此字段将确保管道标记有效。其他一些具有 Hub 集成的库也会自动将管道标签添加到模型卡元数据中。

### 指定许可证

您可以在模型卡元数据部分指定许可证。许可证将显示在模型页面上，用户将能够按许可证过滤模型。使用元数据 UI，您将看到最常见许可证的下拉列表。

如果需要，您还可以通过添加 `other` 作为许可证值并在元数据中指定名称和许可证链接来指定自定义许可证。 

```yaml
# Example from https://huggingface.co/coqui/XTTS-v1
---
license: other
license_name: coqui-public-model-license
license_link: https://coqui.ai/cpml
---
```

如果许可证无法通过 URL 获得，您可以链接到模型存储库中存储的许可证。

### 评估结果

您可以在模型卡元数据中以结构化方式指定**模型的评估结果**。结果由中心解析并显示在模型页面上的小部件中。以下是 [bigcode/starcoder](https://huggingface.co/bigcode/starcoder) 型号的示例：最初的元数据规范基于代码为 [model-index specification](https://github.com/paperswithcode/model-index) 的论文。这使我们能够在适当的时候直接将结果索引到带有代码排行榜的论文中。您还可以链接计算评估结果的源。

> [!提示]
> 新功能：我们为评估结果提供了一种新的、更简单的元数据格式。请在[the dedicated doc page](./eval-results)查看。

以下是模型索引的部分示例，用于描述 [01-ai/Yi-34B](https://huggingface.co/01-ai/Yi-34B) 在 ARC 基准测试中的得分。结果来自 [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)，其定义为 `source`：

```yaml
---
model-index:
  - name: Yi-34B
    results:
      - task:
          type: text-generation
        dataset:
          name: ai2_arc
          type: ai2_arc
        metrics:
          - name: AI2 Reasoning Challenge (25-Shot)
            type: AI2 Reasoning Challenge (25-Shot)
            value: 64.59
        source:
          name: Open LLM Leaderboard
          url: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
---
```

有关如何格式化此数据的更多详细信息，请查看[Model Card specifications](https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1)。

### 二氧化碳排放量

模型卡也是显示有关模型的二氧化碳影响信息的好地方。请访问我们的[guide on tracking and reporting CO2 emissions](./model-cards-co2)了解更多信息。

### 链接论文

如果模型卡包含指向 Paper 页面的链接（HF 或 Arxiv 摘要/PDF），Hugging Face Hub 将提取 arXiv ID 并将其包含在格式为 `arxiv:<PAPER ID>` 的模型标签中。单击该标签将使您：

* 访问论文页面
* 过滤 Hub 上引用同一论文的其他模型。

阅读有关纸质页面[here](./paper-pages)的更多信息。

## 模型卡文本有关如何填写模型卡的人类可读部分（以便可以打印、剪切+粘贴等）的详细信息，请参阅[Annotated Model Card](./model-card-annotated)。

## 常见问题解答

### 模型标签是如何确定的？

每个模型页面在页面标题中模型名称下方列出了所有模型的标签。这些主要是根据模型卡元数据计算的，尽管有些是自动添加的，如[Enabling a Widget](./models-widgets#enabling-a-widget)中所述。

### 我可以向我的模型添加自定义标签吗？

是的，您可以将自定义标签添加到模型中，方法是将它们添加到模型卡元数据中的 `tags` 字段。元数据 UI 会建议一些流行的标签，但您可以添加任何您想要的标签。例如，您可以通过添加 `finance` 标签来表明您的模型专注于金融。

### 如何表明我的模型并不适合所有受众

您可以将 `not-for-all-audiences` 标签添加到模型卡元数据中。当此标签存在时，模型页面上将显示一条消息，指示该模型并不适合所有受众。用户可以点击此消息来查看型号卡。 

### 如何在深色和浅色模式下显示不同的图像？您可以显示针对每个主题优化的图像的不同版本。这对于需要不同配色方案来保持明暗模式下的可见性和美观性的徽标、图表或屏幕截图特别有用。要使用此功能，您需要提供图像的两个版本。

**对于通过 Markdown 编辑器上传的图像**

当您直接从 Markdown 编辑器上传图像（使用拖放）时，请将 URI 片段 `#hf-light-mode-only` 或 `#hf-dark-mode-only` 附加到图像 URL 的末尾，以指定它应在哪个主题中显示：

```markdown
Image only displays when viewing in light mode
![Logo](https://cdn-uploads.huggingface.co/production/uploads/logo-light.png#hf-light-mode-only)

Image only displays when viewing in dark mode
![Logo](https://cdn-uploads.huggingface.co/production/uploads/logo-dark.png#hf-dark-mode-only)
```

**对于已经托管的图像**

如果您想引用已托管的图像而不重新上传它们，请使用 HTML `<img>` 标签和以下 Tailwind CSS 类来指定它应显示在哪个主题中：

```html
// Image only displays when viewing in dark mode
<img class="hidden dark:block" src="https://hf.co/logo-dark.png" alt="Logo" />

// Image only displays when viewing in light mode
<img class="dark:hidden" src="https://hf.co/logo-light.png" alt="Logo" />
```

### 我可以在模型卡中写入 LaTeX 吗？

是的！ Hub 在解析 Markdown 之前使用 [KaTeX](https://katex.org/) 数学排版库在服务器端渲染数学公式。

您必须使用以下分隔符：
- `$$ ... $$` 用于显示模式
- `&#92;&#92;(...\\)` 用于内联模式（斜杠和括号之间没有空格）。

然后你就可以写：

$$
\乳胶
$$$$
\mathrm{MSE} = \left(\frac{1}{n}\right)\sum_{i=1}^{n}(y_{i} - x_{i})^{2}
$$

$$ E=mc^2 $$

### 组织卡
https://huggingface.co/docs/hub/organizations-cards.md