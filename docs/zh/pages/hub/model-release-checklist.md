<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 型号发布清单

[Hugging Face Hub](https://huggingface.co/models)是共享机器学习模型的首选平台。
执行良好的发布可以提高模型的可见性和影响力。本节涵盖了简洁、信息丰富且用户友好的模型发布的**基本**步骤。

> [!提示]
> 使用编码剂发布？安装 [⟦T11⟧ CLI Skill](./agents-cli) 让您的代理为您处理上传、模型卡和元数据。

## ⏳ 准备发布模型

### 上传模型权重

将模型上传到 Hub 时，请遵循以下最佳实践：

- **对不同的模型权重使用单独的存储库**：
   为同一架构的每个变体创建单独的存储库。这使您可以将它们分组到 [collection](https://huggingface.co/docs/hub/en/collections) 中，这比目录列表更容易导航。它还提高了可见性，因为每个模型都有自己的 URL (`hf.co/org/model-name`)，使搜索更容易，并提供每个模型的下载计数。一个很好的例子是最近的 [Qwen3-VL collection](https://huggingface.co/collections/Qwen/qwen3-vl)，它具有 VL 架构的各种变体。

- **对于权重序列化，优先使用 [⟦T13⟧](https://huggingface.co/docs/safetensors/en/index) 而不是 `pickle`。**：
   `safetensors` 比 Python 的 `pickle` 或 `pth` 更安全、更快。如果您有 `.bin` pickle 文件，请使用 [weight conversion tool](https://huggingface.co/docs/safetensors/en/convert-weights) 进行转换。###写一张综合模型卡

精心制作的模型卡（存储库中的`README.md`）对于可发现性、可重复性和有效共享至关重要。确保覆盖：

1. **元数据配置**：
   模型卡顶部的 [metadata section](https://huggingface.co/docs/hub/model-cards#model-card-metadata) (YAML) 是搜索和分类的关键。包括：
   ```yaml
   ---
   pipeline_tag: text-generation    # Specify the task
   library_name: transformers       # Specify the library
   language:
     - en                           # List languages your model supports
   license: apache-2.0              # Specify a license
   datasets:
     - username/dataset             # List datasets used for training
   base_model: username/base-model  # If applicable (your model is a fine-tune, quantized, merged version of another model)
   tags:                            # Add extra tags which would make the repo searchable using the tag
     - tag1 
     - tag2
   ---
   ```

   如果您在 Web UI 中创建 `README.md`，您将看到一个表单，其中包含我们推荐的最重要的元数据字段 🤗。

   | ![metadata template on the hub ui](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/release-checklist/metadata-template.png) |
   | :--: |
   | Hub UI 上的元数据表单 |

2. **详细型号说明**：
   清楚地解释您的模型的用途、其架构及其预期用例。帮助用户快速决定是否适合他们的需求。

3. **使用示例**：
   为推理、微调或其他常见任务提供清晰、可复制并运行的代码片段。将用户所需的编辑保持在最低限度。

   *奖励*：在存储库中添加结构良好的`notebook.ipynb`，显示推理或微调，以便用户可以直接在[Google Colab and Kaggle Notebooks](https://huggingface.co/docs/hub/en/notebooks)中打开它。

   | ![colab and kaggle button](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/release-checklist/colab-kaggle.png) |
   | :--: |
   | Google 和 Kaggle 使用按钮 |

4. **技术规格**：
   包括训练参数、硬件需求和其他帮助用户有效运行模型的细节。5. **性能指标**：
   分享基准和评估结果。包括定量指标和定性示例以显示优势和局限性。

6. **限制和偏见**：
   记录已知的限制、偏见和道德考虑因素，以便用户可以做出明智的选择。

为了使该过程更加无缝，请单击 **导入模型卡模板** 以使用占位符预先填充 `README.md`。

| ![model card template button on the hub ui](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/release-checklist/model-card-template-button.png) | ![model card template on the hub](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/release-checklist/model-card-template.png) |
|:--: | :--: |
|导入模型卡模板的按钮 |导入模板的一部分 |

### 增强模型的可发现性和可用性

为了最大限度地提高覆盖范围和可用性：

1. **库集成**：
   添加对众多 [libraries integrated with the Hugging Face Hub](https://huggingface.co/docs/hub/models-libraries) 之一（例如 `transformers`、`diffusers`、`sentence-transformers`、`timm`）的支持。这种集成显着提高了模型的可访问性，并为用户提供了使用模型的代码片段。
   
   例如，要指定您的模型适用于 `transformers` 库：
   ```yaml
   ---
   library_name: transformers
   ---
   ```

   | ![code snippet tab](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/release-checklist/code-snippet.png) |
   | :--: |
   |代码片段选项卡|

   您还可以[register your own model library](https://huggingface.co/docs/hub/en/models-adding-libraries)或将Hub支持添加到您的库和代码库中，以便用户知道如何从Hub下载模型权重。

   我们编写了有关上传最佳实践的详尽指南[here](https://huggingface.co/docs/hub/models-uploading)。> [!注意]
   > 使用注册库还可以让您跟踪模型随时间的下载情况。

2. **正确的元数据**：
   - **管道标签：** 选择正确的 [pipeline tag](https://huggingface.co/docs/hub/model-cards#specifying-a-task--pipelinetag-)，以便您的模型显示在正确的搜索和小部件中。

   常见管道标签示例：
   - `text-generation` - 用于生成文本的语言模型
   - `text-to-image` - 用于文本到图像生成模型
   - `image-text-to-text` - 用于生成文本的视觉语言模型 (VLM)
   - `text-to-speech` - 适用于从文本生成音频的模型
  
   - **许可证：**
   许可证信息对于用户了解如何使用该模型至关重要。

3. **研究论文**：
   如果您的模型有相关论文，请在模型卡中引用它们。他们将是[cross-linked automatically](https://huggingface.co/docs/hub/model-cards#linking-a-paper)。

   ```markdown
   ## References
   
   * [Model Paper](https://arxiv.org/abs/xxxx.xxxxx)
   ```

4. **收藏**：
   如果您要发布多个相关模型或变体，请将它们组织到 [collection](https://huggingface.co/docs/hub/collections) 中。集合帮助用户发现相关模型并理解跨版本的关系。

5. **演示**：
   创建带有交互式演示的 [Hugging Face Space](https://huggingface.co/docs/hub/spaces)。这使用户无需编写代码即可尝试您的模型。您还可以从空间中选择[link the model](https://huggingface.co/docs/hub/spaces-config-reference)，使其出现在模型页面 UI 上。```markdown
   ## Demo
   
   Try this model directly in your browser: [Space Demo](https://huggingface.co/spaces/username/model-demo)
   ```
   
   创建演示时，从其 Hub 存储库（而不是 Google Drive 等外部源）下载模型。这可以交叉链接工件并提高可见性

6. **量化版本**：
   考虑将量化版本（例如 GGUF）上传到单独的存储库上，以提高计算能力有限的用户的可访问性。使用量化模型卡上的[⟦T32⟧ metadata field](https://huggingface.co/docs/hub/model-cards#specifying-a-base-model)链接这些版本，并记录性能差异。

   ```yaml
   ---
   base_model: username/original-model
   base_model_relation: quantized
   ---
   ```

   | ![model tree showcasing relations](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/release-checklist/model-tree.png) |
   | :--: |
   |显示量化版本的模型树 |

7. **在模型页面上链接数据集**：
   链接元数据中的数据集，以便它们直接显示在模型页面上。

   ```yaml
   ---
   datasets:
   - username/dataset
   - username/dataset-2
   ---
   ```

8. **新模型版本**：
   如果您的模型是现有模型的更新，请在旧模型的卡片上指定它。这将在旧页面上的[display a banner](https://huggingface.co/docs/hub/en/model-cards#specifying-a-new-version)链接到更新。

   ```yaml
   ---
   new_version: username/updated-model
   ---
   ```

9. **视觉示例**：
   对于图像或视频生成模型，请使用 [⟦T33⟧ card component](https://huggingface.co/docs/hub/en/model-cards-components#the-gallery-component) 直接在模型页面上包含示例。

   ```markdown
   <Gallery>
   ![Example 1](./images/example1.png)
   ![Example 2](./images/example2.png)
   </Gallery>
   ```

10. **碳排放**：
   如果可能，请指定训练中的[carbon emissions](https://huggingface.co/docs/hub/model-cards-co2)。
   
   ```yaml
   ---
   co2_eq_emissions:
     emissions: 123.45
     source: "CodeCarbon"
     training_type: "pre-training"
     geographical_location: "US-East"
     hardware_used: "8xA100 GPUs"
   ---
   ```

### 访问控制和可见性1. **可见性设置**：
   准备好分享您的模型时，请在您的 [model settings](https://huggingface.co/docs/hub/repositories-settings) 中将其切换为公开。在此之前，请仔细检查所有文档和代码示例，以确保它们准确且完整。

2. **门控访问**：
   如果您的模型需要受控访问，请使用[gated access feature](https://huggingface.co/docs/hub/models-gated)并明确说明用户必须满足的条件。这对于具有双重用途或商业限制的模型非常重要。

## 🏁 发布模型后

成功的模型发布不仅仅限于最初的发布。为了保持质量并最大化影响：

### 维护和社区参与

1. **验证功能**：
   发布后，在干净的环境中测试所有代码片段，以确认它们按预期工作。这确保用户可以运行您的模型而不会出现错误或混乱。

   例如，如果您的模型是 `transformers` 兼容的 LLM：
   ```python
   from transformers import pipeline

   # This should run without errors
   pipe = pipeline("text-generation", model="your-username/your-model")
   result = pipe("Your test prompt")
   print(result)
   ```2. **分享分享分享**：
   大多数用户通过社交媒体、聊天渠道（例如 Slack 或 Discord）或时事通讯发现模型。在这些空间中共享您的模型链接，并将它们添加到您的网站或 GitHub 存储库。
   
   您的模特获得的访问量和点赞越多，它在[Hugging Face Trending section](https://huggingface.co/models?sort=trending)上的排名就越高，带来更高的曝光度

3. **社区互动**：
   使用“社区”选项卡可以回答问题、处理反馈并及时解决问题。澄清困惑，接受有用的建议，并关闭偏离主题的话题以保持讨论的重点。

### 添加评估结果

如果您对任何[supported benchmark datasets on the Hub](https://huggingface.co/datasets?benchmark=benchmark:official&sort=trending)进行评估，您可以将评估结果添加到您的模型存储库中。这将使基准分数直接在模型页面和数据集存储库中的基准排行榜上可见。完整规格请参阅[Evaluation Results documentation](https://huggingface.co/docs/hub/eval-results)。

要添加评估结果，请在模型存储库的 `.eval_results/` 文件夹中创建 YAML 文件。每个文件引用一个 Hub Benchmark 数据集：

```yaml
# .eval_results/gpqa.yaml
- dataset:
    id: Idavidrein/gpqa
    task_id: diamond
  value: 76.1
  date: "2026-03-19"
  source:
    url: https://huggingface.co/your-org/your-model
    name: Model Card
    user: your-username
```

`task_id` 必须与基准数据集的 `eval.yaml` 中定义的任务匹配。您可以通过检查基准数据集存储库（如[HLE](https://huggingface.co/datasets/cais/hle/blob/main/eval.yaml)）中的`eval.yaml`文件来找到可用的基准及其任务ID。社区中的任何人都可以通过打开 Pull 请求向任何模型提交评估结果。社区提交的分数在模型页面上显示“社区”徽章。为了简化此过程，您可以使用 [community-evals](https://github.com/huggingface/community-evals) 存储库，它提供脚本和代理技能，用于从模型卡中提取分数并自动创建 PR。

### 跟踪使用情况和影响

1. **使用指标**：
   [Track downloads](https://huggingface.co/docs/hub/en/models-download-stats) 并喜欢了解您的模型的影响力和采用率。您可以在模型的设置中查看总下载指标。

2. **查看社区贡献**：
   定期检查模型的存储库以获取其他用户的贡献。社区拉取请求和讨论可以提供有用的反馈、想法和协作机会。

## 🏢 企业功能

[Hugging Face Team & Enterprise](https://huggingface.co/enterprise)订阅为团队和组织提供了额外的功能：

1. **访问控制**：
   设置 [resource groups](https://huggingface.co/docs/hub/security-resource-groups) 来管理特定团队或用户的访问权限。这可确保整个组织内的正确权限和安全协作。

2. **存储区域**：
   选择模型文件的数据存储区域（美国或欧盟），以满足区域数据法规和合规性要求。3. **高级分析**：
   使用 [Publisher Analytics](https://huggingface.co/docs/hub/publisher-analytics) 更深入地了解整个组织的模型使用模式、下载和采用趋势。

4. **扩展存储**：
   随着模型组合的扩展，访问额外的私有存储容量来托管更多模型和更大的工件。

5. **组织博客文章**：
   企业组织现在可以[publish blog articles directly on Hugging Face](https://huggingface.co/blog/huggingface/blog-articles-for-orgs)。这使您可以与更广泛的社区共享模型发布、研究更新和公告，所有这些都来自您组织的个人资料。

通过遵循这些指南和示例，您将使 Hugging Face 上的模型发布清晰、有用且具有影响力。这有助于您的工作惠及更多人，加强人工智能社区，并提高模型的可见性。

我们迫不及待地想看看您接下来分享的内容！ 🤗

### 学术中心
https://huggingface.co/docs/hub/academia-hub.md