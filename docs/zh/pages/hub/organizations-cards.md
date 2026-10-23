<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 组织卡

您可以创建组织卡来帮助用户详细了解您的组织正在做什么以及用户如何使用您的库、模型、数据集和空间。

组织卡显示在组织的个人资料上：

如果您是组织的成员，您将在组织的主页上看到用于创建或编辑组织卡的按钮。组织卡是名为 `README` 的 Space 存储库中的 `README.md` 静态文件。该卡片可以像 Markdown 文本一样简单，也可以使用 HTML 创建更加自定义的外观。

空间必须设置为 **公共** 可见性。如果将其设置为“受保护”或“私有”，组织卡将不会显示在组织的个人资料页面上。

[Hugging Face Course organization](https://huggingface.co/huggingface-course)的卡片，如上所示，[contains the following HTML](https://huggingface.co/spaces/huggingface-course/README/blob/main/README.md)：

```html
<p>
  This is the organization grouping all the models and datasets used in the
  <a href="https://huggingface.co/course/chapter1" class="underline">Hugging Face course</a>.
</p>
```

有关更多示例，请查看：

- [Amazon's](https://huggingface.co/spaces/amazon/README/blob/main/README.md)组织卡源代码
- [spaCy's](https://huggingface.co/spaces/spacy/README/blob/main/README.md)组织卡源代码。

### Webhook 指南：使用 LLM 回复构建讨论机器人
https://huggingface.co/docs/hub/webhooks-guide-discussion-bot.md