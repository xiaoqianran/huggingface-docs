<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 空间设置

您可以在存储库根目录的 **README.md** 文件顶部的 `YAML` 块内配置空间的外观和其他设置。例如，如果您想创建一个名为 `Demo Space` 且带有黄色到橙色渐变缩略图的渐变空间：

```yaml
---
title: Demo Space
emoji: 🤗
colorFrom: yellow
colorTo: orange
sdk: gradio
app_file: app.py
pinned: false
---
```

有关其他设置，请参阅[Reference](./spaces-config-reference)部分。

### 空间作为代理工具
https://huggingface.co/docs/hub/spaces-agents.md