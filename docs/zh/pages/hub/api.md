<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 集线器 API 端点

我们有开放的端点，您可以使用它们从中心检索信息以及执行某些操作，例如创建模型、数据集或空间存储库。我们提供了一个包装器Python客户端[⟦T0⟧](https://github.com/huggingface/huggingface_hub)和一个JS客户端[⟦T1⟧](https://github.com/huggingface/huggingface.js)，可以轻松访问这些端点。我们还提供[webhooks](./webhooks)来接收有关回购的实时增量信息。享受！

> [!注意]
> 我们已将 Hub API 端点文档移至 [OpenAPI Playground](https://huggingface.co/spaces/huggingface/openapi)，它提供了始终最新的全面参考。您还可以直接在[https://huggingface.co/.well-known/openapi.json](https://huggingface.co/.well-known/openapi.json)访问OpenAPI规范，或者如果您想将其发送给您的代理，则可以使用Markdown版本：[https://huggingface.co/.well-known/openapi.md](https://huggingface.co/.well-known/openapi.md)。

> [!注意]
> 所有API调用均遵循HF范围的[Rate limits](./rate-limits)。如果您需要提升的大规模访问权限，请升级您的帐户。

### 摄取数据集
https://huggingface.co/docs/hub/datasets-ingesting.md