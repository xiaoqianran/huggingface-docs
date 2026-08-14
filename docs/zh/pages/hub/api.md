<!-- huggingface-docs: machine-translated zh-CN from English source -->

# Hub API Endpoints

我们有开放的端点，您可以使用它们从中心检索信息以及执行某些操作，例如创建模型、数据集或空间存储库。 We offer a wrapper Python client, [⟦T0⟧](https://github.com/huggingface/huggingface_hub), and a JS client, [⟦T1⟧](https://github.com/huggingface/huggingface.js), that allow easy access to these endpoints. We also provide [webhooks](./webhooks) to receive real-time incremental info about repos.享受！

> [!注意]
> We've moved the Hub API Endpoints documentation to our [OpenAPI Playground](https://huggingface.co/spaces/huggingface/openapi), which provides a comprehensive reference that's always up-to-date. You can also access the OpenAPI specification directly at [https://huggingface.co/.well-known/openapi.json](https://huggingface.co/.well-known/openapi.json), or in Markdown version if you want to send it to your Agent: [https://huggingface.co/.well-known/openapi.md](https://huggingface.co/.well-known/openapi.md).

> [!注意]
> All API calls are subject to the HF-wide [Rate limits](./rate-limits). Upgrade your account if you need elevated, large-scale access.

### Ingesting Datasets
https://huggingface.co/docs/hub/datasets-ingesting.md