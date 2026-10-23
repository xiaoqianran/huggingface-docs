<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 如何处理Spaces中的URL参数

您可以使用 URL 查询参数作为数据共享机制，例如能够深度链接到具有特定状态的应用程序。

在空间页面（`https://huggingface.co/spaces/<user>/<app>`）上，实际应用程序页面（`https://*.hf.space/`）嵌入在 iframe 中。附加到父页面 URL 的查询字符串和哈希会在初始加载时传播到嵌入式应用程序，因此嵌入式应用程序无需特殊考虑即可读取这些值。

相比之下，从嵌入式应用程序更新查询字符串和父页面 URL 的哈希稍微复杂一些。
如果您想在 Docker 或静态空间中执行此操作，则需要添加以下 JS 代码，该代码将消息发送到具有 `queryString` 和/或 `hash` 键的父页面。

```js
const queryString = "...";
const hash = "...";

window.parent.postMessage({
    queryString,
    hash,
}, "https://huggingface.co");
```

**这仅适用于 Docker 或静态空间。**

对于 Streamlit 应用程序，Spaces 会自动同步 URL 参数。 Gradio 应用程序可以从 Spaces 页面读取查询参数，但不会将更新的 URL 参数与父页面同步。

请注意，父页面的 URL 参数*仅*在初始加载时传播到嵌入式应用程序。因此，即使使用此方法更新父 URL 哈希，嵌入式应用程序中的 `location.hash` 也不会更改。这种方法的一个例子可以在这个静态空间中找到，
[⟦T6⟧](https://huggingface.co/spaces/whitphx/static-url-param-sync-example)。

### 显示您模型的碳排放量
https://huggingface.co/docs/hub/model-cards-co2.md