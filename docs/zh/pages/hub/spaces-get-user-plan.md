<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 如何获取用户在 Spaces 中的计划和状态

从空间的 iframe 内部，您可以检查用户是否登录主站点，以及他们是否有 PRO 订阅或其组织之一是否有付费订阅。

```js
window.addEventListener("message", (event) => {
    if (event.data.type === "USER_PLAN") {
        console.log("plan", event.data.plan);
    }
})

window.parent.postMessage({
    type: "USER_PLAN_REQUEST"
}, "https://huggingface.co");
```

`event.data.plan` 的类型为：

```ts
{
    user: "anonymous",
    org: undefined
} | {
    user: "pro" | "free",
    org: undefined | "team" | "enterprise" | "plus" | "academia"
}
```

您将获得用户的状态（已注销 = `"anonymous"`）及其计划。

## 示例

- https://huggingface.co/spaces/huggingfacejs/plan

### Hugging Face Hub 上的 Jupyter 笔记本
https://huggingface.co/docs/hub/notebooks.md