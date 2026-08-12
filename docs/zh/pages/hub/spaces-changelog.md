<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 空间变更日志

## [2026-03-18] - 受保护空间可见性

- 除了公共和私人空间之外，空间现在还支持**受保护的**可见性选项。在空间设置中，可见性现在通过具有三个选项的下拉菜单设置，而不是简单的公共/私人切换。
- [PRO](https://huggingface.co/pro) 和 [Team & Enterprise](https://huggingface.co/enterprise) 计划提供受保护的可见性。
- 受保护的空间在 Hub 上保留其源代码的私密性，而应用程序仍然可以通过其嵌入 URL 或 [custom domain](./spaces-custom-domain) 公开访问。
- 这对于托管网站而不发布源代码特别有用。
- 在[Spaces Overview](./spaces-overview#space-visibility)阅读更多内容。

## [2025-04-30] - 弃用 Streamlit SDK

- Streamlit 不再作为默认内置 SDK 选项提供。 Streamlit 应用程序现在使用 Docker 模板创建。

## [2023-07-28] - `>=1.23.0` 的上游 Streamlit 前端

- Streamlit SDK 使用 PyPI 上针对 `>=1.23.0` 发布的上游软件包，因此新发布的版本从发布之日起即可使用。

## [2023-05-30] - 添加对 Streamlit 1.23.x 和 1.24.0 的支持

- 添加了对 Streamlit `1.23.0`、`1.23.1` 和 `1.24.0` 的支持。
- 从`1.23.0`开始，Streamlit前端从HF定制版本变更为上游版本。## [2023-05-30] - 添加对 Streamlit 1.22.0 的支持

- 添加了对 Streamlit `1.22.0` 的支持。

## [2023-05-15] - 默认 Streamlit 版本
- 默认 Streamlit 版本设置为 `1.21.0`。

## [2023-04-12] - 添加对 Streamlit 最高 1.19.0 的支持
- 新增对`1.16.0`、`1.17.0`、`1.18.1`、`1.19.0`的支持，默认SDK版本为`1.19.0`。

## [2023-03-28] - 错误修复
- 修复了导致无法在嵌入 iframe 或直接访问 Streamlit 应用程序上滚动的错误，该错误已在 https://discuss.huggingface.co/t/how-to-add-scroll-bars-to-a-streamlit-app-using-space-direct-embed-url/34101 中报告。该补丁已应用于Streamlit>=1.18.1。

## [2022-12-15] - Spaces 支持 Docker 容器

- 阅读更多有关以下内容的文档：[Docker Spaces](./spaces-sdks-docker)

## [2022-12-14] - 能够设置自定义`sleep`时间

- 在这里阅读更多文档：[Spaces sleep time](./spaces-gpus#sleep-time)

## [2022-12-07] - 添加对 Streamlit 1.15 的支持

- 公告：https://twitter.com/osanseviero/status/1600881584214638592。

## [2022-06-07] - 添加对 Streamlit 1.10.0 的支持

- 新的多页面应用程序功能在 Spaces 上开箱即用。
- Streamlit 博客文章：https://blog.streamlit.io/introducing-multipage-apps。## [2022-05-23] - Spaces 加速和反应系统主题

- 所有使用 Gradio 3+ 和 Streamlit 1.x.x 的空间在加载方面都有显着的加速。
- 系统主题现在在应用程序内具有反应性。如果用户更改为深色模式，它会自动更改。

## [2022-05-21] - 默认 Debian 软件包和工厂重启

- Spaces 环境现在预装了流行的软件包（`ffmpeg`、`libsndfile1` 等）。
    - 这样，大多数时候，您不需要为您的空间指定任何额外的包即可正常工作。
    - 如果需要，`packages.txt` 文件仍然可以使用。
- 向 Spaces 添加了工厂重启按钮，允许用户完全重启，避免缓存需求并释放 GPU 内存。

## [2022-05-17] - 添加对 Streamlit 1.9.0 的支持

- 现在支持所有`1.x.0`版本（最高`1.9.0`）。

## [2022-05-16] - Gradio 3 已推出！

- 这是创建新空间时的默认版本，不要犹豫[check it out](https://huggingface.co/blog/gradio-blocks)。

## [2022-03-04] - SDK版本锁定

- `sdk_version` 字段现在会在空间创建时自动预填充。
    - 它确保您的空间在更新后保持相同的 SDK 版本。

## [2022-03-02] - Gradio 版本固定- `sdk_version` 配置字段现在可与 Gradio SDK 配合使用。

## [2022-02-21] - Python 版本

- 您可以指定您希望空间运行的 Python 版本。
- 仅支持 Python 3 版本。

## [2022-01-24] - 从 Spaces 自动链接模型和数据集

- 我们尝试自动提取代码中使用的模型和数据集存储库 ID
- 您始终可以在 YAML 中使用 `models` 和 `datasets` 手动定义它们。

## [2021-10-20] - 添加对 Streamlit 1.0 的支持

- 我们现在支持 0.79.0 到 1.0.0 之间的所有版本

## [2021-09-07] - Streamlit 版本固定

- 您现在可以选择在您的空间中安装哪个版本的 Streamlit

## [2021-09-06] - 升级 Streamlit 至 `0.84.2`

- 支持会话状态API
- [Streamlit changelog](https://github.com/streamlit/streamlit/releases/tag/0.84.0)

## [2021-08-10] - 将 Streamlit 升级到 `0.83.0`

- [Streamlit changelog](https://github.com/streamlit/streamlit/releases/tag/0.83.0)

## [2021-08-04] - Debian 软件包

- 您现在可以将 `apt-get` 依赖项添加到 `packages.txt` 文件中

## [2021-08-03] - Streamlit 组件

- 添加对[Streamlit components](https://streamlit.io/components)的支持

## [2021-08-03] - Flax/Jax GPU 改进

- 对于 GPU 激活的空间，确保 Flax / Jax 在 GPU 上顺利运行

## [2021-08-02] - 升级 Streamlit 至 `0.82.0`

- [Streamlit changelog](https://github.com/streamlit/streamlit/releases/tag/0.82.0)

## [2021-08-01] - 提供原始日志- 添加空间存储库中原始日志（构建和容器）的链接（具有空间写入权限的用户可以查看）

### Webhook 指南：为模型和数据集设置自动元数据质量审查
https://huggingface.co/docs/hub/webhooks-guide-metadata-review.md