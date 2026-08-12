<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 空间中的 Cookie 限制

在 Hugging Face Spaces 中，应用程序在使用 cookie 时有一定的限制。这主要是由于空间页面 (`https://huggingface.co/spaces/<user>/<app>`) 的结构所致，其中包含托管在 iframe 内不同域 (`*.hf.space`) 上的应用程序。出于安全原因，现代浏览器倾向于限制使用托管在与父页面不同的域上的 iframe 页面的 cookie。

## 对使用 Docker SDK 托管 Streamlit 应用程序的影响

这些 cookie 限制可能会出现问题的一种情况是使用 Docker SDK 托管 Streamlit 应用程序时。默认情况下，Streamlit 启用基于 cookie 的 XSRF 保护。因此，某些向服务器提交数据的组件（例如`st.file_uploader()`）将无法在 Cookie 使用受到限制的 HF 空间上正常工作。

要解决此问题，您需要将 Streamlit 中的 `server.enableXsrfProtection` 选项设置为 `false`。有两种方法可以做到这一点：

1. 命令行参数：运行 Streamlit 应用程序时，可以将选项指定为命令行参数。这是示例命令：
   ```shell
   streamlit run app.py --server.enableXsrfProtection false
   ```2. 配置文件：或者，您可以在 Streamlit 配置文件`.streamlit/config.toml` 中指定选项。你可以这样写：
   ```toml
   [server]
   enableXsrfProtection = false
   ```

> [!提示]
> 当您使用 Streamlit SDK 时，您无需担心这一点，因为 SDK 会为您做这件事。

### 管理组织
https://huggingface.co/docs/hub/organizations-managing.md