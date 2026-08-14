<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 关于空间的证据

**Evidence** 是一个开源框架，旨在使用 SQL 和 Markdown 构建数据驱动的应用程序、报告和仪表板。借助 Evidence，您可以快速创建决策支持工具、报告和交互式仪表板，而无需依赖传统的拖放式商业智能 (BI) 平台。 

证据使您能够：
- 使用 SQL 支持的组件直接在 Markdown 中编写报告和仪表板。
- 集成多个来源的数据，包括 SQL 数据库和 API。
- 使用模板页面根据单个模板自动生成多个页面。
- 将报告无缝部署到各种托管解决方案。

Visit [Evidence’s documentation](https://docs.evidence.dev/) for guides, examples, and best practices for using Evidence to create data products.

## Deploy Evidence on Spaces

您只需点击几下即可在 Hugging Face Spaces 上部署 Evidence：

    

创建后，空间将显示`Building`状态。如果状态没有自动更新为`Running`，请刷新页面。

您的 Evidence 应用程序将自动部署在 Hugging Face Spaces 上。 

## Editing your Evidence app from the CLI

要编辑您的应用程序，请克隆 Space 并在本地编辑文件。

```bash
git clone https://huggingface.co/spaces/your-username/your-space-name 
cd your-space-name
npm install
npm run sources
npm run dev
```You can then modify pages/index.md to change the content of your app.

## Editing your Evidence app from VS Code

使用 Evidence 进行开发的最简单方法是使用 [VS Code Extension](https://marketplace.visualstudio.com/items?itemName=Evidence.evidence-vscode)：

1. Install the extension from the VS Code Marketplace
2. 打开命令面板（Ctrl/Cmd + Shift + P）并输入`Evidence: Copy Existing Project`
3. Paste the URL of the Hugging Face Spaces Evidence app you'd like to copy (e.g. `https://huggingface.co/spaces/your-username/your-space-name`) and press Enter
4. Select the folder you'd like to clone the project to and press Enter
5. Press `Start Evidence` in the bottom status bar

Check out the docs for [alternative install methods](https://docs.evidence.dev/getting-started/install-evidence), Github Codespaces, and alongside dbt.

## 了解更多

- [Docs](https://docs.evidence.dev/)
- [Github](https://github.com/evidence-dev/evidence)
- [Slack Community](https://slack.evidence.dev/)
- [Evidence Home Page](https://www.evidence.dev)

### 评估结果
https://huggingface.co/docs/hub/eval-results.md