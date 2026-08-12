<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 关于空间的证据

**Evidence** 是一个开源框架，旨在使用 SQL 和 Markdown 构建数据驱动的应用程序、报告和仪表板。借助 Evidence，您可以快速创建决策支持工具、报告和交互式仪表板，而无需依赖传统的拖放式商业智能 (BI) 平台。 

证据使您能够：
- 使用 SQL 支持的组件直接在 Markdown 中编写报告和仪表板。
- 集成多个来源的数据，包括 SQL 数据库和 API。
- 使用模板页面根据单个模板自动生成多个页面。
- 将报告无缝部署到各种托管解决方案。

请访问[Evidence’s documentation](https://docs.evidence.dev/)，获取使用 Evidence 创建数据产品的指南、示例和最佳实践。

## 在空间上部署证据

您只需点击几下即可在 Hugging Face Spaces 上部署 Evidence：

    

创建后，空间将显示`Building`状态。如果状态没有自动更新为`Running`，请刷新页面。

您的 Evidence 应用程序将自动部署在 Hugging Face Spaces 上。 

## 从 CLI 编辑您的 Evidence 应用程序

要编辑您的应用程序，请克隆 Space 并在本地编辑文件。

```bash
git clone https://huggingface.co/spaces/your-username/your-space-name 
cd your-space-name
npm install
npm run sources
npm run dev
```然后，您可以修改pages/index.md来更改应用程序的内容。

## 从 VS Code 编辑您的 Evidence 应用程序

使用 Evidence 进行开发的最简单方法是使用 [VS Code Extension](https://marketplace.visualstudio.com/items?itemName=Evidence.evidence-vscode)：

1. 从 VS Code Marketplace 安装扩展
2. 打开命令面板（Ctrl/Cmd + Shift + P）并输入`Evidence: Copy Existing Project`
3. 粘贴您要复制的 Hugging Face Spaces Evidence 应用程序的 URL（例如 `https://huggingface.co/spaces/your-username/your-space-name`），然后按 Enter
4. 选择要将项目克隆到的文件夹，然后按 Enter
5. 按底部状态栏`Start Evidence`

查看 [alternative install methods](https://docs.evidence.dev/getting-started/install-evidence)、Github Codespaces 以及 dbt 的文档。

## 了解更多

- [Docs](https://docs.evidence.dev/)
- [Github](https://github.com/evidence-dev/evidence)
- [Slack Community](https://slack.evidence.dev/)
- [Evidence Home Page](https://www.evidence.dev)

### 评估结果
https://huggingface.co/docs/hub/eval-results.md