<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 技能

> [!提示]
> 正在寻找 `hf` CLI 技能？这是将您的代理连接到 Hugging Face Hub 和生态系统的最快方式。请参阅[Hugging Face CLI for AI Agents](./agents-cli)指南。

Hugging Face 提供了一套专为 AI 构建者打造的精选技能。训练模型、创建数据集、运行评估、跟踪实验。每个技能都是一个独立的`SKILL.md`，您的代理在执行任务时遵循它。

技能适用于所有主要编码代理：Claude Code、OpenAI Codex、Google Gemini CLI 和 Cursor。了解有关格式的更多信息，请访问[agentskills.io](https://agentskills.io)。

## 安装

```bash
# register the skills marketplace
/plugin marketplace add huggingface/skills

# install a specific Skill
/plugin install <skill-name>@huggingface/skills
```

将技能从[repository](https://github.com/huggingface/skills)复制或符号链接到法典标准`.agents/skills`位置之一（例如`$REPO_ROOT/.agents/skills`或`$HOME/.agents/skills`）。 Codex 通过代理技能标准自动发现它们。

或者，使用捆绑的 [⟦T7⟧](https://github.com/huggingface/skills/blob/main/agents/AGENTS.md) 作为后备。

```bash
gemini extensions install https://github.com/huggingface/skills.git --consent
```

使用 [repository URL](https://github.com/huggingface/skills) 通过 Cursor 插件流程安装。该存储库包括 `.cursor-plugin/plugin.json` 和 `.mcp.json` 清单。

## 可用技能|技能|它有什么作用 |
| -----| ------------ |
| [⟦T10⟧](https://github.com/huggingface/skills/tree/main/skills/hf-cli) |通过`hf` CLI 进行集线器操作：下载、上传、管理存储库、运行作业 |
| [⟦T12⟧](https://github.com/huggingface/skills/tree/main/skills/huggingface-datasets) |探索数据集、对行进行分页、搜索文本、应用过滤器 |
| [⟦T13⟧](https://github.com/huggingface/skills/tree/main/skills/huggingface-llm-trainer) |在 HF 职位上使用 TRL（SFT、DPO、GRPO）培训或微调法学硕士 |
| [⟦T14⟧](https://github.com/huggingface/skills/tree/main/skills/huggingface-vision-trainer) |训练目标检测和图像分类模型 |
| [⟦T15⟧](https://github.com/huggingface/skills/tree/main/skills/huggingface-community-evals) |在本地硬件上对 Hugging Face Hub 上的模型运行评估 |
| [⟦T16⟧](https://github.com/huggingface/skills/tree/main/skills/huggingface-trackio) |使用 Trackio 跟踪和可视化 ML 训练实验 |
| [⟦T17⟧](https://github.com/huggingface/skills/tree/main/skills/huggingface-papers) |在 Markdown 中查找并阅读 Hugging Face 纸质页面 |
| [⟦T18⟧](https://github.com/huggingface/skills/tree/main/skills/huggingface-paper-publisher) |在该中心发布和管理研究论文 |
| [⟦T19⟧](https://github.com/huggingface/skills/tree/main/skills/huggingface-tool-builder) |为 HF API 操作构建可重用脚本 |
| [⟦T20⟧](https://github.com/huggingface/skills/tree/main/skills/huggingface-gradio) |构建 Gradio Web UI 和演示 |
| [⟦T21⟧](https://github.com/huggingface/skills/tree/main/skills/transformers-js) |使用 WebGPU/WASM 在 JavaScript/TypeScript 中运行 ML 模型 |

## 使用技巧

安装后，直接在提示中提及技能：

- “使用 HF 模型训练器 Skill 在 Capybara 数据集上通过 SFT 微调 Qwen3-0.6B”
- “使用 HF 评估技能将基准结果添加到我的模型卡中”
- “使用 HF 数据集技能从这些示例创建新数据集”您的代理会自动加载相应的`SKILL.md`指令和帮助程序脚本。

## 资源

- [Skills Repository](https://github.com/huggingface/skills) - 浏览并贡献
- [Agent Skills format](https://agentskills.io/home) - 规范和文档
- [CLI Guide](./agents-cli) - 用于 AI 代理的 Hugging Face CLI
- [MCP Guide](./agents-mcp) - 与技能一起使用

### 如何使用 Okta 配置 SCIM
https://huggingface.co/docs/hub/security-sso-okta-scim.md