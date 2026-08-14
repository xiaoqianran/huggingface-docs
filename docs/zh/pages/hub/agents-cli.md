<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 用于 AI 代理的 Hugging Face CLI

`hf` CLI 是将代理连接到 Hugging Face Hub 和生态系统的好方法。搜索模型、管理数据集和存储桶、启动 Spaces 以及从任何编码代理运行作业。

> [!提示]
> 这是有关使用 CLI 的代理的快速指南。欲了解更多详细信息，请参阅[CLI Reference itself](https://huggingface.co/docs/huggingface_hub/guides/cli)。

## 安装 CLI

确保 `hf` CLI 已安装并且是最新的。请参阅 [CLI installation guide](https://huggingface.co/docs/huggingface_hub/guides/cli#getting-started) 了解设置说明。

推荐的独立安装程序还会全局安装 CLI Skill，包括 Claude Code 位置，因此代理无需额外步骤即可获取它。通过 `--exclude-skill`（或 Windows 上的 `-ExcludeSkill`）来跳过它。

## 添加 CLI 技能

技能为您的代理提供有效使用工具所需的背景。独立安装程序为您添加了 CLI 技能 - 如果您使用 pip 或 Homebrew、跳过它或希望在单个项目中使用它，请手动安装它。在[agentskills.io](https://agentskills.io)了解有关技能的更多信息。

```bash
# install globally (available in all projects, works with Codex, Cursor, OpenCode,
# and any agent that loads skills from ~/.agents/skills)
hf skills add --global

# for Claude Code use the --claude flag
hf skills add --claude --global

# or install for the current project only (works with Codex, Cursor, OpenCode,
# and any agent that loads skills from .agents/skills)
hf skills add

# for Claude Code, use the --claude flag
hf skills add --claude
```> [!提示]
> 该技能是从您本地安装的 CLI 版本生成的，因此当您升级时它就会过时。 `hf update` 刷新全局安装的技能 - 永远不会重新添加您删除的技能 - `hf skills update -g` 自行刷新它。 `hf` 每天提醒您一次技能缺失或过时；使用 `HF_HUB_DISABLE_UPDATE_CHECK=1` 使其静音。

或者，您可以通过 Claude Code 插件系统安装：

```bash
claude
/plugin marketplace add huggingface/skills
/plugin install hf-cli@huggingface/skills
```

## 资源

- [CLI Reference](https://huggingface.co/docs/huggingface_hub/guides/cli) - 完整的命令文档
- [Token Settings](https://huggingface.co/settings/tokens) - 管理您的代币
- [Jobs Documentation](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-jobs) - 计算作业指南

### 技能
https://huggingface.co/docs/hub/agents-skills.md