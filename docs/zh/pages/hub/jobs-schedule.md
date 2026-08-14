<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 安排工作

安排和管理将在 HF 基础设施上运行的作业。

将 `hf jobs uv run ` 或 `hf jobs run` 与`@annually`、`@yearly`、`@monthly`、`@weekly`、`@daily`、`@hourly` 的计划或 CRON 计划表达式（例如，每周一上午 9 点使用`"0 9 * * 1"`）配合使用：

```bash
# Schedule a job that runs every hour
>>> hf jobs scheduled uv run @hourly python -c "print('This runs every hour!')"

# Use the CRON syntax
>>> hf jobs scheduled uv run "*/5 * * * *" python -c "print('This runs every five minutes!')"

# Schedule with GPU
>>> hf jobs scheduled uv run --flavor a10g-small --with torch @hourly python -c 'import torch; print(f"This code ran with the following GPU: {torch.cuda.get_device_name()}")'

# Schedule with a Docker image
>>> hf jobs scheduled run @hourly python:3.12 python -c "print('This runs every hour!')"

# Schedule a Python script with a label
>>> hf jobs scheduled uv run --label fine-tuning @hourly my_script.py

# Schedule a named job (names show up in the UI and do not have to be unique)
>>> hf jobs scheduled run --name hourly-task @hourly python:3.12 python -c "print('This runs every hour!')"
```

使用与`hf jobs uv run`和`hf jobs run`相同的参数来传递环境变量、秘密、超时、标签等。

使用 `hf jobs scheduled ps`、`hf jobs scheduled inspect`、`hf jobs scheduled suspend`、`hf jobs scheduled resume`、`hf jobs scheduled trigger` 和 `hf jobs scheduled delete` 管理计划作业：

```python
# List your active scheduled jobs
>>> hf jobs scheduled ps

# List all your scheduled jobs (including suspended jobs)
>>> hf jobs scheduled ps -a

# Inspect the status of a job
>>> hf jobs scheduled inspect <scheduled-job-id>

# Suspend (pause) a scheduled job
>>> hf jobs scheduled suspend <scheduled-job-id>

# Resume a scheduled job
>>> hf jobs scheduled resume <scheduled-job-id>

# Trigger a scheduled job to run right now (does not change the schedule)
>>> hf jobs scheduled trigger <scheduled-job-id>

# Delete a scheduled job
>>> hf jobs scheduled delete <scheduled-job-id>
```

### 基本单点登录
https://huggingface.co/docs/hub/security-sso-basic.md