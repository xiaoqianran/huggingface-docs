<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 私有和门控数据集的身份验证

要访问私有或门控数据集，您需要在 DuckDB Secrets Manager 中配置 Hugging Face 令牌。

访问[Hugging Face Settings - Tokens](https://huggingface.co/settings/tokens)获取您的访问令牌。

> [!提示]
> 如果您通过 [⟦T4⟧](/docs/huggingface_hub/package_reference/cli#hf-datasets-sql) 查询，则会从您登录的会话（或 `--token`）自动配置令牌机密 — 无需手动设置机密。

DuckDB 支持两个提供者来管理机密：

- `CONFIG`：要求用户将所有配置信息传递到 CREATE SECRET 语句中。
- `CREDENTIAL_CHAIN`：自动尝试获取凭据。对于Hugging Face令牌，它将尝试从`~/.cache/huggingface/token`获取。

有关 DuckDB Secrets 的更多信息，请访问 [Secrets Manager](https://duckdb.org/docs/configuration/secrets_manager.html) 指南。

## 使用 `CONFIG` 提供商创建秘密

要使用 CONFIG 提供程序创建机密，请使用以下命令：

```bash
CREATE SECRET hf_token (TYPE HUGGINGFACE, TOKEN 'your_hf_token');
```

将 `your_hf_token` 替换为您实际的 Hugging Face 令牌。

## 使用 `CREDENTIAL_CHAIN` 提供者创建秘密

要使用 CREDENTIAL_CHAIN 提供程序创建密钥，请使用以下命令：

```bash
CREATE SECRET hf_token (TYPE HUGGINGFACE, PROVIDER credential_chain);
```

该命令自动从`~/.cache/huggingface/token`检索存储的令牌。

首先你需要[Login with your Hugging Face account](/docs/huggingface_hub/quick-start#login)，例如使用：

```bash
hf auth login
```

或者，您可以将 Hugging Face 令牌设置为环境变量：```bash
export HF_TOKEN="REDACTED"
```

有关身份验证的更多信息，请参阅 [Hugging Face authentication](/docs/huggingface_hub/main/en/quick-start#authentication) 文档。

### 古夫
https://huggingface.co/docs/hub/gguf.md