<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 值得信赖的出版商

**从 CI 推送到集线器，而不存储 `HF_TOKEN` 秘密。**

您的 CI 作业使用 CI 提供商提供的短期 OpenID Connect (OIDC) 令牌向 Hugging Face 证明其身份，并作为交换取回短期 Hugging Face 令牌。没有 HF 令牌可以作为秘密存储或轮换。

| |个人访问令牌 |值得信赖的出版商 |
| ---| ---| ---|
|终身|直至撤销| 1小时|
|存储| CI秘密|没有什么可存储的 |
|旋转|手册|自动，每次运行 |
|如果泄露|有效期至您撤销为止 |最多约 1 小时，范围仅限于一个存储库 |

受信任的发布者与 PyPI 的 [Trusted Publishers](https://docs.pypi.org/trusted-publishers/) 和 [npm's Trusted Publishing](https://docs.npmjs.com/trusted-publishers) 的想法相同。

## 简单示例：从 GitHub Actions 发布模型

您在 HF Hub 上维护 `acme/awesome-model`，并希望在外部托管的 `acme/awesome-model-training` 存储库（在 GitHub、GitLab 或任何符合 OIDC 的提供商上）中进行 CI 来推送新的检查点 — 无 `HF_TOKEN` 秘密。

### 1. 在 Hub 上配置受信任的发布者

在 `https://huggingface.co/acme/awesome-model/settings` 上，打开 **受信任的发布者** 并添加：

- **提供者**：GitHub Actions
- **声明**（所有声明都必须匹配才能成功交换）：
  - `repository` = `acme/awesome-model-training`
  - `branch` = `main`
  - `workflow` = `publish.yml`> [!提示]
> `repository` 单独将发布者范围限定为 GitHub 存储库。添加 `branch` 和/或 `workflow` 也可以将其固定到分支或工作流程文件 - 推荐。

### 2. 将工作流程添加到您的 GitHub 存储库

`.github/workflows/publish.yml`：

```yaml
name: Publish to Hugging Face

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  publish:
    runs-on: ubuntu-latest

    permissions:
      id-token: write  # required so the job can request an OIDC token
      contents: read

    steps:
      - uses: actions/checkout@v4

      - name: Install the hf CLI
        run: |
          curl -LsSf https://hf.co/cli/install.sh | bash
          echo "$HOME/.local/bin" >> "$GITHUB_PATH"

      - name: Upload checkpoint
        env:
          # The HF repo to publish to. For non-model repos, prefix accordingly:
          #   datasets/acme/awesome-dataset, spaces/acme/awesome-space, kernels/acme/awesome-kernel
          HF_OIDC_RESOURCE: acme/awesome-model
        run: hf upload acme/awesome-model ./checkpoint . --commit-message "Publish from ${GITHUB_SHA::7}"
```

就是这样。在 GitHub Actions 上，`hf` CLI (`huggingface_hub>=1.19.0`) 检测提供者、执行交换并自动使用生成的令牌。您只需设置`HF_OIDC_RESOURCE`。

> [!提示]
> 在一次运行中发布到多个存储库（例如模型**和**数据集）？每个步骤设置`HF_OIDC_RESOURCE`，以便每个令牌的范围仅限于该步骤推送到的存储库。

#### 其他 CI 提供商

目前，`hf` CLI 仅在 GitHub Actions 上本地生成 ID 令牌。在 GitLab、CircleCI、Bitbucket 或任何其他提供商上，您自己创建 ID 令牌（请参阅[Supported CI providers](#supported-ci-providers)）并通过 `HF_OIDC_ID_TOKEN` 传递它 — CLI 直接交换它：

```yaml
# GitLab CI (.gitlab-ci.yml)
publish:
  id_tokens:
    HF_ID_TOKEN:
      aud: https://huggingface.co
  script:
    - curl -LsSf https://hf.co/cli/install.sh | bash
    - export PATH="$HOME/.local/bin:$PATH"
    - HF_OIDC_ID_TOKEN="REDACTED" HF_OIDC_RESOURCE="acme/awesome-model" hf upload acme/awesome-model ./checkpoint .
```

完整的工作示例：

- GitHub 操作 — [⟦T28⟧](https://github.com/coyotte508/publish-to-hf)
- GitLab CI — [⟦T29⟧](https://gitlab.com/coyotte508/publish-to-hf)

## 两种风格：repo 与 user|风味 |配置于 |你得到什么 |用它来... |
| ---| ---| ---| ---|
| **回购发布者** |存储库的**设置 → 受信任的发布者** |具有**对该一个存储库的写入权限**的令牌 |从 CI 发布模型、数据集、空间、内核或存储桶 |
| **用户发布者** |您的帐户是 [**Authentication settings → CI/CD Access**](https://huggingface.co/settings/authentication#ci-cd-access) |具有 `gated-repos` 范围的只读令牌 |阅读您有权访问的**门控存储库并使用 CI | 的速率限制

两个令牌都会在 60 分钟后过期。您需要 Hub 存储库上的 **Write** 角色来管理其受信任的发布者。

### 从 CI 访问门控存储库

如果您只需要*读取*门控存储库（例如从作业下载模型），请在您的帐户上的[**Authentication settings → CI/CD Access**](https://huggingface.co/settings/authentication#ci-cd-access)下配置发布者，而不是在特定的存储库上，然后使用您的**用户名**作为资源。

在 **GitHub Actions** 上，`hf` CLI 会为您进行交换，只需设置 `HF_OIDC_RESOURCE`：

```yaml
      - name: Download a gated model
        env:
          HF_OIDC_RESOURCE: your-hf-username
        run: hf download acme/gated-model
```

在 **其他提供商** 上，您自己铸造 ID 令牌（请参阅[Other CI providers](#other-ci-providers)）并通过 `HF_OIDC_ID_TOKEN` 传递它：

```bash
# $ID_TOKEN is the OIDC token your provider minted (e.g. $HF_ID_TOKEN on GitLab)
HF_OIDC_ID_TOKEN="$ID_TOKEN" HF_OIDC_RESOURCE="your-hf-username" hf download acme/gated-model
```

生成的令牌可以读取您有权访问的门控存储库并使用您帐户的速率限制。它**无法**写入任何内容，并且**无法**读取您的私人存储库。> [!提示]
> 需要令牌本身（对于 `curl`、`git clone` 或读取 `HF_TOKEN` 的工具）？ `hf auth token` 执行交换并将短期令牌打印到标准输出：
>
> ```yaml
>       - name: Get a short-lived HF token
>         env:
>           HF_OIDC_RESOURCE: your-hf-username  # or a repo, e.g. acme/awesome-model
>         run: echo "HF_TOKEN=$(hf auth token)" >> "$GITHUB_ENV"
> ```
>
> 这适用于用户发布者和存储库发布者 - 令牌的范围仅限于 `HF_OIDC_RESOURCE` 指向的任何内容。

## 支持的 CI 提供商

设置 UI 附带了以下提供商的预设，但任何符合 OIDC 的提供商都可以使用（AWS、GCP、Buildkite、您自己的 IdP，...）。

|供应商|发行人 |如何获取ID令牌|
| ---| ---| ---|
| **GitHub 操作** | `https://token.actions.githubusercontent.com` |设置`permissions: id-token: write`，然后使用`audience=https://huggingface.co`调用元数据端点。 [Docs](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect)。 |
| **亚搏体育appGitLab CI** | `https://gitlab.com`（或您的自托管 URL）|在工作中声明`id_tokens: { HF_ID_TOKEN: { aud: https://huggingface.co } }`；阅读`$HF_ID_TOKEN`。 [Docs](https://docs.gitlab.com/ci/yaml/#id_tokens)。 |
| **圆CI** | `https://oidc.circleci.com/org/<org-uuid>` |使用`$CIRCLE_OIDC_TOKEN_V2`（v2 允许您在项目设置中设置受众）。 [Docs](https://circleci.com/docs/openid-connect-tokens/)。 |
| **Bitbucket 管道** | `https://api.bitbucket.org/2.0/workspaces/<workspace>/pipelines-config/identity/oidc` |在台阶上设置`oidc: true`；阅读`$BITBUCKET_STEP_OIDC_TOKEN`。 [Docs](https://support.atlassian.com/bitbucket-cloud/docs/integrate-pipelines-with-resource-servers-using-oidc/)。 |

一旦您拥有 ID 令牌，跨提供商的交换调用是相同的 - 只有您在集线器端配置的**声明**有所不同。

## 它是如何工作的1. 您的 CI 提供商会创建一个短暂的 **OIDC ID 令牌** 来描述作业（哪个存储库、哪个分支、哪个工作流程……）。
2. 您的工作流程 `POST` 将该令牌连同 `resource`（它想要访问的存储库或用户名）一起发送给 `https://huggingface.co/oauth/token`。
3. Hub 根据您为该资源配置的发布者检查令牌的签名和声明，并返回 Hugging Face 令牌。

```
┌──────────┐  1. mint ID token   ┌──────────┐  2. exchange   ┌────────────┐
│ CI job   │ ─────────────────▶  │ CI OIDC  │ ──────────────▶│ huggingface│
│          │                     │ issuer   │                │  /oauth/   │
│          │ ◀──────────────────────────────────────────────│  token     │
└──────────┘        3. short-lived HF token (valid 1 h)      └────────────┘
```

## API 参考

端点、请求和响应

**端点：** `POST https://huggingface.co/oauth/token` 与 `Content-Type: application/json`。

不需要客户端身份验证 - OIDC ID 令牌对请求进行身份验证。交换遵循[RFC 8693 — OAuth 2.0 Token Exchange](https://www.rfc-editor.org/rfc/rfc8693.html)。

**请求正文：**

|领域|必填|价值|
| ---| ---| ---|
| `grant_type` |是的 | `urn:ietf:params:oauth:grant-type:token-exchange` |
| `subject_token_type` |是的 | `urn:ietf:params:oauth:token-type:id_token` |
| `subject_token` |是的 |来自 CI 提供商的原始 OIDC ID 令牌 (JWT)。其`aud`声称**必须**是`https://huggingface.co`。 |
| `resource` |是的 |用于用户范围令牌的 Hub 存储库（`namespace/name`、`datasets/namespace/name`、`spaces/namespace/name`、`kernels/namespace/name`、`buckets/namespace/name`）或 Hub **用户名**（无斜线）。 |

**成功响应：**

```json
{
  "access_token":      "hf_jwt_…",   // "hf_oauth_…" for user resources
  "token_type":        "bearer",
  "expires_in":        3600,
  "issued_token_type": "urn:ietf:params:oauth:token-type:access_token"
}
```

**错误** — 带有 OAuth 样式正文的 `400 Bad Request`：| `error` |为什么 |
| ---| ---|
| `invalid_request` |参数丢失/格式错误，或 `resource` 格式错误。 |
| `invalid_grant` |未找到存储库或用户；没有发行商与该发行人匹配；配置的声明不匹配；签名或观众检查失败；帐户被锁定。 |

当 `hf` CLI 执行交换时，故障会显示 `error` 代码以及 `(Request ID: …)` — 在报告问题时包含该请求 ID，以便我们可以在日志中跟踪交换。

## 安全模型

- **令牌是短暂的。** 自交换之日起 60 分钟 — 时钟仅在您调用端点时启动，而不是在工作流程启动时启动。没有刷新令牌；长期工作应该重新调换。
- **回购代币是回购范围的。** `acme/awesome-model` 的代币不能触及 `acme/anything-else`。推送归因于合成的`[OIDC]`系统用户，并参考原始发行者和主题。
- **用户令牌是只读的。**仅限`gated-repos`范围 - 没有写入，没有私人存储库，没有帐户管理。
- - **声明完全匹配。**没有正则表达式，没有前缀匹配。
- **审核日志。** 记录添加或删除发布者，并成功交换更新上次使用的时间。

## 另请参阅- [User Access Tokens](./security-tokens) — 人类和一次性脚本的正确选择
- [OAuth / Sign in with HF](./oauth) — 相同的 `/oauth/token` 端点，用于交互流
- [Managing Spaces with GitHub Actions](./spaces-github-actions)
- [GitHub Actions integration for the Hub](./repositories-github-actions)

### 身份验证
https://huggingface.co/docs/hub/datasets-polars-auth.md