<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 秘密扫描

管理[your secrets (env variables) properly](./spaces-overview#managing-secrets)很重要。人们向外界公开秘密的最常见方式是直接将秘密硬编码到代码文件中，这使得恶意用户可以利用您的秘密以及您的秘密可以访问的服务。

例如，受损的 `app.py` 文件可能如下所示：

```py
import numpy as np
import scipy as sp

api_key = "REDACTED"

def call_inference(prompt: str) -> str:
    result = call_api(prompt, api_key)
    return result
```

为了防止出现此问题，我们会在您每次推送时运行 [TruffleHog](https://trufflesecurity.com/trufflehog)。 TruffleHog 扫描硬编码的秘密，我们将在检测到后向您发送一封电子邮件。

您只会收到经过验证的机密的电子邮件，这些机密已被确认可用于针对其各自提供商的身份验证。但请注意，未经验证的秘密不一定是无害或无效的：验证可能会因技术原因（例如网络错误）而失败。

TruffleHog 可以验证跨多个服务工作的秘密，它不仅限于 Hugging Face 令牌。

您可以选择退出来自 [your settings](https://huggingface.co/settings/notifications) 的电子邮件通知。

如果泄露的秘密是 Hugging Face 访问令牌，您可以立即使其失效 - 包括不属于您的令牌 - 请参阅[Revoking a leaked token](./security-tokens#revoking-a-leaked-token)。### 下载分析
https://huggingface.co/docs/hub/download-analytics.md