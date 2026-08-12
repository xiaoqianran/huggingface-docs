<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 门控数据集

为了更好地控制数据集的使用方式，中心允许数据集作者为其数据集启用**访问请求**。用户必须同意与数据集作者共享其联系信息（用户名和电子邮件地址），才能在启用后访问数据集文件。数据集作者可以使用其他字段配置此请求。启用访问请求的数据集称为**门控数据集**。访问请求始终授予个人用户而不是整个组织。门控数据集的一个常见用例是在更广泛的发布之前提供对早期研究数据集的访问。

## 作为数据集作者管理门控数据集

 
 

要启用访问请求，请转到数据集设置页面。默认情况下，数据集没有门控。单击右上角的“**启用访问请求**”。

    
    

默认情况下，当用户请求数据集时，系统会自动授予对数据集的访问权限。这称为**自动批准**。在此模式下，任何用户在与您共享个人信息后都可以访问您的数据集。如果您想手动批准哪些用户可以访问您的数据集，则必须将其设置为**手动批准**。在这种情况下，您会注意到更多选项：
- **添加访问权限**允许您搜索用户并授予他们访问权限，即使他们没有请求。
- **通知频率**可让您配置在新用户请求访问时何时收到通知。可以设置为每天一次或实时一次。默认情况下，电子邮件会发送到您的主要电子邮件地址。对于组织下托管的数据集，电子邮件默认发送给组织的前 5 位管理员。在这两种情况下（用户或组织），您都可以在 **通知电子邮件** 字段中设置不同的电子邮件地址。

    
    

### 审查访问请求

启用访问请求后，您可以完全控制谁可以访问您的数据集，无论批准模式是手动还是自动。您可以从 UI 或通过 API 查看和管理请求。

### 从用户界面您可以通过单击“**查看访问请求**”按钮，从设置页面查看谁有权访问您的门控数据集。这将打开一个包含 3 个用户列表的模式：
- **待处理**：等待批准访问您的数据集的用户列表。除非您选择了**手动批准**，否则此列表为空。您可以**接受**或**拒绝**该需求。如果请求被拒绝，用户将无法访问您的数据集，也无法再次请求访问。
- **已接受**：有权访问您的数据集的用户的完整列表。您可以随时选择**拒绝**任何用户的访问，无论审批模式是手动还是自动。您还可以**取消**批准，这会将用户移至*待处理*列表。
- **拒绝**：您手动拒绝的用户列表。这些用户无法访问您的数据集。如果他们访问您的数据集存储库，他们将看到一条消息 *您访问此存储库的请求已被存储库作者拒绝*。

    
    

#### 通过 API

您可以使用 API 自动批准访问请求。您必须通过 `token` 和 `write` 访问门控存储库。要生成令牌，请转到[your user settings](https://huggingface.co/settings/tokens)。|方法|统一资源定位符 |描述 |标题 |有效载荷
| ------ | ---| ----------- | -------- | -------- |
| `GET` | `/api/datasets/{repo_id}/user-access-request/pending` |检索待处理请求的列表。 | `{"authorization": "Bearer $token"}` | |
| `GET` | `/api/datasets/{repo_id}/user-access-request/accepted` |检索已接受请求的列表。 | `{"authorization": "Bearer $token"}` | |
| `GET` | `/api/datasets/{repo_id}/user-access-request/rejected` |检索被拒绝的请求的列表。 | `{"authorization": "Bearer $token"}` | |
| `POST` | `/api/datasets/{repo_id}/user-access-request/handle` |将给定访问请求的状态更改为`status`。 | `{"authorization": "Bearer $token"}` | `{"status": "accepted"/"rejected"/"pending", "user": "username", "rejectionReason": "Optional rejection reason that will be visible to the user (max 200 characters)."}}` |
| `POST` | `/api/datasets/{repo_id}/user-access-request/grant` |允许特定用户访问您的存储库。 | `{"authorization":  "Bearer $token"}` | `{"user": "username"} ` |

上述 HTTP 端点的基本 URL 是 `https://huggingface.co`。

**新！** 我们的 Python 客户端 `huggingface_hub` 现已正式支持这些端点。使用 [⟦T27⟧](/docs/huggingface_hub/main/en/package_reference/hf_api#huggingface_hub.HfApi.list_pending_access_requests)、[⟦T28⟧](/docs/huggingface_hub/main/en/package_reference/hf_api#huggingface_hub.HfApi.list_accepted_access_requests) 和 [⟦T29⟧](/docs/huggingface_hub/main/en/package_reference/hf_api#huggingface_hub.HfApi.list_rejected_access_requests) 列出对数据集的访问请求。您还可以使用[⟦T30⟧](/docs/huggingface_hub/main/en/package_reference/hf_api#huggingface_hub.HfApi.accept_access_request)、[⟦T31⟧](/docs/huggingface_hub/main/en/package_reference/hf_api#huggingface_hub.HfApi.cancel_access_request)、[⟦T32⟧](/docs/huggingface_hub/main/en/package_reference/hf_api#huggingface_hub.HfApi.reject_access_request)接受、取消和拒绝访问请求。最后，您可以使用 [⟦T33⟧](/docs/huggingface_hub/main/en/package_reference/hf_api#huggingface_hub.HfApi.grant_access) 向用户授予访问权限。

### 下载访问报告您可以使用 **下载用户访问报告** 按钮下载门控数据集的所有访问请求的报告。单击它可下载包含用户列表的 json 文件。对于每个条目，您拥有：
- **用户**：用户 ID。示例：*julien-c*。
- **全名**：集线器上用户的名称。示例：*朱利安·肖蒙*。
- **状态**：请求的状态。 `"pending"`、`"accepted"` 或 `"rejected"`。
- **电子邮件**：用户的电子邮件。
- **时间**：用户最初发出请求时的日期时间。
- **reviewedAt**：请求被接受或拒绝的日期时间。未设置待处理请求。

 

### 自定义请求的信息

默认情况下，登录您的门控数据集的用户将被要求通过单击“**同意并发送访问存储库的请求**”按钮来分享他们的联系信息（电子邮件和用户名）。

    
    

如果您想请求更多用户信息以提供访问权限，您可以配置其他字段。可以从“**设置**”选项卡访问此信息。为此，请将 `extra_gated_fields` 属性添加到包含键/值对列表的 [dataset card metadata](./datasets-cards#dataset-card-metadata) 中。 *key* 是字段的名称，*value* 是其类型或具有 `type` 字段的对象。字段类型列表为：- `text`：单行文本字段。
- `checkbox`：复选框字段。
- `date_picker`：日期选择器字段。
- `country`：国家/地区下拉列表。国家列表基于[ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2)标准。
- `select`：带有选项列表的下拉菜单。选项列表在 `options` 字段中定义。示例：`options: ["option 1", "option 2", {label: "option3", value: "opt3"}]`。

最后，您还可以使用 `extra_gated_prompt` 额外字段个性化向用户显示的消息。

以下是自定义请求表单的示例，其中要求用户提供其公司名称和国家/地区，并确认数据集仅供非商业用途。

```yaml
---
extra_gated_prompt: "You agree to not use the dataset to conduct experiments that cause harm to human subjects."
extra_gated_fields:
  Company: text
  Country: country
  Specific date: date_picker
  I want to use this dataset for:
    type: select
    options: 
      - Research
      - Education
      - label: Other
        value: other
  I agree to use this dataset for non-commercial use ONLY: checkbox
---
```

在某些情况下，您可能还想修改门标题、描述和按钮中的默认文本。 For those use cases, you can modify `extra_gated_heading`, `extra_gated_description` and `extra_gated_button_content` like this:

```yaml
---
extra_gated_heading: "Acknowledge license to accept the repository"
extra_gated_description: "Our team may take 2-3 days to process your request"
extra_gated_button_content: "Acknowledge license"
---
```

### 组织成员的访问权限

对于组织下托管的门控数据集，您还可以要求组织的**自己的成员**提交访问请求。在数据集设置页面的门控选项下，启用 **Alsogate access for `{org}`** 的成员。启用此选项后，组织成员必须像任何其他用户一样请求访问数据集。以下角色绕过请求并保持直接访问：

- 组织管理员
- 创建存储库的用户
- [Resource Group](./security-resource-groups) 管理员，当数据集属于资源组时

所有其他成员必须完成访问请求流程。这包括具有读取、贡献者或写入组织角色的成员，以及不具有管理员角色的资源组成员。

## 作为一个组织（团队和企业）管理门控数据集

[Team & Enterprise](https://huggingface.co/docs/hub/en/enterprise) 订阅者可以创建门控组集合，以一次性授予（或拒绝）对集合中所有模型和数据集的访问权限。

有关门控组集合的更多信息可以在[our dedicated doc](https://huggingface.co/docs/hub/en/enterprise-gating-group-collections)中找到。

## 以用户身份访问门控数据集

作为用户，如果您想使用门控数据集，您将需要请求访问它。这意味着您必须登录 Hugging Face 用户帐户。

请求访问只能通过您的浏览器完成。转到 Hub 上的数据集，系统将提示您共享您的信息：单击 **同意**，即表示您同意与数据集作者共享您的用户名和电子邮件地址。在某些情况下，可能会要求附加字段。为了帮助数据集作者决定是否授予您访问权限，请尝试尽可能完整地填写表格。

一旦发送访问请求，就有两种可能性。如果批准机制是自动的，您可以立即访问数据集文件。否则，请求必须由作者手动批准，这可能需要更多时间。

> [!警告]
> 数据集作者对数据集访问拥有完全控制权。特别是，他们可以随时决定阻止您访问数据集，恕不另行通知，无论批准机制如何或您的请求是否已获得批准。

### 下载文件

要从门控数据集中下载文件，您需要经过身份验证。在浏览器中，只要您使用帐户登录，此操作就会自动进行。如果您使用脚本，则需要提供 [user token](./security-tokens)。在Hugging Face Python生态系统（`transformers`、`diffusers`、`datasets`等）中，您可以使用[⟦T54⟧](/docs/huggingface_hub/index)库登录您的机器并在终端中运行：

```bash
hf auth login
```或者，您可以在笔记本或脚本中使用 `login()` 以编程方式登录：

```python
>>> from huggingface_hub import login
>>> login()
```

您还可以直接从脚本向库中的大多数加载方法（`from_pretrained`、`hf_hub_download`、`load_dataset` 等）提供 `token` 参数。

有关如何登录的更多详细信息，请查看[login guide](/docs/huggingface_hub/quick-start#login)。

### 限制欧盟用户的访问

对于门控数据集，您可以添加额外的访问控制层以专门限制来自欧盟国家/地区的用户。如果您的数据集的许可或使用条款禁止其在欧盟分发，这非常有用。

要启用此功能，请将 `extra_gated_eu_disallowed: true` 属性添加到数据集卡的元数据中。

**重要提示：** 仅当您的数据集已被门控时，此功能才会激活。如果`gated: false`或未设置该属性，则此限制不适用。

```yaml
---
license: mit
gated: true
extra_gated_eu_disallowed: true
---
```

系统根据用户的 IP 地址识别用户的位置。

### 在拥抱脸部使用 ESPnet
https://huggingface.co/docs/hub/espnet.md