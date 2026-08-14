<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 门控组集合

> [!警告]
> 此功能是团队和企业计划的一部分。

门控组集合允许组织一次授予（或拒绝）对集合中所有模型和数据集的访问权限，而不是针对每个存储库。用户只需完成**一次访问请求**。

要在集合中启用门控组：

- 馆藏所有者必须是一个组织
- 组织必须订阅团队或企业计划
- 集合中的所有模型和数据集必须由与集合相同的组织拥有
- 集合中的每个模型或数据集可能只属于一个门控组集合（但它们仍然可以包含在非门控即_常规_集合中）。

> [!提示]
> 门控仅适用于模型和数据集；集合的任何其他资源部分（例如空间或纸张）不会受到影响。

## 作为组织管理员管理门控组

要启用访问请求，请转到集合页面并单击右下角的“**门控组**”。

    <img
        class="block dark:hidden m-0!"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/gating-group-collection-disabled.webp"
        alt="Hugging Face collection page with gating group collection feature disabled"
    />
    <img
        class="hidden dark:block m-0!"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/dark-gating-group-collection-disabled.webp"
        alt="Hugging Face collection page with gating group collection feature disabled"
    />

默认情况下，门控组处于禁用状态：单击“**配置访问请求**”打开设置

    <img
        class="block dark:hidden m-0!"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/gating-group-modal-disabled.webp"
        alt="Hugging Face gating group collection settings with gating disabled"
    />
    <img
        class="hidden dark:block m-0!"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/dark-gating-group-modal-disabled.webp"
        alt="Hugging Face gating group collection settings with gating disabled"
    />默认情况下，当用户请求时，会自动授予对集合中存储库的访问权限。这称为**自动批准**。在这种模式下，任何用户一旦同意与您分享他们的联系信息，就可以访问您的存储库。

    <img
        class="block dark:hidden m-0!"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/gating-group-modal-enabling.webp"
        alt="Hugging Face gating group collection settings with automatic mode selected"
    />
    <img
        class="hidden dark:block m-0!"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/dark-gating-group-modal-enabling.webp"
        alt="Hugging Face gating group collection settings with automatic mode selected"  
    />

如果您想手动批准哪些用户可以访问您集合中的存储库，则必须将其设置为 **手动审核**。在这种情况下，您会注意到一个新选项：

**通知频率**，可让您配置何时收到有关请求访问的新用户的通知。可以设置为每天一次或实时一次。默认情况下，电子邮件会发送给组织的前 5 位管理员。您还可以在 **通知电子邮件** 字段中设置不同的电子邮件地址。

    <img
        class="block dark:hidden m-0!"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/gating-group-modal-manual.webp"
        alt="Hugging Face gating group collection settings with manual review mode selected"
    />
    <img
        class="hidden dark:block m-0!"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/dark-gating-group-modal-manual.webp"
        alt="Hugging Face gating group collection settings with manual review mode selected"
    />

### 审查访问请求

启用访问请求后，您可以完全控制谁可以访问门控组集合中的存储库，无论审批模式是手动还是自动。您可以从 UI 或通过 API 查看和管理请求。

**批准对门控组集合中的存储库的请求将自动批准对该集合中所有存储库（模型和数据集）的访问。**#### 从用户界面

您可以通过单击“**查看访问请求**”按钮，从集合中任何存储库的设置页面查看谁有权访问 Gating Group 集合中的所有存储库：

    <img
        class="block dark:hidden m-0!"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/gating-group-repo-settings.webp"
        alt="Hugging Face repo access settings when repo is in a gating group collection"
    />
    <img
        class="hidden dark:block m-0!"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/dark-gating-group-repo-settings.webp"
        alt="Hugging Face repo access settings when repo is in a gating group collection"
    />

这将打开一个包含 3 个用户列表的模式：

- **待处理**：等待批准访问您的存储库的用户列表。除非您选择了**手动审核**，否则此列表为空。您可以**接受**或**拒绝**每个请求。如果请求被拒绝，用户将无法访问您的存储库，也无法再次请求访问。
- **已接受**：有权访问您的存储库的用户的完整列表。您可以随时选择**拒绝**任何用户的访问，无论审批模式是手动还是自动。您还可以**取消**批准，这会将用户移至**待定**列表。
- **拒绝**：您手动拒绝的用户列表。这些用户无法访问您的存储库。如果他们访问您的存储库，他们将看到一条消息_您访问此存储库的请求已被存储库作者拒绝_。

    <img 
        class="block dark:hidden"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/models-gated-enabled-pending-users.png"
        alt="Manage access requests modal for a repo in a gating group collection"
    />
    <img
        class="hidden dark:block"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/models-gated-enabled-pending-users-dark.png"
        alt="Manage access requests modal for a repo in a gating group collection"
    />

#### 通过 API您可以通过任何模型或数据集的 API 以编程方式管理门控组集合中的访问请求。

请访问我们的 [gated models](https://huggingface.co/docs/hub/models-gated#via-the-api) 或 [gated datasets](https://huggingface.co/docs/hub/datasets-gated#via-the-api) 文档以了解更多信息。

#### 下载访问报告

您可以通过任何模型或数据集的设置页面下载门控组集合的访问报告。

请访问我们的 [gated models](https://huggingface.co/docs/hub/models-gated#download-access-report) 或 [gated datasets](https://huggingface.co/docs/hub/datasets-gated#download-access-report) 文档以了解更多信息。

#### 自定义请求的信息

组织可以自定义门控参数以及每个门控存储库收集的用户信息。请访问我们的 [gated models](https://huggingface.co/docs/hub/models-gated#customize-requested-information) 或 [gated datasets](https://huggingface.co/docs/hub/datasets-gated#customize-requested-information) 文档了解更多详细信息。

> [!警告]
> 目前无法集中定制闸门参数和请求信息。如果您希望无论用户请求访问哪个集合的存储库，都收集相同的数据，则需要在该集合的所有模型和数据集的元数据中添加相同的门参数，并保持同步。

＃＃ 高级设置Enterprise Plus 组织可以在其所有模型和数据集中自动拒绝或阻止来自特定位置的访问者。这些设置位于**发布商分析**设置中的**高级控制**选项卡下。

### 执行

**强制**设置控制拒绝哪些被阻止的访问者：

- **门控存储库**：对组织的门控存储库的访问请求会被自动拒绝。
- **所有存储库**：除了自动拒绝访问请求之外，还会拒绝组织的每个存储库（包括公共存储库）的下载。被阻止的访问者会在存储库页面上看到“在您所在的区域不可用”通知，并且数据集查看器对他们禁用。

强制执行适用于来自被封锁国家和地区的访问者，无论他们是否登录。该组织的成员也不例外。

### 被封锁的地点

两个列表定义了阻止的位置：

- **被封锁的国家**：由 [ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) 代码标识的国家/地区。
- **封锁地区**：除了上述国家/地区之外，要封锁的特定地区，因为这些地区不是不同的国家/地区，因此无法在上面的列表中选择。## 以用户身份访问门控组集合中的门控存储库

门控组集合在其名称旁边显示一个特定图标：

    <img
        class="block dark:hidden m-0!"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/gating-group-collection-enabled.webp"
        alt="Hugging Face collection page with gating group collection feature enabled"
    />
    <img
        class="hidden dark:block m-0!"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/dark-gating-group-collection-enabled.webp"
        alt="Hugging Face collection page with gating group collection feature enabled"
    />

要访问门控组集合中的模型和数据集，需要在任何这些存储库的页面上发出单个访问请求。一旦您的请求获得批准，您将能够访问集合中的所有其他存储库，包括未来的存储库。

请访问我们的 [gated models](https://huggingface.co/docs/hub/models-gated#access-gated-models-as-a-user) 或 [gated datasets](https://huggingface.co/docs/hub/datasets-gated#access-gated-datasets-as-a-user) 文档，了解有关请求访问存储库的更多信息。

### HF PRO 订阅 🔥
https://huggingface.co/docs/hub/pro.md