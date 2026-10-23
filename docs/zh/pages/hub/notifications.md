<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 通知

通知可让您了解属于您正在观看的用户或组织的模型、数据集和空间何时发生新活动（**拉取请求或讨论**）。

默认情况下，如果出现以下情况，您将收到通知：

- 有人在讨论/公关中提到您。
- 您参与的讨论/公关中发布了新评论。
- 新的讨论/PR 或评论发布在您正在观看的组织或用户的存储库之一中。
- 有人回复您的帖子、博客文章或纸质页面。

![Notifications page](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/notifications-page.png)

您将通过电子邮件和[directly on the website](https://huggingface.co/notifications)收到新通知，您可以在[notifications settings](#notifications-settings)中更改此设置。

## 过滤和管理通知

在 [notifications page](https://huggingface.co/notifications) 上，您有多种选项可以更有效地过滤和管理您的通知：
 - 按存储库过滤：选择仅显示来自特定存储库的通知。
 - 按已读状态过滤：仅显示未读通知或所有通知。
 - 按参与过滤：显示您已参与或直接提及的通知。

此外，您可以采取以下操作来管理您的通知：- 标记为已读/未读：更改通知的状态以将其标记为已读或未读。
 - 标记为完成：标记为完成后，通知将不再出现在通知中心（它们将被删除）。
 
默认情况下，对通知所做的更改将仅应用于屏幕上选定的通知。但是，您还可以将更改应用于所有匹配的通知（例如 Gmail），以更加方便。

## 观察用户和组织

默认情况下，您将监视您所属的所有组织，并将收到有关这些组织的任何新活动的通知。

您还可以选择接收有关任意用户或组织的通知。为此，请使用 HF 配置文件上的“观看存储库”按钮。请注意，您还可以直接从 [notifications settings](#notifications-settings) 快速观看/取消观看用户和组织。

最后，您可以选择监视特定存储库并获得有关任何新活动的通知，而无需监视整个组织或用户帐户。

## 通知设置在您的[notifications settings](https://huggingface.co/settings/notifications)页面中，您可以根据活动类型选择接收通知的特定渠道，例如，收到直接提及的电子邮件，但仅收到有关关注的用户和组织的新活动的网络通知。默认情况下，您将收到有关任何新活动的电子邮件和网络通知，但您可以根据需要随意调整设置。

_请注意，单击电子邮件中的取消订阅链接将使您取消订阅该类型的活动，例如直接提及。_

![Notifications settings page](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/notifications-settings.png)

您可以使用专用搜索栏按名称搜索任何用户/组织，从而快速将其添加到您的监视列表中。
只需取消选中相应的复选框即可取消订阅特定用户/组织。

## 静音特定存储库的通知

可以通过使用存储库上下文菜单中的“静音通知”操作来静音特定存储库的通知。
这将阻止您收到该特定存储库的任何新通知。您可以随时通过单击同一存储库菜单中的“取消静音通知”操作来取消存储库静音。

![mute notification menu](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/notifications-mute-menu.png)_注意，如果存储库被静音，除非您被直接提及或参与讨论，否则您将不会收到任何新通知。_ 

静音存储库列表可从通知设置页面获取：

![Notifications settings page muted repositories](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/notifications-settings-muted.png)

## 静音特定讨论或 PR 的通知

您还可以通过单击标题中的静音图标来静音个别讨论或拉取请求的通知。这样做可以防止您收到来自该特定讨论或 PR 的任何进一步通知，包括直接提及。

您可以随时再次单击同一图标来取消静音。

![Notifications mute discussions](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/notifications-mute-discussion.png)

### 优化
https://huggingface.co/docs/hub/datasets-polars-optimizations.md