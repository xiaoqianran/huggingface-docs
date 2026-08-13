<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 🟧 空间标签工作室

[Label Studio](https://labelstud.io) 是用于标签的 [open-source data labeling
platform](https://github.com/heartexlabs/label-studio)，
注释并探索许多不同的数据类型。此外，标签工作室
包括一个强大的[machine learning
interface](https://labelstud.io/guide/ml.html)，可用于新型号
培训、主动学习、监督学习和许多其他培训
技术。

本指南将教您如何部署 Label Studio 进行数据处理
Hugging Face Hub 中的标签和注释。您可以使用默认的
将 Label Studio 配置为完全托管的独立应用程序
在 Hub 上使用 Docker 进行演示和评估，或者您也可以
附加您自己的数据库和云存储来托管功能齐全的
托管在 Spaces 上的生产就绪应用程序。

## ⚡️ 在 Spaces 上部署 Label Studio

只需单击几下，您就可以在 Spaces 上部署 Label Studio：

  

空间要求您定义：

* **所有者**：您的个人帐户或您所属的组织
  的一部分。 

* A **空间名称**：帐户内空间的名称
  你正在创造空间。* **可见性**：_私人_如果您想要
  仅您或您的组织可见的空间，如果您愿意，也可以_公开_
  使用 Label Studio API 对其他用户或应用程序可见
  （建议）。

## 🚀 使用默认配置

默认情况下，Label Studio 安装在 Spaces 中，配置使用
应用程序数据库的本地存储，用于存储配置、帐户
凭证和项目信息。还保存标记任务和数据项
在本地存储中。 

> [!警告]
> 默认情况下，拥抱面部空间中的存储是短暂的。保存您的数据
> 在重新启动时，附加一个 [Storage Bucket](./storage-buckets) — 请参阅
> [persistence section](#enable-persistence-with-hf-storage-buckets) 下面。启动 Label Studio 后，您将看到标准登录信息
屏幕。您可以首先使用您的电子邮件地址创建一个新帐户，然后
使用您的新凭据登录。登录后定期，标签
Studio 会警告您存储是短暂的，数据可能会丢失
如果您的空间重新启动，则会丢失。您还将收到以下提示的预设
Heidi，乐于助人的 Label Studio 吉祥物，用于创建一个新项目来启动
标记您的数据。要开始使用，请查看 Label Studio ["Zero to One"
tutorial](https://labelstud.io/blog/introduction-to-label-studio-in-hugging-face-spaces/)
有关如何构建用于情感分析的注释界面的指南。 

## 🛠️ 配置 Label Studio 的生产就绪实例

为了使您的 Space 做好生产准备，您需要进行三项配置
变化：

* 禁止无限制地创建新帐户。

* 通过附加[Storage Bucket](./storage-buckets)或外部数据库来启用持久性。

* （可选）附加云存储以执行标记任务。

### 禁用无限制创建新帐户Label Studio 上的默认配置允许无限制的创建
为任何拥有您的应用程序 URL 的人创建新帐户。你可以
[restrict signups](https://labelstud.io/guide/signup.html#Restrict-signup-for-local-deployments)
通过将以下配置机密添加到您的空间**设置**。

* `LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK`：将此值设置为`true`将
  禁用无限制的帐户创建。 

* `LABEL_STUDIO_USERNAME`：这是您要使用的帐户的用户名
  作为 Label Studio 空间中的第一个用户使用。它应该是一个有效的电子邮件
  地址。

* `LABEL_STUDIO_PASSWORD`：将与第一个关联的密码
   用户帐户。

重新启动空间以应用这些设置。创建新帐户的能力
从登录屏幕将被禁用。要创建新帐户，您需要
在 Label Studio 的 `Organization` 设置中邀请新用户
应用程序。

### 使用 HF 存储桶实现持久性

默认情况下，该Space将所有项目配置和数据注释存储在
使用 SQLite 进行本地存储。如果重置空间，所有配置和
空间中的注释数据将丢失。启用持久性的最简单方法是附加一个[Storage Bucket](./storage-buckets)，
它将持久对象存储直接安装到空间中。标签工作室
将其 SQLite 数据库和媒体上传写入已安装的存储桶中，因此
项目和注释在重新启动后仍然有效。

1. **创建桶：**

   ```bash
   hf buckets create <your-namespace>/label-studio-data
   ```

2. **在空间设置→存储桶中附加**，挂载路径`/data`。

3. **设置两个空间变量：**

   ```
   LABEL_STUDIO_BASE_DATA_DIR=/data
   STORAGE_PERSISTENCE=1
   ```

4. **工厂重建**空间。

> [!提示]
> 设置 `SECRET_KEY` Space Secret 以在重新启动后保持用户会话处于活动状态。
> 如果没有它，Label Studio 会在每次启动时和所有用户生成一个随机密钥
> 重新启动时注销。

#### 让编码代理为您做这件事

如果您不想点击空间设置，您可以要求有权访问 `huggingface_hub` 的编码代理为您配置空间。告诉它你的空间 ID 和存储桶名称，它可以运行相当于：

```python
from huggingface_hub import HfApi, Volume

api = HfApi()
space_id = "<your-namespace>/<your-space>"

# Attach the bucket at /data
api.set_space_volumes(
    space_id,
    volumes=[
        Volume(type="bucket", source="<your-namespace>/label-studio-data", mount_path="/data"),
    ],
)

# Tell Label Studio to write its SQLite DB and media into the mounted bucket
api.add_space_variable(space_id, "LABEL_STUDIO_BASE_DATA_DIR", "/data")
api.add_space_variable(space_id, "STORAGE_PERSISTENCE", "1")

# Optional: set a stable SECRET_KEY so sessions survive restarts
api.add_space_secret(space_id, "SECRET_KEY", "<random-string>")

# Factory rebuild so the new mount and variables take effect
api.restart_space(space_id, factory_reboot=True)
```

有关通过 `huggingface_hub` 管理空间和卷安装的更多信息，请参阅 [⟦T11⟧ guide](/docs/huggingface_hub/guides/manage-spaces)。

### 使用 Postgres 启用持久性

对于较重的多用户部署，您可以通过以下方式启用持久性
[connecting an external Postgres database to your
space](https://labelstud.io/guide/storedata.html#PostgreSQL-database),
确保保留所有项目和注释设置。设置以下秘密变量以匹配您自己的托管实例
Postgres。我们强烈建议将这些设置为秘密以防止泄露
有关您空间中向公众提供的数据库服务的信息
定义。

* `DJANGO_DB`：将其设置为`default`。

* `POSTGRE_NAME`：将其设置为 Postgres 数据库的名称。

* `POSTGRE_USER`：将其设置为 Postgres 用户名。

* `POSTGRE_PASSWORD`：将此设置为您的 Postgres 用户的密码。

* `POSTGRE_HOST`：将其设置为 Postgres 数据库运行的主机
   上。

* `POSTGRE_PORT`：将其设置为 Postgres 数据库正在运行的端口
  上。

* `STORAGE_PERSISTENCE`：将其设置为`1`以删除有关短暂的警告
  存储。

重新启动空间以应用这些设置。有关用户、项目的信息，
注释将存储在数据库中，并由 Label 重新加载
Studio 如果空间重新启动或重置。

### 启用云存储

默认情况下，为此空间启用的唯一数据存储是本地的。在这种情况下
空间重置后，所有数据都将丢失。要启用永久存储，您必须
启用[cloud storage connector](https://labelstud.io/guide/storage.html)。
选择适当的云连接器并为其配置机密。

#### 亚马逊 S3

* `STORAGE_TYPE`：将其设置为`s3`。

* `STORAGE_AWS_ACCESS_KEY_ID`: `<YOUR_ACCESS_KEY_ID>`* `STORAGE_AWS_SECRET_ACCESS_KEY`: `<YOUR_SECRET_ACCESS_KEY>`

* `STORAGE_AWS_BUCKET_NAME`: `<YOUR_BUCKET_NAME>`

* `STORAGE_AWS_REGION_NAME`: `<YOUR_BUCKET_REGION>`

* `STORAGE_AWS_FOLDER`：将其设置为空字符串。

#### 谷歌云存储

* `STORAGE_TYPE`：将其设置为`gcs`。

* `STORAGE_GCS_BUCKET_NAME`: `<YOUR_BUCKET_NAME>`

* `STORAGE_GCS_PROJECT_ID`: `<YOUR_PROJECT_ID>`

* `STORAGE_GCS_FOLDER`：将其设置为空字符串。

* `GOOGLE_APPLICATION_CREDENTIALS`：将其设置为`/opt/heartex/secrets/key.json`。

#### Azure Blob 存储

* `STORAGE_TYPE`：将其设置为`azure`。

* `STORAGE_AZURE_ACCOUNT_NAME`: `<YOUR_STORAGE_ACCOUNT>`

* `STORAGE_AZURE_ACCOUNT_KEY`: `<YOUR_STORAGE_KEY>`

* `STORAGE_AZURE_CONTAINER_NAME`: `<YOUR_CONTAINER_NAME>`

* `STORAGE_AZURE_FOLDER`：将其设置为空字符串。

## 🤗 后续步骤、反馈和支持

要开始使用 Label Studio，请查看 Label Studio ["Zero to One"
tutorial](https://labelstud.io/blog/introduction-to-label-studio-in-hugging-face-spaces/)，
它引导您完成情感分析注释项目示例。你
可以找到有关 Label Studio 和 Label Studio 的全套资源
社区在[Label Studio Home Page](https://labelstud.io)。这个
包括[full documentation](https://labelstud.io/guide/)，一个[interactive
playground](https://labelstud.io/playground/)，用于尝试不同的
注释接口，以及加入[Label Studio Slack
Community](https://slack.labelstudio.heartex.com/?source=spaces)的链接。

### 空间概述
https://huggingface.co/docs/hub/spaces-overview.md