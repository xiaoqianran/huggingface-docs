<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 空间上的磁盘使用情况

每个空间都附带少量磁盘存储空间。该磁盘空间是短暂的，这意味着如果您的空间重新启动或停止，其内容将会丢失。
如果您需要保存比空间本身更长生命周期的数据，您可以将一个或多个 [Storage Buckets](./storage-buckets) 作为卷附加。

## 附卷

[Storage Buckets](./storage-buckets) 是在空间中保存数据的推荐方法。附加的存储桶将按照您指定的路径安装到 Space 容器中，使其内容在运行时可作为本地文件使用。

创建空间时，可以从空间设置 UI 或通过 [⟦T0⟧](/docs/huggingface_hub/guides/manage-spaces#mount-volumes-in-your-space) Python API 以编程方式附加存储桶。它们可以以读写（默认）或只读方式安装。

有关创建和使用存储桶的完整详细信息，请参阅[Storage Buckets documentation](./storage-buckets)。

### 查看附加卷

空间页面在操作下拉列表中显示附加的卷。每个卷都会显示其源存储桶、容器内的挂载路径以及是否以只读或读写方式挂载。

## 安装模型、数据集和其他空间

模型、数据集和其他空间可以通过 [⟦T1⟧](/docs/huggingface_hub/guides/manage-spaces#mount-volumes-in-your-space) Python API 作为卷附加。它们始终以只读方式安装。连接后，存储库卷将显示在存储桶旁边的空间操作下拉列表中，并且可以从 UI 中查看或卸载。

当卷引用私有存储库时，无访问权限的用户仍将看到列出的卷（及其安装路径和访问模式），但源将被屏蔽为带有“（私有）”标签的`****/******`。

### 在拥抱脸部时使用 OpenCLIP
https://huggingface.co/docs/hub/open_clip.md