<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 管理集合

查看 [HfApi](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi) 文档页面，获取管理 Hub 上的空间的方法参考。

- 获取收藏内容：[get_collection()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.get_collection)
- 创建新集合：[create_collection()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_collection)
- 更新收藏：[update_collection_metadata()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.update_collection_metadata)
- 更新集合的资源组：[update_collection_resource_group()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.update_collection_resource_group)
- 删除收藏：[delete_collection()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.delete_collection)
- 将项目添加到集合中：[add_collection_item()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.add_collection_item)
- 更新集合中的项目：[update_collection_item()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.update_collection_item)
- 从集合中删除项目：[delete_collection_item()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.delete_collection_item)

### 集合[[huggingface_hub.Collection]]

#### Huggingface_hub.Collection[[huggingface_hub.Collection]]

```python
huggingface_hub.Collection(**kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L1491)

**参数：**

slug (`str`)：集合中的slug。例如。 `"TheBloke/recent-models-64f9a55bb3115b4f513ec026"`。

title (`str`) ：集合的标题。例如。 `"Recent models"`。

所有者 (`str`) ：集合的所有者。例如。 `"TheBloke"`。

items (`list[CollectionItem]`) ：集合中的项目列表。

last_updated (`datetime`) ：集合上次更新的日期。

位置 (`int`) ：集合在所有者集合列表中的位置。

private (`bool`) : 集合是否是私有的。

theme (`str`) ：集合的主题。例如。 `"green"`。

upvotes (`int`) : 集合的点赞数。

描述（`str`，*可选*）：集合的描述，作为纯文本。url (`str`) ：（属性）Hub 上集合的 URL。

包含有关 Hub 上集合的信息。

### CollectionItem[[huggingface_hub.CollectionItem]]

#### Huggingface_hub.CollectionItem[[huggingface_hub.CollectionItem]]

```python
huggingface_hub.CollectionItem(_id: str, id: str, type: CollectionItemType_T, position: int, note: dict | None = None, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L1443)

**参数：**

item_object_id (`str`) ：集合中项目的唯一 ID。

item_id (`str`) ：Hub 上底层对象的 ID。可以是 repo_id、论文 id、集合 slug 或存储桶 id。例如`"jbilcke-hf/ai-comic-factory"`、`"2307.09288"`、`"celinah/cerebras-function-calling-682607169c35fbfa98b30b9a"`。

item_type (`str`) ：底层对象的类型。可以是 `"model"`、`"dataset"`、`"space"`、`"paper"`、`"collection"` 或 `"bucket"` 之一。

位置 (`int`) ：集合中项目的位置。

note (`str`, *可选*) ：与项目关联的注释，作为纯文本。

包含有关集合（模型、数据集、空间、论文、集合或存储桶）的项目的信息。

### 推理类型
https://huggingface.co/docs/huggingface_hub/v1.27.0/package_reference/inference_types.md