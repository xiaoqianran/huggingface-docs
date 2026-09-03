<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 收藏

集合是 Hub 上的一组相关项目（模型、数据集、空间、论文、集合、存储桶），它们在同一页面上组织在一起。集合对于创建您自己的投资组合、为类别中的内容添加书签或呈现您想要共享的项目的精选列表非常有用。查看此[guide](https://huggingface.co/docs/hub/collections)，更详细地了解集合是什么以及它们在 Hub 上的外观。

您可以直接在浏览器中管理集合，但在本指南中，我们将重点介绍如何以编程方式管理它们。

## 获取集合

使用 [get_collection()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.get_collection) 获取您的收藏或任何公共收藏。您必须拥有集合的 *slug* 才能检索集合。 slug 是基于标题和唯一 ID 的集合的标识符。您可以在集合页面的 URL 中找到该 slug。

    

让我们使用 `"TheBloke/recent-models-64f9a55bb3115b4f513ec026"` 获取集合：

```py
>>> from huggingface_hub import get_collection
>>> collection = get_collection("TheBloke/recent-models-64f9a55bb3115b4f513ec026")
>>> collection
Collection(
  slug='TheBloke/recent-models-64f9a55bb3115b4f513ec026',
  title='Recent models',
  owner='TheBloke',
  items=[...],
  last_updated=datetime.datetime(2023, 10, 2, 22, 56, 48, 632000, tzinfo=datetime.timezone.utc),
  position=1,
  private=False,
  theme='green',
  upvotes=90,
  description="Models I've recently quantized. Please note that currently this list has to be updated manually, and therefore is not guaranteed to be up-to-date."
)
>>> collection.items[0]
CollectionItem(
  item_object_id='651446103cd773a050bf64c2',
  item_id='TheBloke/U-Amethyst-20B-AWQ',
  item_type='model',
  position=88,
  note=None
)
```

[get_collection()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.get_collection)返回的[Collection](/docs/huggingface_hub/v1.30.0/en/package_reference/collections#huggingface_hub.Collection)对象包含：
- 高级元数据：`slug`、`owner`、`title`、`description`等。
- [CollectionItem](/docs/huggingface_hub/v1.30.0/en/package_reference/collections#huggingface_hub.CollectionItem) 对象列表；每个项目代表一个模型、一个数据集、一个空间、一篇论文、一个集合或一个存储桶。所有收藏品保证具有：
- 唯一的`item_object_id`：这是数据库中集合项的id
- `item_id`：这是底层项目（模型、数据集、空间、论文、集合、存储桶）在 Hub 上的 id；它不一定是唯一的，只有`item_id`/`item_type`对是唯一的
- `item_type`：模型、数据集、空间、论文、集合、桶
- 集合中项目的`position`，可以更新以重新组织您的集合（请参阅下面的[update_collection_item()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.update_collection_item)）

`note` 也可以附加到该物品上。这对于添加有关该项目的附加信息（评论、博客文章的链接等）非常有用。如果项目没有注释，该属性仍然具有 `None` 值。

除了这些基本属性之外，返回的项目还可以根据其类型具有其他属性：`author`、`private`、`lastModified`、`gated`、`title`、`likes`、`upvotes` 等。不保证返回这些属性。

## 列出集合

我们还可以使用[list_collections()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_collections)检索集合。可以使用一些参数来过滤集合。让我们列出用户[⟦T31⟧](https://huggingface.co/teknium)的所有收藏。
```py
>>> from huggingface_hub import list_collections

>>> collections = list_collections(owner="teknium")
```这将返回 `Collection` 对象的可迭代对象。例如，我们可以迭代它们来打印每个集合的赞成票数。

```py
>>> for collection in collections:
...   print("Number of upvotes:", collection.upvotes)
Number of upvotes: 1
Number of upvotes: 5
```

> [!警告]
> 列出集合时，每个集合的项目列表将被截断为最多 4 个项目。要从集合中检索所有项目，您必须使用[get_collection()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.get_collection)。

可以进行更高级的过滤。让我们获取包含模型 [TheBloke/OpenHermes-2.5-Mistral-7B-GGUF](https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF) 的所有集合，按趋势排序，并将计数限制为 5。
```py
>>> collections = list_collections(item="models/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF", sort="trending", limit=5):
>>> for collection in collections:
...   print(collection.slug)
teknium/quantized-models-6544690bb978e0b0f7328748
AmeerH/function-calling-65560a2565d7a6ef568527af
PostArchitekt/7bz-65479bb8c194936469697d8c
gnomealone/need-to-test-652007226c6ce4cdacf9c233
Crataco/favorite-7b-models-651944072b4fffcb41f8b568
```

参数`sort`必须是`"last_modified"`、`"trending"`或`"upvotes"`之一。参数`item`接受任何特定项目。例如：
* `"models/teknium/OpenHermes-2.5-Mistral-7B"`
* `"spaces/julien-c/open-gpt-rhyming-robot"`
* `"datasets/squad"`
* `"papers/2311.12983"`
* `"buckets/namespace/bucket-name"`

欲了解更多详情，请查看[list_collections()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_collections)参考资料。

## 创建一个新集合

现在我们知道如何获得 [Collection](/docs/huggingface_hub/v1.30.0/en/package_reference/collections#huggingface_hub.Collection)，让我们创建自己的吧！使用 [create_collection()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_collection) 以及标题和描述。要在组织页面上创建集合，请在创建集合时传递 `namespace="my-cool-org"`。最后，您还可以通过传递`private=True`来创建私有集合。

```py
>>> from huggingface_hub import create_collection

>>> collection = create_collection(
...     title="ICCV 2023",
...     description="Portfolio of models, papers and demos I presented at ICCV 2023",
... )
```

它将返回一个 [Collection](/docs/huggingface_hub/v1.30.0/en/package_reference/collections#huggingface_hub.Collection) 对象，其中包含高级元数据（标题、描述、所有者等）和一个空的项目列表。您现在可以使用其 `slug` 来引用该集合。

```py
>>> collection.slug
'owner/iccv-2023-15e23b46cb98efca45'
>>> collection.title
"ICCV 2023"
>>> collection.owner
"username"
>>> collection.url
'https://huggingface.co/collections/owner/iccv-2023-15e23b46cb98efca45'
```

## 管理集合中的项目现在我们有了一个 [Collection](/docs/huggingface_hub/v1.30.0/en/package_reference/collections#huggingface_hub.Collection)，我们想要向其中添加项目并组织它们。

### 添加项目

必须使用[add_collection_item()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.add_collection_item)一项一项地添加项目。您只需要知道`collection_slug`、`item_id`和`item_type`即可。或者，您还可以向项目添加 `note`（最多 500 个字符）。

```py
>>> from huggingface_hub import create_collection, add_collection_item

>>> collection = create_collection(title="OS Week Highlights - Sept 18 - 24", namespace="osanseviero")
>>> collection.slug
"osanseviero/os-week-highlights-sept-18-24-650bfed7f795a59f491afb80"

>>> add_collection_item(collection.slug, item_id="coqui/xtts", item_type="space")
>>> add_collection_item(
...     collection.slug,
...     item_id="warp-ai/wuerstchen",
...     item_type="model",
...     note="Würstchen is a new fast and efficient high resolution text-to-image architecture and model"
... )
>>> add_collection_item(collection.slug, item_id="lmsys/lmsys-chat-1m", item_type="dataset")
>>> add_collection_item(collection.slug, item_id="warp-ai/wuerstchen", item_type="space") # same item_id, different item_type
>>> add_collection_item(collection.slug, item_id="namespace/my-bucket", item_type="bucket")
```

如果集合中已存在某个项目（相同的 `item_id`/`item_type` 对），则会引发 HTTP 409 错误。您可以通过设置`exists_ok=True`来选择忽略此错误。

### 为现有项目添加注释

您可以使用 [update_collection_item()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.update_collection_item) 修改现有项目以添加或修改附加到其的注释。让我们重用上面的例子：

```py
>>> from huggingface_hub import get_collection, update_collection_item

# Fetch collection with newly added items
>>> collection_slug = "osanseviero/os-week-highlights-sept-18-24-650bfed7f795a59f491afb80"
>>> collection = get_collection(collection_slug)

# Add note the `lmsys-chat-1m` dataset
>>> update_collection_item(
...     collection_slug=collection_slug,
...     item_object_id=collection.items[2].item_object_id,
...     note="This dataset contains one million real-world conversations with 25 state-of-the-art LLMs.",
... )
```

### 重新订购商品

集合中的项目是有序的。顺序由每个项目的 `position` 属性决定。默认情况下，项目通过在集合末尾附加新项目来排序。您可以使用 [update_collection_item()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.update_collection_item) 更新订单，就像添加注释一样。

让我们重用上面的例子：

```py
>>> from huggingface_hub import get_collection, update_collection_item

# Fetch collection
>>> collection_slug = "osanseviero/os-week-highlights-sept-18-24-650bfed7f795a59f491afb80"
>>> collection = get_collection(collection_slug)

# Reorder to place the two `Wuerstchen` items together
>>> update_collection_item(
...     collection_slug=collection_slug,
...     item_object_id=collection.items[3].item_object_id,
...     position=2,
... )
```

### 删除项目

最后，您还可以使用 [delete_collection_item()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.delete_collection_item) 删除项目。

```py
>>> from huggingface_hub import get_collection, update_collection_item

# Fetch collection
>>> collection_slug = "osanseviero/os-week-highlights-sept-18-24-650bfed7f795a59f491afb80"
>>> collection = get_collection(collection_slug)

# Remove `coqui/xtts` Space from the list
>>> delete_collection_item(collection_slug=collection_slug, item_object_id=collection.items[0].item_object_id)
```

## 删除集合

可以使用[delete_collection()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.delete_collection)删除集合。

> [!警告]
> 这是不可恢复的操作。已删除的集合无法恢复。

```py
>>> from huggingface_hub import delete_collection
>>> collection = delete_collection("username/useless-collection-64f9a55bb3115b4f513ec026", missing_ok=True)
```### 创建和管理存储库
https://huggingface.co/docs/huggingface_hub/v1.30.0/guides/repository.md