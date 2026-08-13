<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 合并数据集并导出

在本节中，我们将演示如何组合两个数据集并导出结果。第一个数据集为 CSV 格式，第二个数据集为 Parquet 格式。让我们首先检查我们的数据集：

第一个是[TheFusion21/PokemonCards](https://huggingface.co/datasets/TheFusion21/PokemonCards)：

```bash
FROM 'hf://datasets/TheFusion21/PokemonCards/train.csv' LIMIT 3;
┌─────────┬──────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬────────────┬───────┬─────────────────┐
│   id    │      image_url       │                                                                 caption                                                                 │    name    │  hp   │    set_name     │
│ varchar │       varchar        │                                                                 varchar                                                                 │  varchar   │ int64 │     varchar     │
├─────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────┼───────┼─────────────────┤
│ pl3-1   │ https://images.pok…  │ A Basic, SP Pokemon Card of type Darkness with the title Absol G and 70 HP of rarity Rare Holo from the set Supreme Victors.  It has …  │ Absol G    │    70 │ Supreme Victors │
│ ex12-1  │ https://images.pok…  │ A Stage 1 Pokemon Card of type Colorless with the title Aerodactyl and 70 HP of rarity Rare Holo evolved from Mysterious Fossil from …  │ Aerodactyl │    70 │ Legend Maker    │
│ xy5-1   │ https://images.pok…  │ A Basic Pokemon Card of type Grass with the title Weedle and 50 HP of rarity Common from the set Primal Clash and the flavor text: It…  │ Weedle     │    50 │ Primal Clash    │
└─────────┴──────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴────────────┴───────┴─────────────────┘
```

第二个是[wanghaofan/pokemon-wiki-captions](https://huggingface.co/datasets/wanghaofan/pokemon-wiki-captions)：

```bash
FROM 'hf://datasets/wanghaofan/pokemon-wiki-captions/data/*.parquet' LIMIT 3;

┌──────────────────────┬───────────┬──────────┬──────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────┐
│        image         │  name_en  │ name_zh  │                           text_en                            │                                              text_zh                                               │
│ struct(bytes blob,…  │  varchar  │ varchar  │                           varchar                            │                                              varchar                                               │
├──────────────────────┼───────────┼──────────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ {'bytes': \x89PNG\…  │ abomasnow │ 暴雪王   │ Grass attributes,Blizzard King standing on two feet, with …  │ 草属性，双脚站立的暴雪王，全身白色的绒毛，淡紫色的眼睛，几缕长条装的毛皮盖着它的嘴巴               │
│ {'bytes': \x89PNG\…  │ abra      │ 凯西     │ Super power attributes, the whole body is yellow, the head…  │ 超能力属性，通体黄色，头部外形类似狐狸，尖尖鼻子，手和脚上都有三个指头，长尾巴末端带着一个褐色圆环 │
│ {'bytes': \x89PNG\…  │ absol     │ 阿勃梭鲁 │ Evil attribute, with white hair, blue-gray part without ha…  │ 恶属性，有白色毛发，没毛发的部分是蓝灰色，头右边类似弓的角，红色眼睛                               │
└──────────────────────┴───────────┴──────────┴──────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────┘

```

现在，让我们尝试通过加入 `name` 列来组合这两个数据集：

```bash
SELECT a.image_url
        , a.caption AS card_caption
        , a.name
        , a.hp
        , b.text_en as wiki_caption 
FROM 'hf://datasets/TheFusion21/PokemonCards/train.csv' a 
JOIN 'hf://datasets/wanghaofan/pokemon-wiki-captions/data/*.parquet' b 
ON LOWER(a.name) = b.name_en
LIMIT 3;

┌──────────────────────┬──────────────────────┬────────────┬───────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│      image_url       │     card_caption     │    name    │  hp   │                                                                 wiki_caption                                                                 │
│       varchar        │       varchar        │  varchar   │ int64 │                                                                   varchar                                                                    │
├──────────────────────┼──────────────────────┼────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ https://images.pok…  │ A Stage 1 Pokemon …  │ Aerodactyl │    70 │ A Pokémon with rock attributes, gray body, blue pupils, purple inner wings, two sharp claws on the wings, jagged teeth, and an arrow-like …  │
│ https://images.pok…  │ A Basic Pokemon Ca…  │ Weedle     │    50 │ Insect-like, caterpillar-like in appearance, with a khaki-yellow body, seven pairs of pink gastropods, a pink nose, a sharp poisonous need…  │
│ https://images.pok…  │ A Basic Pokemon Ca…  │ Caterpie   │    50 │ Insect attributes, caterpillar appearance, green back, white abdomen, Y-shaped red antennae on the head, yellow spindle-shaped tail, two p…  │
└──────────────────────┴──────────────────────┴────────────┴───────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

```

我们可以使用 `COPY` 命令将结果导出到 Parquet 文件：

```bash
COPY (SELECT a.image_url
        , a.caption AS card_caption
        , a.name
        , a.hp
        , b.text_en as wiki_caption 
FROM 'hf://datasets/TheFusion21/PokemonCards/train.csv' a 
JOIN 'hf://datasets/wanghaofan/pokemon-wiki-captions/data/*.parquet' b 
ON LOWER(a.name) = b.name_en) 
TO 'output.parquet' (FORMAT PARQUET);
```

让我们验证新的 Parquet 文件：

```bash
SELECT COUNT(*) FROM 'output.parquet';

┌──────────────┐
│ count_star() │
│    int64     │
├──────────────┤
│         9460 │
└──────────────┘

```

> [!提示]
> 您还可以导出为 [CSV](https://duckdb.org/docs/guides/file_formats/csv_export)、[Excel](https://duckdb.org/docs/guides/file_formats/excel_export
> ) 和 [JSON](https://duckdb.org/docs/guides/file_formats/json_export
> ) 格式。

最后，我们将生成的数据集推送到 Hub。您可以使用 Hub UI、`huggingface_hub` 客户端库等来上传 Parquet 文件，请参阅更多信息 [here](./datasets-adding)。

就是这样！您已成功合并两个数据集、导出结果并将其上传到 Hugging Face Hub。

### 计费
https://huggingface.co/docs/hub/billing.md