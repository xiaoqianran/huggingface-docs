# List Parquet files

Datasets can be published in any format (CSV, JSONL, directories of images, etc.) to the Hub, and they are easily accessed with the 🤗 [Datasets](https://huggingface.co/docs/datasets/) library. For a more performant experience (especially when it comes to large datasets), the dataset viewer automatically converts every dataset to the [Parquet](https://parquet.apache.org/) format.

## What is Parquet?

Parquet is a columnar storage format optimized for querying and processing large datasets. Parquet is a popular choice for big data processing and analytics and is widely used for data processing and machine learning.

In Parquet, data is divided into chunks called "row groups", and within each row group, it is stored in columns rather than rows. Each row group column is compressed separately using the best compression algorithm depending on the data, and contains metadata and statistics (min/max value, number of NULL values) about the data it contains.

This structure allows for efficient data reading and querying:
- only the necessary columns are read from disk (projection pushdown); no need to read the entire file. This reduces the memory requirement for working with Parquet data. 
- entire row groups are skipped if the statistics stored in its metadata do not match the data of interest (automatic filtering)
- the data is compressed, which reduces the amount of data that needs to be stored and transferred.

A Parquet file contains a single table. If a dataset has multiple tables (e.g. multiple splits or subsets), each table is stored in a separate Parquet file.

## Conversion to Parquet

The Parquet files are published to the Hub on a specific `refs/convert/parquet` branch (like this `fancyzhx/amazon_polarity` [branch](https://huggingface.co/datasets/fancyzhx/amazon_polarity/tree/refs%2Fconvert%2Fparquet) for example) that parallels the `main` branch.

> [!TIP]
> In order for the dataset viewer to generate a Parquet version of a dataset, the dataset must be _public_, or owned by a [PRO user](https://huggingface.co/pricing) or an [Enterprise Hub organization](https://huggingface.co/enterprise).

## Using the dataset viewer API

This guide shows you how to use the dataset viewer's `/parquet` endpoint to retrieve a list of a dataset's files converted to Parquet. Feel free to also try it out with [Postman](https://www.postman.com/huggingface/workspace/hugging-face-apis/request/23242779-f0cde3b9-c2ee-4062-aaca-65c4cfdd96f8), [RapidAPI](https://rapidapi.com/hugging-face-hugging-face-default/api/hugging-face-datasets-api), or [ReDoc](https://redocly.github.io/redoc/?url=https://datasets-server.huggingface.co/openapi.json#operation/listSplits).

The `/parquet` endpoint accepts the dataset name as its query parameter:

<inferencesnippet> <python>
```python
import requests
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://datasets-server.huggingface.co/parquet?dataset=ibm/duorc"
def query():
    response = requests.get(API_URL, headers=headers)
    return response.json()
data = query()
```
</python> <js>
```js
import fetch from "node-fetch";
async function query(data) {
    const response = await fetch(
        "https://datasets-server.huggingface.co/parquet?dataset=ibm/duorc",
        {
            headers: { Authorization: `Bearer ${API_TOKEN}` },
            method: "GET"
        }
    );
    const result = await response.json();
    return result;
}
query().then((response) => {
    console.log(JSON.stringify(response));
});
```
</js> <curl>
```curl
curl https://datasets-server.huggingface.co/parquet?dataset=ibm/duorc \
        -X GET \
        -H "Authorization: Bearer ${API_TOKEN}"
```
</curl> </inferencesnippet>

The endpoint response is a JSON containing a list of the dataset's files in the Parquet format. For example, the [`ibm/duorc`](https://huggingface.co/datasets/ibm/duorc) dataset has six Parquet files, which corresponds to the `test`, `train` and `validation` splits of its two subsets, `ParaphraseRC` and `SelfRC` (see the [List splits and subsets](./splits) guide for more details about splits and subsets).

The endpoint also gives the filename and size of each file:

```json
{
   "parquet_files":[
      {
         "dataset":"ibm/duorc",
         "config":"ParaphraseRC",
         "split":"test",
         "url":"https://huggingface.co/datasets/ibm/duorc/resolve/refs%2Fconvert%2Fparquet/ParaphraseRC/test/0000.parquet",
         "filename":"0000.parquet",
         "size":6136591
      },
      {
         "dataset":"ibm/duorc",
         "config":"ParaphraseRC",
         "split":"train",
         "url":"https://huggingface.co/datasets/ibm/duorc/resolve/refs%2Fconvert%2Fparquet/ParaphraseRC/train/0000.parquet",
         "filename":"0000.parquet",
         "size":26005668
      },
      {
         "dataset":"ibm/duorc",
         "config":"ParaphraseRC",
         "split":"validation",
         "url":"https://huggingface.co/datasets/ibm/duorc/resolve/refs%2Fconvert%2Fparquet/ParaphraseRC/validation/0000.parquet",
         "filename":"0000.parquet",
         "size":5566868
      },
      {
         "dataset":"ibm/duorc",
         "config":"SelfRC",
         "split":"test",
         "url":"https://huggingface.co/datasets/ibm/duorc/resolve/refs%2Fconvert%2Fparquet/SelfRC/test/0000.parquet",
         "filename":"0000.parquet",
         "size":3035736
      },
      {
         "dataset":"ibm/duorc",
         "config":"SelfRC",
         "split":"train",
         "url":"https://huggingface.co/datasets/ibm/duorc/resolve/refs%2Fconvert%2Fparquet/SelfRC/train/0000.parquet",
         "filename":"0000.parquet",
         "size":14851720
      },
      {
         "dataset":"ibm/duorc",
         "config":"SelfRC",
         "split":"validation",
         "url":"https://huggingface.co/datasets/ibm/duorc/resolve/refs%2Fconvert%2Fparquet/SelfRC/validation/0000.parquet",
         "filename":"0000.parquet",
         "size":3114390
      }
   ],
   "pending":[
      
   ],
   "failed":[
      
   ],
   "partial":false
}
```

## Sharded Parquet files

Big datasets are partitioned into Parquet files (shards) of about 500MB each. The filename contains the name of the dataset, the split, the shard index, and the total number of shards (`dataset-name-train-0000-of-0004.parquet`). For a given split, the elements in the list are sorted by their shard index, in ascending order. For example, the `train` split of the [`fancyzhx/amazon_polarity`](https://datasets-server.huggingface.co/parquet?dataset=fancyzhx/amazon_polarity) dataset is partitioned into 4 shards:

```json
{
   "parquet_files":[
      {
         "dataset":"fancyzhx/amazon_polarity",
         "config":"amazon_polarity",
         "split":"test",
         "url":"https://huggingface.co/datasets/fancyzhx/amazon_polarity/resolve/refs%2Fconvert%2Fparquet/amazon_polarity/test/0000.parquet",
         "filename":"0000.parquet",
         "size":117422360
      },
      {
         "dataset":"fancyzhx/amazon_polarity",
         "config":"amazon_polarity",
         "split":"train",
         "url":"https://huggingface.co/datasets/fancyzhx/amazon_polarity/resolve/refs%2Fconvert%2Fparquet/amazon_polarity/train/0000.parquet",
         "filename":"0000.parquet",
         "size":259761770
      },
      {
         "dataset":"fancyzhx/amazon_polarity",
         "config":"amazon_polarity",
         "split":"train",
         "url":"https://huggingface.co/datasets/fancyzhx/amazon_polarity/resolve/refs%2Fconvert%2Fparquet/amazon_polarity/train/0001.parquet",
         "filename":"0001.parquet",
         "size":258363554
      },
      {
         "dataset":"fancyzhx/amazon_polarity",
         "config":"amazon_polarity",
         "split":"train",
         "url":"https://huggingface.co/datasets/fancyzhx/amazon_polarity/resolve/refs%2Fconvert%2Fparquet/amazon_polarity/train/0002.parquet",
         "filename":"0002.parquet",
         "size":255471883
      },
      {
         "dataset":"fancyzhx/amazon_polarity",
         "config":"amazon_polarity",
         "split":"train",
         "url":"https://huggingface.co/datasets/fancyzhx/amazon_polarity/resolve/refs%2Fconvert%2Fparquet/amazon_polarity/train/0003.parquet",
         "filename":"0003.parquet",
         "size":254410930
      }
   ],
   "pending":[
      
   ],
   "failed":[
      
   ],
   "partial":false
}
```

To read and query the Parquet files, take a look at the [Query datasets from the dataset viewer API](parquet_process) guide.

## Partially converted datasets

The Parquet version can be partial in two cases:
- if the dataset is already in Parquet format but it contains row groups bigger than the recommended size (100-300MB uncompressed). This size is better for memory usage since Parquet is streamed row group per row group in most data libraries.
- if the dataset is not already in Parquet format or if it is bigger than 5GB.

In that case the Parquet files are generated up to 5GB and placed in a split directory prefixed with "partial", e.g. "partial-train" instead of "train".

You can check the row groups size directly on Hugging Face using the Parquet metadata sidebar, for example [here](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/data/CC-MAIN-2013-20?show_file_info=data%2FCC-MAIN-2013-20%2Ftrain-00000-of-00014.parquet):

![clic-parquet-metadata-sidebar](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets-server/clic-parquet-metadata-sidebar.png)

![parquet-metadata-sidebar-total-byte-size](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets-server/parquet-metadata-sidebar-total-byte-size.png)

## Parquet-native datasets

When the dataset is already in Parquet format, the data are not converted and the files in `refs/convert/parquet` are links to the original files. This rule suffers an exception to ensure the dataset viewer API to stay fast: if the [row group](https://parquet.apache.org/docs/concepts/) size of the original Parquet files is too big, new Parquet files are generated.

## Using the Hugging Face Hub API

For convenience, you can directly use the Hugging Face Hub `/api/parquet` endpoint which returns the list of Parquet URLs:

<inferencesnippet> <python>
```python
import requests
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://huggingface.co/api/datasets/ibm/duorc/parquet"
def query():
    response = requests.get(API_URL, headers=headers)
    return response.json()
urls = query()
```
</python> <js>
```js
import fetch from "node-fetch";
async function query(data) {
    const response = await fetch(
        "https://huggingface.co/api/datasets/ibm/duorc/parquet",
        {
            headers: { Authorization: `Bearer ${API_TOKEN}` },
            method: "GET"
        }
    );
    const urls = await response.json();
    return urls;
}
query().then((response) => {
    console.log(JSON.stringify(response));
});
```
</js> <curl>
```curl
curl https://huggingface.co/api/datasets/ibm/duorc/parquet \
        -X GET \
        -H "Authorization: Bearer ${API_TOKEN}"
```
</curl> </inferencesnippet>

The endpoint response is a JSON containing a list of the dataset's files URLs in the Parquet format for each split and subset. For example, the [`ibm/duorc`](https://huggingface.co/datasets/ibm/duorc) dataset has one Parquet file for the train split of the "ParaphraseRC" subset (see the [List splits and subsets](./splits) guide for more details about splits and subsets).

```json
{
   "ParaphraseRC":{
      "test":[
         "https://huggingface.co/api/datasets/ibm/duorc/parquet/ParaphraseRC/test/0.parquet"
      ],
      "train":[
         "https://huggingface.co/api/datasets/ibm/duorc/parquet/ParaphraseRC/train/0.parquet"
      ],
      "validation":[
         "https://huggingface.co/api/datasets/ibm/duorc/parquet/ParaphraseRC/validation/0.parquet"
      ]
   },
   "SelfRC":{
      "test":[
         "https://huggingface.co/api/datasets/ibm/duorc/parquet/SelfRC/test/0.parquet"
      ],
      "train":[
         "https://huggingface.co/api/datasets/ibm/duorc/parquet/SelfRC/train/0.parquet"
      ],
      "validation":[
         "https://huggingface.co/api/datasets/ibm/duorc/parquet/SelfRC/validation/0.parquet"
      ]
   }
}
```

Optionally you can specify which subset name to return, as well as which split:

<inferencesnippet> <python>
```python
import requests
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://huggingface.co/api/datasets/ibm/duorc/parquet/ParaphraseRC/train"
def query():
    response = requests.get(API_URL, headers=headers)
    return response.json()
urls = query()
```
</python> <js>
```js
import fetch from "node-fetch";
async function query(data) {
    const response = await fetch(
        "https://huggingface.co/api/datasets/ibm/duorc/parquet/ParaphraseRC/train",
        {
            headers: { Authorization: `Bearer ${API_TOKEN}` },
            method: "GET"
        }
    );
    const urls = await response.json();
    return urls;
}
query().then((response) => {
    console.log(JSON.stringify(response));
});
```
</js> <curl>
```curl
curl https://huggingface.co/api/datasets/ibm/duorc/parquet/ParaphraseRC/train \
        -X GET \
        -H "Authorization: Bearer ${API_TOKEN}"
```
</curl> </inferencesnippet>

```json
[
  "https://huggingface.co/api/datasets/ibm/duorc/parquet/ParaphraseRC/train/0.parquet"
]
```

Each parquet file can also be accessed using its shard index: `https://huggingface.co/api/datasets/ibm/duorc/parquet/ParaphraseRC/train/0.parquet` redirects to `https://huggingface.co/datasets/ibm/duorc/resolve/refs%2Fconvert%2Fparquet/ParaphraseRC/train/0000.parquet` for example.

### Get the number of rows and the bytes size
https://huggingface.co/docs/dataset-viewer/size.md

# Get the number of rows and the size in bytes

This guide shows you how to use the dataset viewer's `/size` endpoint to retrieve a dataset's size programmatically. Feel free to also try it out with [ReDoc](https://redocly.github.io/redoc/?url=https://datasets-server.huggingface.co/openapi.json#operation/getSize).

The `/size` endpoint accepts the dataset name as its query parameter:

<inferencesnippet> <python>
```python
import requests
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://datasets-server.huggingface.co/size?dataset=ibm/duorc"
def query():
    response = requests.get(API_URL, headers=headers)
    return response.json()
data = query()
```
</python> <js>
```js
import fetch from "node-fetch";
async function query(data) {
    const response = await fetch(
        "https://datasets-server.huggingface.co/size?dataset=ibm/duorc",
        {
            headers: { Authorization: `Bearer ${API_TOKEN}` },
            method: "GET"
        }
    );
    const result = await response.json();
    return result;
}
query().then((response) => {
    console.log(JSON.stringify(response));
});
```
</js> <curl>
```curl
curl https://datasets-server.huggingface.co/size?dataset=ibm/duorc \
        -X GET \
        -H "Authorization: Bearer ${API_TOKEN}"
```
</curl> </inferencesnippet>

The endpoint response is a JSON containing the size of the dataset, as well as each of its subsets and splits. It provides the number of rows, the number of colums (where applicable) and the size in bytes for the different forms of the data: original files, size in memory (RAM) and auto-converted parquet files. For example, the [ibm/duorc](https://huggingface.co/datasets/ibm/duorc) dataset has 187.213 rows along all its subsets and splits, for a total of 97MB.

```json
{
   "size":{
      "dataset":{
         "dataset":"ibm/duorc",
         "num_bytes_original_files":58710973,
         "num_bytes_parquet_files":58710973,
         "num_bytes_memory":1060742354,
         "num_rows":187213
      },
      "configs":[
         {
            "dataset":"ibm/duorc",
            "config":"ParaphraseRC",
            "num_bytes_original_files":37709127,
            "num_bytes_parquet_files":37709127,
            "num_bytes_memory":704394283,
            "num_rows":100972,
            "num_columns":7
         },
         {
            "dataset":"ibm/duorc",
            "config":"SelfRC",
            "num_bytes_original_files":21001846,
            "num_bytes_parquet_files":21001846,
            "num_bytes_memory":356348071,
            "num_rows":86241,
            "num_columns":7
         }
      ],
      "splits":[
         {
            "dataset":"ibm/duorc",
            "config":"ParaphraseRC",
            "split":"train",
            "num_bytes_parquet_files":26005668,
            "num_bytes_memory":494389683,
            "num_rows":69524,
            "num_columns":7
         },
         {
            "dataset":"ibm/duorc",
            "config":"ParaphraseRC",
            "split":"validation",
            "num_bytes_parquet_files":5566868,
            "num_bytes_memory":106733319,
            "num_rows":15591,
            "num_columns":7
         },
         {
            "dataset":"ibm/duorc",
            "config":"ParaphraseRC",
            "split":"test",
            "num_bytes_parquet_files":6136591,
            "num_bytes_memory":103271281,
            "num_rows":15857,
            "num_columns":7
         },
         {
            "dataset":"ibm/duorc",
            "config":"SelfRC",
            "split":"train",
            "num_bytes_parquet_files":14851720,
            "num_bytes_memory":248966361,
            "num_rows":60721,
            "num_columns":7
         },
         {
            "dataset":"ibm/duorc",
            "config":"SelfRC",
            "split":"validation",
            "num_bytes_parquet_files":3114390,
            "num_bytes_memory":56359392,
            "num_rows":12961,
            "num_columns":7
         },
         {
            "dataset":"ibm/duorc",
            "config":"SelfRC",
            "split":"test",
            "num_bytes_parquet_files":3035736,
            "num_bytes_memory":51022318,
            "num_rows":12559,
            "num_columns":7
         }
      ]
   },
   "pending":[
      
   ],
   "failed":[
      
   ],
   "partial":false
}
```

If the size has `partial: true` it means that the actual size of the dataset couldn't been determined because it's too big.

In that case the number of rows and bytes can be inferior to the actual numbers.

### Explore dataset statistics
https://huggingface.co/docs/dataset-viewer/statistics.md

# Explore statistics over split data

The dataset viewer provides a `/statistics` endpoint for fetching some basic statistics precomputed for a requested dataset. This will get you a quick insight on how the data is distributed.

> [!WARNING]
> Currently, statistics are computed only for [datasets with Parquet exports](./parquet).

The `/statistics` endpoint requires three query parameters:

- `dataset`: the dataset name, for example `nyu-mll/glue`
- `config`: the subset name, for example `cola`
- `split`: the split name, for example `train`

Let's get some stats for `nyu-mll/glue` dataset, `cola` subset, `train` split:

<inferencesnippet> <python>
```python
import requests
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://datasets-server.huggingface.co/statistics?dataset=nyu-mll/glue&config=cola&split=train"
def query():
    response = requests.get(API_URL, headers=headers)
    return response.json()
data = query()
```
</python> <js>
```js
import fetch from "node-fetch";
async function query(data) {
    const response = await fetch(
        "https://datasets-server.huggingface.co/statistics?dataset=nyu-mll/glue&config=cola&split=train",
        {
            headers: { Authorization: `Bearer ${API_TOKEN}` },
            method: "GET"
        }
    );
    const result = await response.json();
    return result;
}
query().then((response) => {
    console.log(JSON.stringify(response));
});
```
</js> <curl>
```curl
curl https://datasets-server.huggingface.co/statistics?dataset=nyu-mll/glue&config=cola&split=train \
        -X GET \
        -H "Authorization: Bearer ${API_TOKEN}"
```
</curl> </inferencesnippet>

The response JSON contains three keys:
* `num_examples` - number of samples in a split or number of samples in the first chunk of data if dataset is larger than 5GB (see `partial` field below).
* `statistics` - list of dictionaries of statistics per each column, each dictionary has three keys: `column_name`, `column_type`, and `column_statistics`. Content of `column_statistics` depends on a column type, see [Response structure by data types](./statistics#response-structure-by-data-type) for more details
* `partial` - `true` if statistics are computed on the first 5 GB of data, not on the full split, `false` otherwise.

```json
{
  "num_examples": 8551,
  "statistics": [
    {
      "column_name": "idx",
      "column_type": "int",
      "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0,
        "min": 0,
        "max": 8550,
        "mean": 4275,
        "median": 4275,
        "std": 2468.60541,
        "histogram": {
          "hist": [
            856,
            856,
            856,
            856,
            856,
            856,
            856,
            856,
            856,
            847
          ],
          "bin_edges": [
            0,
            856,
            1712,
            2568,
            3424,
            4280,
            5136,
            5992,
            6848,
            7704,
            8550
          ]
        }
      }
    },
    {
      "column_name": "label",
      "column_type": "class_label",
      "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0,
        "no_label_count": 0,
        "no_label_proportion": 0,
        "n_unique": 2,
        "frequencies": {
          "unacceptable": 2528,
          "acceptable": 6023
        }
      }
    },
    {
      "column_name": "sentence",
      "column_type": "string_text",
      "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0,
        "min": 6,
        "max": 231,
        "mean": 40.70074,
        "median": 37,
        "std": 19.14431,
        "histogram": {
          "hist": [
            2260,
            4512,
            1262,
            380,
            102,
            26,
            6,
            1,
            1,
            1
          ],
          "bin_edges": [
            6,
            29,
            52,
            75,
            98,
            121,
            144,
            167,
            190,
            213,
            231
          ]
        }
      }
    }
  ],
  "partial": false
}
```

## Response structure by data type

Currently, statistics are supported for strings, float and integer numbers, lists, datetimes, audio and image data and the special [`datasets.ClassLabel`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.ClassLabel) feature type of the [`datasets`](https://huggingface.co/docs/datasets/) library.

`column_type` in response can be one of the following values:

* `class_label` - for [`datasets.ClassLabel`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.ClassLabel) feature which represents categorical data
* `float` - for float data types
* `int` - for integer data types
* `bool` - for boolean data type
* `string_label` - for string data types being treated as categories (see below)
* `string_text` - for string data types if they do not represent categories (see below)
* `list` - for lists of any other data types (including lists)
* `audio` - for audio data
* `image` - for image data
* `datetime` - for datetime data 

### `class_label`

This type represents categorical data encoded as [`ClassLabel`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.ClassLabel) feature. The following measures are computed:

* number and proportion of `null` values
* number and proportion of values with no label
* number of unique values (excluding `null` and `no label`)
* value counts for each label (excluding `null` and `no label`)

**Example**

```json
{
  "column_name": "label",
  "column_type": "class_label",
  "column_statistics": {
    "nan_count": 0,
    "nan_proportion": 0,
    "no_label_count": 0,
    "no_label_proportion": 0,
    "n_unique": 2,
    "frequencies": {
      "unacceptable": 2528,
      "acceptable": 6023
    }
  }
}
```

### float

The following measures are returned for float data types:

* minimum, maximum, mean, median, and standard deviation values
* number and proportion of `null` and `NaN` values (`NaN` values are treated as `null`)
* histogram with 10 bins

**Example**

```json
{
  "column_name": "clarity",
  "column_type": "float",
  "column_statistics": {
    "nan_count": 0,
    "nan_proportion": 0,
    "min": 0,
    "max": 2,
    "mean": 1.67206,
    "median": 1.8,
    "std": 0.38714,
    "histogram": {
      "hist": [
        17,
        12,
        48,
        52,
        135,
        188,
        814,
        15,
        1628,
        2048
      ],
      "bin_edges": [
        0,
        0.2,
        0.4,
        0.6,
        0.8,
        1,
        1.2,
        1.4,
        1.6,
        1.8,
        2
      ]
    }
  }
}
```

### int

The following measures are returned for integer data types:

* minimum, maximum, mean, median, and standard deviation values
* number and proportion of `null` values
* histogram with less than or equal to 10 bins

**Example**

```json
{
    "column_name": "direction",
    "column_type": "int",
    "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0.0,
        "min": 0,
        "max": 1,
        "mean": 0.49925,
        "median": 0.0,
        "std": 0.5,
        "histogram": {
            "hist": [
                50075,
                49925
            ],
            "bin_edges": [
                0,
                1,
                1
            ]
        }
    }
}
```

### bool

The following measures are returned for bool data type:

* number and proportion of `null` values
* value counts for `'True'` and `'False'` values

**Example**

```json
{
  "column_name": "penalty",
  "column_type": "bool",
  "column_statistics":
    {
        "nan_count": 3,
        "nan_proportion": 0.15,
        "frequencies": {
            "False": 7,
            "True": 10
        }
    }
}
```

### string_label

If the proportion of unique values in a string column within requested split is lower than or equal to 0.2 and the number of unique values is lower than 1000, or if the number of unique values is lower or equal to 10 (independently of the proportion), it is considered to be a category. The following measures are returned:

* number and proportion of `null` values
* number of unique values (excluding `null`)
* value counts for each label (excluding `null`)

**Example**

```json
{
  "column_name": "answerKey",
  "column_type": "string_label",
  "column_statistics": {
    "nan_count": 0,
    "nan_proportion": 0,
    "n_unique": 4,
    "frequencies": {
      "D": 1221,
      "C": 1146,
      "A": 1378,
      "B": 1212
    }
  }
}

```

### string_text

If string column does not satisfy the conditions to be treated as a `string_label`, it is considered to be a column containing texts and response contains statistics over text lengths which are calculated by character number. The following measures are computed:

* minimum, maximum, mean, median, and standard deviation of text lengths
* number and proportion of `null` values
* histogram of text lengths with 10 bins

**Example**

```json
{
  "column_name": "sentence",
  "column_type": "string_text",
  "column_statistics": {
    "nan_count": 0,
    "nan_proportion": 0,
    "min": 6,
    "max": 231,
    "mean": 40.70074,
    "median": 37,
    "std": 19.14431,
    "histogram": {
      "hist": [
        2260,
        4512,
        1262,
        380,
        102,
        26,
        6,
        1,
        1,
        1
      ],
      "bin_edges": [
        6,
        29,
        52,
        75,
        98,
        121,
        144,
        167,
        190,
        213,
        231
      ]
    }
  }
}
```

### list

For lists, the distribution of their lengths is computed. The following measures are returned:

* minimum, maximum, mean, median, and standard deviation of lists lengths
* number and proportion of `null` values
* histogram of lists lengths with up to 10 bins

**Example**

```json
{
    "column_name": "chat_history",
    "column_type": "list",
    "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0.0,
        "min": 1,
        "max": 3,
        "mean": 1.01741,
        "median": 1.0,
        "std": 0.13146,
        "histogram": {
            "hist": [
                11177,
                196,
                1
            ],
            "bin_edges": [
                1,
                2,
                3,
                3
            ]
        }
    }
}
```

Note that dictionaries of lists are not supported.

### audio

For audio data, the distribution of audio files durations is computed. The following measures are returned:

* minimum, maximum, mean, median, and standard deviation of audio files durations
* number and proportion of `null` values
* histogram of audio files durations with 10 bins

**Example**

```json
{
    "column_name": "audio",
    "column_type": "audio",
    "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0,
        "min": 1.02,
        "max": 15,
        "mean": 13.93042,
        "median": 14.77,
        "std": 2.63734,
        "histogram": {
            "hist": [
                32,
                25,
                18,
                24,
                22,
                17,
                18,
                19,
                55,
                1770
            ],
            "bin_edges": [
                1.02,
                2.418,
                3.816,
                5.214,
                6.612,
                8.01,
                9.408,
                10.806,
                12.204,
                13.602,
                15
            ]
        }
    }
}
```

### image

For image data, the distribution of images widths is computed. The following measures are returned:

* minimum, maximum, mean, median, and standard deviation of widths of image files
* number and proportion of `null` values
* histogram of images widths with 10 bins

**Example**

```json
{
    "column_name": "image",
    "column_type": "image",
    "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0.0,
        "min": 256,
        "max": 873,
        "mean": 327.99339,
        "median": 341.0,
        "std": 60.07286,
        "histogram": {
            "hist": [
                1734,
                1637,
                1326,
                121,
                10,
                3,
                1,
                3,
                1,
                2
            ],
            "bin_edges": [
                256,
                318,
                380,
                442,
                504,
                566,
                628,
                690,
                752,
                814,
                873
            ]
        }
    }
}
```

### datetime

The distribution of datetime is computed. The following measures are returned:

* minimum, maximum, mean, median, and standard deviation of datetimes represented as strings with precision up to seconds
* number and proportion of `null` values
* histogram of datetimes with 10 bins

**Example**

```json
{
    "column_name": "date",
    "column_type": "datetime",
    "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0.0,
        "min": "2013-05-18 04:54:11",
        "max": "2013-06-20 10:01:41",
        "mean": "2013-05-27 18:03:39",
        "median": "2013-05-23 11:55:50",
        "std": "11 days, 4:57:32.322450",
        "histogram": {
            "hist": [
                318776,
                393036,
                173904,
                0,
                0,
                0,
                0,
                0,
                0,
                206284
            ],
            "bin_edges": [
                "2013-05-18 04:54:11",
                "2013-05-21 12:36:57",
                "2013-05-24 20:19:43",
                "2013-05-28 04:02:29",
                "2013-05-31 11:45:15",
                "2013-06-03 19:28:01",
                "2013-06-07 03:10:47",
                "2013-06-10 10:53:33",
                "2013-06-13 18:36:19",
                "2013-06-17 02:19:05",
                "2013-06-20 10:01:41"
            ]
        }
    }
}
```

### Get Croissant metadata
https://huggingface.co/docs/dataset-viewer/croissant.md
