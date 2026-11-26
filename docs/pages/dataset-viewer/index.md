# 🤗 Dataset viewer

> The dataset page includes a table with the dataset's contents, arranged by pages of 100 rows.

Each page below links to its Markdown. The text of every page is in [llms-full.txt](https://huggingface.co/docs/dataset-viewer/llms-full.txt) (186 KB), and the text of each section in:

- [Get Started](https://huggingface.co/docs/dataset-viewer/llms-full/get-started.txt): 3 pages, 30 KB
- [Guides](https://huggingface.co/docs/dataset-viewer/llms-full/guides.txt): 20 pages, 148 KB
- [Conceptual Guides](https://huggingface.co/docs/dataset-viewer/llms-full/conceptual-guides.txt): 3 pages, 8 KB

## Get Started

- [🤗 Dataset viewer](https://huggingface.co/docs/dataset-viewer/index.md): The dataset page includes a table with the dataset's contents, arranged by pages of 100 rows.
- [Quickstart](https://huggingface.co/docs/dataset-viewer/quick_start.md): In this quickstart, you'll learn how to use the dataset viewer's REST API to:
- [Analyze a dataset on the Hub](https://huggingface.co/docs/dataset-viewer/analyze_data.md): In the Quickstart, you were introduced to various endpoints for interacting with datasets on the Hub.

## Guides

- [Check dataset validity](https://huggingface.co/docs/dataset-viewer/valid.md): Before you download a dataset from the Hub, it is helpful to know if a specific dataset you're interested in is available.
- [List splits and subsets](https://huggingface.co/docs/dataset-viewer/splits.md): Datasets typically have splits and may also have subsets.
- [Get dataset information](https://huggingface.co/docs/dataset-viewer/info.md): The dataset viewer provides an `/info` endpoint for exploring the general information about dataset, including such fields as description, citation, homepage, license and features.
- [Preview a dataset](https://huggingface.co/docs/dataset-viewer/first_rows.md): The dataset viewer provides a `/first-rows` endpoint for visualizing the first 100 rows of a dataset.
- [Download slices of rows](https://huggingface.co/docs/dataset-viewer/rows.md): The dataset viewer provides a `/rows` endpoint for visualizing any slice of rows of a dataset.
- [Search text in a dataset](https://huggingface.co/docs/dataset-viewer/search.md): The dataset viewer provides a `/search` endpoint for searching words in a dataset.
- [Filter rows in a dataset](https://huggingface.co/docs/dataset-viewer/filter.md): The dataset viewer provides a `/filter` endpoint for filtering rows in a dataset.
- [List Parquet files](https://huggingface.co/docs/dataset-viewer/parquet.md): Datasets can be published in any format (CSV, JSONL, directories of images, etc.) to the Hub, and they are easily accessed with the 🤗 Datasets library.
- [Get the number of rows and the bytes size](https://huggingface.co/docs/dataset-viewer/size.md): This guide shows you how to use the dataset viewer's `/size` endpoint to retrieve a dataset's size programmatically.
- [Explore dataset statistics](https://huggingface.co/docs/dataset-viewer/statistics.md): The dataset viewer provides a `/statistics` endpoint for fetching some basic statistics precomputed for a requested dataset.
- [Get Croissant metadata](https://huggingface.co/docs/dataset-viewer/croissant.md): The dataset viewer automatically generates the metadata in Croissant format (JSON-LD) for every dataset on the Hugging Face Hub.

### Query datasets from dataset viewer API

- [Overview](https://huggingface.co/docs/dataset-viewer/parquet_process.md): The dataset viewer automatically converts and publishes public datasets less than 5GB on the Hub as Parquet files.
- [ClickHouse](https://huggingface.co/docs/dataset-viewer/clickhouse.md): ClickHouse is a fast and efficient column-oriented database for analytical workloads, making it easy to analyze Hub-hosted datasets with SQL.
- [cuDF](https://huggingface.co/docs/dataset-viewer/cudf.md): cuDF is a Python GPU DataFrame library.
- [DuckDB](https://huggingface.co/docs/dataset-viewer/duckdb.md): DuckDB is a database that supports reading and querying Parquet files really fast.
- [Pandas](https://huggingface.co/docs/dataset-viewer/pandas.md): Pandas is a popular DataFrame library for data analysis.
- [Polars](https://huggingface.co/docs/dataset-viewer/polars.md): Polars is a fast DataFrame library written in Rust with Arrow as its foundation.
- [PostgreSQL](https://huggingface.co/docs/dataset-viewer/postgresql.md): PostgreSQL is a powerful, open source object-relational database system.
- [mlcroissant](https://huggingface.co/docs/dataset-viewer/mlcroissant.md): mlcroissant is a library to load datasets from Croissant metadata.
- [PySpark](https://huggingface.co/docs/dataset-viewer/pyspark.md): pyspark is the Python interface for Apache Spark, enabling large-scale data processing and real-time analytics in a distributed environment using Python.

## Conceptual Guides

- [Splits and subsets](https://huggingface.co/docs/dataset-viewer/configs_and_splits.md): Machine learning datasets are commonly organized in splits and they may also have subsets (also called configurations).
- [Data types](https://huggingface.co/docs/dataset-viewer/data_types.md): Datasets supported by the dataset viewer have a tabular format, meaning a data point is represented in a row and its features are contained in columns.
- [Server infrastructure](https://huggingface.co/docs/dataset-viewer/server.md): The dataset viewer has two main components that work together to return queries about a dataset instantly:

### 🤗 Dataset viewer
https://huggingface.co/docs/dataset-viewer/index.md

# 🤗 Dataset viewer

The dataset page includes a table with the dataset's contents, arranged by pages of 100 rows. You can navigate between pages using the buttons at the bottom of the table, filter, search, look at basic statistics, and more.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets-server/openbookqa_light.png)

Dataset viewer of the [OpenBookQA dataset](https://huggingface.co/datasets/openbookqa)

## Contents of the documentation

These documentation pages are focused on the **dataset viewer's backend** (code in https://github.com/huggingface/dataset-viewer), which provides the table with pre-computed data through an API for all the datasets on the Hub. You can explore the sections if you want to consume the API for your application or to understand how we preprocess the datasets.

Otherwise, if you want to learn about creating datasets from the Hub's web-based interface, [**configuring the dataset viewer**](https://huggingface.co/docs/hub/datasets-data-files-configuration) for data, [images](https://huggingface.co/docs/hub/datasets-image), or audio, or fixing errors, you might prefer reading the [Datasets Hub documentation pages](https://huggingface.co/docs/hub/datasets). Take also a look to the [example datasets](https://huggingface.co/datasets-examples) collections: [splits configuration](https://huggingface.co/collections/datasets-examples/file-names-and-splits-655e28af4471bd95709eb135), [subsets configuration](https://huggingface.co/collections/datasets-examples/manual-configuration-655e293cea26da0acab95b87), [CSV data files](https://huggingface.co/collections/datasets-examples/format-csv-and-tsv-655f681cb9673a4249cccb3d) and [image datasets](https://huggingface.co/collections/datasets-examples/image-dataset-6568e7cf28639db76eb92d65).

## Dataset viewer's backend

The dataset viewer's backend provides an API for visualizing and exploring all types of datasets - computer vision, speech, text, and tabular - stored on the Hugging Face [Hub](https://huggingface.co/datasets).

The main feature of the dataset viewer's backend is to auto-convert all the [Hub datasets](https://huggingface.co/datasets) to [Parquet](https://parquet.apache.org/). Read more in the [Parquet section](./parquet).

As datasets increase in size and data type richness, the cost of preprocessing (storage and compute) these datasets can be challenging and time-consuming.
To help users access these modern datasets, The dataset viewer runs a server behind the scenes to generate the API responses ahead of time and stores them in a database so they are instantly returned when you make a query through the API.

Let the dataset viewer take care of the heavy lifting so you can use a simple **REST API** on any of the **100,000+ datasets on Hugging Face** to:

- List the **dataset splits, column names and data types**
- Get the **dataset size** (in number of rows or bytes)
- Download and view **rows at any index** in the dataset
- **Search** a word in the dataset
- **Filter** rows based on a query string
- Get insightful **statistics** about the data
- Access the dataset as **parquet files** to use in your favorite **processing or analytics framework**

Join the growing community on the [forum](https://discuss.huggingface.co/) or [Discord](https://discord.com/invite/JfAtkvEtRb) today, and give the [dataset viewer repository](https://github.com/huggingface/dataset-viewer) a ⭐️ if you're interested in the latest updates!

### Quickstart
https://huggingface.co/docs/dataset-viewer/quick_start.md
