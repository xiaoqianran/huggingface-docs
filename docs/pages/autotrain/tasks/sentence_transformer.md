# Sentence Transformers

For all sentence transformers tasks, one needs to map columns to `sentence1_column`, `sentence2_column`, `sentence3_column` & `target_column` column.
Not all columns need to be mapped for all trainers of sentence transformers.

## `pair`:

```
{"sentence1_column": "anchor", "sentence2_column": "positive"}
```

## `pair_class`:

```
{"sentence1_column": "premise", "sentence2_column": "hypothesis", "target_column": "label"}
```

## `pair_score`:

```
{"sentence1_column": "sentence1", "sentence2_column": "sentence2", "target_column": "score"}
```

## `triplet`:

```
{"sentence1_column": "anchor", "sentence2_column": "positive", "sentence3_column": "negative"}
```

## `qa`:

```
{"sentence1_column": "query", "sentence2_column": "answer"}
```

# Extractive Question Answering

For extractive question answering, the column mapping should be as follows:

```
{"text": "context", "question": "question", "answer": "answers"}
```

where `answer` is a dictionary with keys `text` and `answer_start`.

## Ensuring Accurate Mapping

To ensure your model trains correctly:

- Verify Column Names: Double-check that the names used in the mapping dictionary accurately reflect those in your dataset.

- Format Appropriately: Especially in token classification, ensure your data format matches expectations (e.g., lists of strings).

- Update Mappings for New Datasets: Each new dataset might require its unique mappings based on its structure and the task at hand.

By following these guidelines and using the provided examples as templates, 
you can effectively instruct AutoTrain on how to interpret and handle your 
data for various machine learning tasks. This process is fundamental for 
achieving optimal results from your model training endeavors.

### AutoTrain Configs
https://huggingface.co/docs/autotrain/main/config.md
