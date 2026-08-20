# Configuration
ENDPOINT_URL = "https://your-endpoint-name.endpoints.huggingface.cloud/v1/" # Endpoint URL + version
HF_TOKEN=REDACTED # Your Hugging Face Hub token from hf.co/settings/tokens

# Initialize OpenAI client for your endpoint
client = OpenAI(
    base_url=ENDPOINT_URL,
    api_key=HF_TOKEN,
)
```

Your OpenAI client is now configured to connect to your Inference Endpoint. For further reading you can check out the client documentation on text embeddings here.

### Step 3: Create the embedding function

Next, we'll create a function to process batches of text and return embeddings. 

```python
def get_embeddings(examples):
    """Get embeddings for a batch of texts."""
    response = client.embeddings.create(
        model="your-endpoint-name",  # Replace with your actual endpoint name
        input=examples["context"], # In the squad dataset, the text is in the "context" column
    )
    
    # Extract embeddings from response objects
    embeddings = [sample.embedding for sample in response.data]
    
    return {"embeddings": embeddings} # datasets expects a dictionary with a key "embeddings" and a value of a list of embeddings
```

The `datasets` library will pass our function a batch of examples from the dataset, as a dictionary of batch values. The key will be the name of the column we want to embed, and the value will be a list of values from that column.

### Step 4: Load and process your dataset

Load your dataset and apply the embedding function:

```python
# Load a sample dataset (you can replace this with your own)
dataset = load_dataset("squad", split="train[:100]")  # Using first 100 examples for demo

# Process the dataset with embeddings
dataset_with_embeddings = dataset.map(
    get_embeddings,
    batched=True,
    batch_size=10,  # Process in small batches to avoid timeouts
    desc="Adding embeddings",
)
```

The `datasets` library's `map` function is optimized for performance and will automatically batch the rows for us. Inference Endpoints can also scale to meet the demand of the batch size, so to get the best performance, you should calibrate the batch size with your Inference Endpoints's configuration.

For example, select the highest possible batch size for you model and synchronize the batch size with your Inference Endpoint's configuration in `max_concurrent_requests`.

### Step 5: Save and Share your results

Finally, let's save our embedded dataset locally or push it to the Hugging Face Hub:

```python
# Save the processed dataset locally
dataset_with_embeddings.save_to_disk("./embedded_dataset")

# Or push directly to Hugging Face Hub
dataset_with_embeddings.push_to_hub("your-username/squad-embeddings")
```

## Next steps

Nice work! You've now built an embedding pipeline that can process any dataset. Here's the complete script:

Click to view the complete script

```python
import os
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Configuration
ENDPOINT_URL = "https://your-endpoint-name.endpoints.huggingface.cloud/v1/"
HF_TOKEN=REDACTED

# Initialize OpenAI client for your endpoint
client = OpenAI(
    base_url=ENDPOINT_URL,
    api_key=HF_TOKEN,
)

def get_embeddings(examples):
    """Get embeddings for a batch of texts."""
    response = client.embeddings.create(
        model="your-endpoint-name",  # Replace with your actual endpoint name
        input=examples["context"],
    )
    
    # Extract embeddings from response
    embeddings = [sample.embedding for sample in response.data]
    
    return {"embeddings": embeddings}

# Load a sample dataset (you can replace this with your own)
print("Loading dataset...")
dataset = load_dataset("squad", split="train[:1000]")  # Using first 1000 examples for demo

# Process the dataset with embeddings
print("Processing dataset with embeddings...")
dataset_with_embeddings = dataset.map(
    get_embeddings,
    batched=True,
    batch_size=10,  # Process in small batches to avoid timeouts
    desc="Adding embeddings",
)

# Save the processed dataset locally
print("Saving processed dataset...")
dataset_with_embeddings.save_to_disk("./embedded_dataset")

# Or push directly to Hugging Face Hub
print("Pushing to Hugging Face Hub...")
dataset_with_embeddings.push_to_hub("your-username/squad-embeddings")

print("Dataset processing complete!")
```

Here are some ways to extend your script:

- **Process multiple datasets**: Modify the script to handle different dataset sources
- **Add error handling**: Implement retry logic for failed API calls
- **Optimize batch sizes**: Experiment with different batch sizes for better performance
- **Add validation**: Check embedding quality and dimensions
- **Custom preprocessing**: Add text cleaning or normalization steps
- **Build a Semantic Search Application**: Use the embeddings to build a semantic search application.

Your embedded datasets are now ready for downstream tasks like semantic search, recommendation systems, or RAG applications!

### FAQs
https://huggingface.co/docs/inference-endpoints/support/faq.md
