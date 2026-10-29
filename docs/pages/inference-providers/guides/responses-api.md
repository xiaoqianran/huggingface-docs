# Responses API (beta)

The Responses API (from OpenAI) provides a unified interface for model interactions with Hugging Face Inference Providers. Use your existing OpenAI SDKs to access features like multi-provider routing, event streaming, structured outputs, and Remote MCP tools.

> [!TIP]
> This guide assumes you have a Hugging Face account and access token. You can create a free account at [huggingface.co](https://huggingface.co) and get your token from your [settings page](https://huggingface.co/settings/tokens).

## Why build with the Responses API?

The Responses API provides a unified interface built for agentic apps. With it, you get:

- **Built-in tool orchestration.** Invoke functions, server-side MCP tools, and schema-validated outputs without changing endpoints.
- **Event-driven streaming.** Receive semantic events such as `response.created`, `output_text.delta`, and `response.completed` to power incremental UIs.
- **Reasoning controls and structured outputs.** Dial up or down reasoning effort and require models to return schema-compliant JSON every time.

## Prerequisites

- A Hugging Face account with remaining Inference Providers credits (purchased, or included with PRO, Team, or Enterprise).
- A fine-grained [Hugging Face token](https://huggingface.co/settings/tokens) with “Make calls to Inference Providers” permission stored in `HF_TOKEN`.

> [!TIP]
> All Inference Providers chat completion models should be compatible with the Responses API. You can browse available models on the [Inference Models page](https://huggingface.co/inference/models).

## Configure your Responses client

Install the OpenAI SDK for your language of choice before running the snippets below (`pip install openai` for Python or `npm install openai` for Node.js). If you prefer issuing raw HTTP calls, any standard tool such as `curl` will work as well.

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

response = client.responses.create(
    model="openai/gpt-oss-120b:groq",
    instructions="You are a helpful assistant.",
    input="Tell me a three-sentence bedtime story about a unicorn.",
)

print(response.output_text)
```

```ts
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: REDACTED,
});

const response = await client.responses.create({
  model: "openai/gpt-oss-120b:groq",
  instructions: "You are a helpful assistant.",
  input: "Tell me a three-sentence bedtime story about a unicorn.",
});

console.log(response.output_text);
```

```bash
curl https://router.huggingface.co/v1/responses \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b:groq",
    "instructions": "You are a helpful assistant.",
    "input": "Tell me a three-sentence bedtime story about a unicorn."
  }'
```

> [!TIP]
> If you plan to use a specific provider, append it to the model id as `<repo>:<provider>` (for example `moonshotai/Kimi-K2-Instruct-0905:groq`). Otherwise, omit the suffix and the fastest available provider will be selected by default (equivalent to the `:fastest` policy). You can also use `:cheapest` or `:preferred` (follows your [Inference Provider settings](https://hf.co/settings/inference-providers) order).

## Core Response patterns

### Plain text output

For a single response message, pass a string as input. The Responses API returns both the full `response` object and a convenience `output_text` helper.

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

response = client.responses.create(
    model="moonshotai/Kimi-K2-Instruct-0905:groq",
    instructions="You are a helpful assistant.",
    input="Tell me a three sentence bedtime story about a unicorn.",
)

print(response.output_text)
```

```ts
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: REDACTED,
});

const response = await client.responses.create({
  model: "moonshotai/Kimi-K2-Instruct-0905:groq",
  instructions: "You are a helpful assistant.",
  input: "Tell me a three sentence bedtime story about a unicorn.",
});

console.log(response.output_text);
```

```bash
curl https://router.huggingface.co/v1/responses \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "moonshotai/Kimi-K2-Instruct-0905:groq",
    "instructions": "You are a helpful assistant.",
    "input": "Tell me a three sentence bedtime story about a unicorn."
  }'
```

### Multimodal inputs

Mix text and vision content by passing a list of content parts. The Responses API unifies text and images into a single `input` array.

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

response = client.responses.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "what is in this image?"},
                {
                    "type": "input_image",
                    "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                },
            ],
        }
    ],
)

print(response.output_text)
```

```ts
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: REDACTED,
});

const response = await client.responses.create({
  model: "Qwen/Qwen2.5-VL-7B-Instruct",
  input: [
    {
      role: "user",
      content: [
        { type: "input_text", text: "what is in this image?" },
        {
          type: "input_image",
          image_url:
            "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        },
      ],
    },
  ],
});

console.log(response.output_text);
```

```bash
curl https://router.huggingface.co/v1/responses \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "input": [
      {
        "role": "user",
        "content": [
          {"type": "input_text", "text": "what is in this image?"},
          {
            "type": "input_image",
            "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
          }
        ]
      }
    ]
  }'
```

### Multi-turn conversations

Responses requests accept conversation history. Add `developer`, `system`, and `user` messages to control the assistant's behavior without managing chat state yourself.

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

response = client.responses.create(
    model="moonshotai/Kimi-K2-Instruct-0905:groq",
    input=[
        {"role": "developer", "content": "Talk like a pirate."},
        {"role": "user", "content": "Are semicolons optional in JavaScript?"},
    ],
)

print(response.output_text)
```

```ts
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: REDACTED,
});

const response = await client.responses.create({
  model: "moonshotai/Kimi-K2-Instruct-0905:groq",
  input: [
    { role: "developer", content: "Talk like a pirate." },
    { role: "user", content: "Are semicolons optional in JavaScript?" },
  ],
});

console.log(response.output_text);
```

```bash
curl https://router.huggingface.co/v1/responses \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "moonshotai/Kimi-K2-Instruct-0905:groq",
    "input": [
      {"role": "developer", "content": "Talk like a pirate."},
      {"role": "user", "content": "Are semicolons optional in JavaScript?"}
    ]
  }'
```

## Advanced features

Advanced features use the same request format.

### Event-based streaming

Set `stream=True` to receive incremental `response.*` events. Each event arrives as JSON, so you can render words as they stream in or monitor tool execution in real time.

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

stream = client.responses.create(
    model="moonshotai/Kimi-K2-Instruct-0905:groq",
    input=[{"role": "user", "content": "Say 'double bubble bath' ten times fast."}],
    stream=True,
)

for event in stream:
    print(event)
```

```ts
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: REDACTED,
});

const stream = await client.responses.create({
  model: "moonshotai/Kimi-K2-Instruct-0905:groq",
  input: [{ role: "user", content: "Say 'double bubble bath' ten times fast." }],
  stream: true,
});

for await (const event of stream) {
  console.log(event);
}
```

```bash
curl -N https://router.huggingface.co/v1/responses \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "moonshotai/Kimi-K2-Instruct-0905:groq",
    "input": [
      {"role": "user", "content": "Say \"double bubble bath\" ten times fast."}
    ],
    "stream": true
  }'
```

### Tool calling and routing

Add a `tools` array to let the model call your functions. The router handles the function calls and returns tool events.

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

tools = [
    {
        "type": "function",
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location", "unit"],
        },
    }
]

response = client.responses.create(
    model="moonshotai/Kimi-K2-Instruct-0905:groq",
    tools=tools,
    input="What is the weather like in Boston today?",
    tool_choice="auto",
)

print(response)
```

```ts
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: REDACTED,
});

const tools = [
  {
    type: "function",
    name: "get_current_weather",
    description: "Get the current weather in a given location",
    parameters: {
      type: "object",
      properties: {
        location: { type: "string", description: "The city and state, e.g. San Francisco, CA" },
        unit: { type: "string", enum: ["celsius", "fahrenheit"] },
      },
      required: ["location", "unit"],
    },
  },
];

const response = await client.responses.create({
  model: "moonshotai/Kimi-K2-Instruct-0905:groq",
  tools,
  input: "What is the weather like in Boston today?",
  tool_choice: "auto",
});

console.log(response);
```

```bash
curl https://router.huggingface.co/v1/responses \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "moonshotai/Kimi-K2-Instruct-0905:groq",
    "input": "What is the weather like in Boston today?",
    "tool_choice": "auto",
    "tools": [
      {
        "type": "function",
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
          },
          "required": ["location", "unit"]
        }
      }
    ]
  }'
```

### Structured outputs

Force the model to return JSON matching a schema by supplying a `response_format`. The Python SDK exposes a `.parse` helper that converts the response directly into your target type.

> [!NOTE]
> When calling `openai/gpt-oss-120b:groq` from JavaScript or raw HTTP, include a brief instruction to return JSON. Without it the model may emit markdown even when a schema is provided.

```python
from openai import OpenAI
from pydantic import BaseModel
import os

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

response = client.responses.parse(
    model="openai/gpt-oss-120b:groq",
    input=[
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
    ],
    text_format=CalendarEvent,
)

print(response.output_parsed)
```

```ts
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: REDACTED,
});

const response = await client.responses.create({
  model: "openai/gpt-oss-120b:groq",
  instructions: "Return JSON that matches the CalendarEvent schema (fields name, date, participants).",
  input: [
    { role: "system", content: "Extract the event information." },
    { role: "user", content: "Alice and Bob are going to a science fair on Friday." },
  ],
  response_format: {
    type: "json_schema",
    json_schema: {
      name: "CalendarEvent",
      schema: {
        type: "object",
        properties: {
          name: { type: "string" },
          date: { type: "string" },
          participants: {
            type: "array",
            items: { type: "string" },
          },
        },
        required: ["name", "date", "participants"],
        additionalProperties: false,
      },
      strict: true,
    },
  },
});

const parsed = JSON.parse(response.output_text);
console.log(parsed);
```

```bash
curl https://router.huggingface.co/v1/responses \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b:groq",
    "instructions": "Return JSON that matches the CalendarEvent schema (fields name, date, participants).",
    "input": [
      {"role": "system", "content": "Extract the event information."},
      {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."}
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "CalendarEvent",
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "date": {"type": "string"},
            "participants": {
              "type": "array",
              "items": {"type": "string"}
            }
          },
          "required": ["name", "date", "participants"],
          "additionalProperties": false
        },
        "strict": true
      }
    }
  }'
```

### Remote MCP execution

Remote MCP lets you call server-hosted tools that implement the Model Context Protocol. Provide the MCP server URL and allowed tools, and the Responses API handles the calls for you.

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

response = client.responses.create(
    model="moonshotai/Kimi-K2-Instruct-0905:groq",
    input="how does tiktoken work?",
    tools=[
        {
            "type": "mcp",
            "server_label": "gitmcp",
            "server_url": "https://gitmcp.io/openai/tiktoken",
            "allowed_tools": ["search_tiktoken_documentation", "fetch_tiktoken_documentation"],
            "require_approval": "never",
        },
    ],
)

for output in response.output:
    print(output)
```

```ts
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: REDACTED,
});

const response = await client.responses.create({
  model: "moonshotai/Kimi-K2-Instruct-0905:groq",
  input: "how does tiktoken work?",
  tools: [
    {
      type: "mcp",
      server_label: "gitmcp",
      server_url: "https://gitmcp.io/openai/tiktoken",
      allowed_tools: ["search_tiktoken_documentation", "fetch_tiktoken_documentation"],
      require_approval: "never",
    },
  ],
});

for (const output of response.output) {
  console.log(output);
}
```

```bash
curl https://router.huggingface.co/v1/responses \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "moonshotai/Kimi-K2-Instruct-0905:groq",
    "input": "how does tiktoken work?",
    "tools": [
      {
        "type": "mcp",
        "server_label": "gitmcp",
        "server_url": "https://gitmcp.io/openai/tiktoken",
        "allowed_tools": ["search_tiktoken_documentation", "fetch_tiktoken_documentation"],
        "require_approval": "never"
      }
    ]
  }'
```

### Reasoning effort controls

Some open-source reasoning models expose effort tiers. Pass `reasoning={"effort": "low" | "medium" | "high"}` to trade off latency and depth.

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

response = client.responses.create(
    model="openai/gpt-oss-120b:groq",
    instructions="You are a helpful assistant.",
    input="Say hello to the world.",
    reasoning={"effort": "low"},
)

for i, item in enumerate(response.output):
    print(f"Output #{i}: {item.type}", item.content)
```

```ts
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: REDACTED,
});

const response = await client.responses.create({
  model: "openai/gpt-oss-120b:groq",
  instructions: "You are a helpful assistant.",
  input: "Say hello to the world.",
  reasoning: { effort: "low" },
});

response.output.forEach((item, index) => {
  console.log(`Output #${index}: ${item.type}`, item.content);
});
```

```bash
curl https://router.huggingface.co/v1/responses \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b:groq",
    "instructions": "You are a helpful assistant.",
    "input": "Say hello to the world.",
    "reasoning": {"effort": "low"}
  }'
```

## API reference

Read the official [OpenAI Responses reference](https://platform.openai.com/docs/api-reference/responses).

### Template
https://huggingface.co/docs/inference-providers/providers/cohere.md

### Template

If you want to update the content related to cohere's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/cohere.handlebars`.

### Logos

If you want to update cohere's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `cohere-light.png` and `cohere-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Cohere

> [!TIP]
> All supported Cohere models can be found [here](https://huggingface.co/models?inference_provider=cohere&sort=trending)

    
        
        
    

    
        
        
    

Cohere brings you cutting-edge multilingual models, advanced retrieval, and an AI workspace tailored for the modern enterprise — all within a single, secure platform.

Cohere is the first model creator to share and serve their models directly on the Hub as an Inference Provider.

## Supported tasks

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"cohere":{"modelId":"CohereLabs/tiny-aya-global","providerModelId":"tiny-aya-global"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"cohere":{"modelId":"CohereLabs/aya-vision-32b","providerModelId":"c4ai-aya-vision-32b"} } }
conversational />

### Template
https://huggingface.co/docs/inference-providers/providers/publicai.md

### Template

If you want to update the content related to publicai's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/publicai.handlebars`.

### Logos

If you want to update publicai's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `publicai-light.png` and `publicai-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# PublicAI

> [!TIP]
> All models supported by the PublicAI Inference Utility can be found [here](https://huggingface.co/models?inference_provider=publicai&sort=trending)

    
        
        
    

    
        
        
    

The Public AI Inference Utility is a nonprofit, open-source project. Their team builds products and organizes advocacy to support the work of public AI model builders like the Swiss AI Initiative, AI Singapore, AI Sweden, and the Barcelona Supercomputing Center.

They believe in public AI — AI as public infrastructure like highways, water, or electricity. Think of a BBC for AI, a public utility for AI, or public libraries for AI.

## Supported tasks

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"publicai":{"modelId":"speakleash/Bielik-11B-v3.0-Instruct","providerModelId":"speakleash/Bielik-11B-v3.0-Instruct"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"publicai":{"modelId":"swiss-ai/Apertus-v1.5-70B","providerModelId":"swiss-ai/apertus-v1.5-70b"} } }
conversational />

### Template
https://huggingface.co/docs/inference-providers/providers/ovhcloud.md

### Template

If you want to update the content related to ovhcloud's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/ovhcloud.handlebars`.

### Logos

If you want to update ovhcloud's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `ovhcloud-light.png` and `ovhcloud-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# OVHcloud AI Endpoints

> [!TIP]
> All supported OVHcloud AI Endpoints models can be found [here](https://huggingface.co/models?inference_provider=ovhcloud&sort=trending)

    
        
        
    

    
        
        
    

[OVHcloud AI Endpoints](https://www.ovhcloud.com/en/public-cloud/ai-endpoints/) is a managed AI inference service that provides access to a wide range of state-of-the-art machine learning models. As part of OVHcloud's Public Cloud offering, AI Endpoints enables developers to easily integrate AI capabilities into their applications with data sovereignty, privacy, and GDPR compliance.

## Supported tasks

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"ovhcloud":{"modelId":"openai/gpt-oss-120b","providerModelId":"gpt-oss-120b"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"ovhcloud":{"modelId":"Qwen/Qwen3.8-27B","providerModelId":"Qwen3.8-27B"} } }
conversational />

### Template
https://huggingface.co/docs/inference-providers/providers/novita.md

### Template

If you want to update the content related to novita's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/novita.handlebars`.

### Logos

If you want to update novita's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `novita-light.png` and `novita-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Novita

> [!TIP]
> All supported Novita models can be found [here](https://huggingface.co/models?inference_provider=novita&sort=trending)

    
        
        
    

    
        
        
    

[Novita](https://novita.ai) is the go-to inference platform for AI developers seeking a low-cost, reliable, and simple solution for shipping AI models.

Offering 200+ APIs (LLMs, image, video, audio) with fully managed deployment — enterprise-grade, scalable, and maintenance-free.

## Supported tasks

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"novita":{"modelId":"zai-org/GLM-5.3","providerModelId":"zai-org/glm-5.3"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"novita":{"modelId":"deepseek-ai/DeepSeek-V4.1-Flash-1017","providerModelId":"deepseek/deepseek-v4.1-flash-1017"} } }
conversational />

### Template
https://huggingface.co/docs/inference-providers/providers/hf-inference.md

### Template

If you want to update the content related to hf-inference's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/hf-inference.handlebars`.

### Logos

If you want to update hf-inference's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `hf-inference-light.png` and `hf-inference-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# HF Inference

> [!TIP]
> All supported HF Inference models can be found [here](https://huggingface.co/models?inference_provider=hf-inference&sort=trending)

    
        
        
    

    
        
        
    

HF Inference is the serverless Inference API powered by Hugging Face. This service used to be called "Inference API (serverless)" prior to Inference Providers.
If you are interested in deploying models to a dedicated and autoscaling infrastructure managed by Hugging Face, check out [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) instead.

As of July 2025, hf-inference focuses mostly on CPU inference (e.g. embedding, text-ranking, text-classification, or smaller LLMs that have historical importance like BERT or GPT-2).

## Supported tasks

### Automatic Speech Recognition

Find out more about Automatic Speech Recognition [here](../tasks/automatic-speech-recognition).

<InferenceSnippet
    pipeline=automatic-speech-recognition
    providersMapping={ {"hf-inference":{"modelId":"openai/whisper-large-v3","providerModelId":"openai/whisper-large-v3"} } }
/>

### Feature Extraction

Find out more about Feature Extraction [here](../tasks/feature-extraction).

<InferenceSnippet
    pipeline=feature-extraction
    providersMapping={ {"hf-inference":{"modelId":"perplexity-ai/pplx-embed-v2-late-0.6b","providerModelId":"perplexity-ai/pplx-embed-v2-late-0.6b"} } }
/>

### Fill Mask

Find out more about Fill Mask [here](../tasks/fill-mask).

<InferenceSnippet
    pipeline=fill-mask
    providersMapping={ {"hf-inference":{"modelId":"google-bert/bert-base-uncased","providerModelId":"google-bert/bert-base-uncased"} } }
/>

### Image Classification

Find out more about Image Classification [here](../tasks/image-classification).

<InferenceSnippet
    pipeline=image-classification
    providersMapping={ {"hf-inference":{"modelId":"Falconsai/nsfw_image_detection","providerModelId":"Falconsai/nsfw_image_detection"} } }
/>

### Image Segmentation

Find out more about Image Segmentation [here](../tasks/image-segmentation).

<InferenceSnippet
    pipeline=image-segmentation
    providersMapping={ {"hf-inference":{"modelId":"nvidia/segformer-b5-finetuned-ade-640-640","providerModelId":"nvidia/segformer-b5-finetuned-ade-640-640"} } }
/>

### Object Detection

Find out more about Object Detection [here](../tasks/object-detection).

<InferenceSnippet
    pipeline=object-detection
    providersMapping={ {"hf-inference":{"modelId":"facebook/detr-resnet-50","providerModelId":"facebook/detr-resnet-50"} } }
/>

### Question Answering

Find out more about Question Answering [here](../tasks/question-answering).

<InferenceSnippet
    pipeline=question-answering
    providersMapping={ {"hf-inference":{"modelId":"deepset/roberta-base-squad2","providerModelId":"deepset/roberta-base-squad2"} } }
/>

### Summarization

Find out more about Summarization [here](../tasks/summarization).

<InferenceSnippet
    pipeline=summarization
    providersMapping={ {"hf-inference":{"modelId":"facebook/bart-large-cnn","providerModelId":"facebook/bart-large-cnn"} } }
/>

### Table Question Answering

Find out more about Table Question Answering [here](../tasks/table-question-answering).

<InferenceSnippet
    pipeline=table-question-answering
    providersMapping={ {"hf-inference":{"modelId":"google/tapas-base-finetuned-wtq","providerModelId":"google/tapas-base-finetuned-wtq"} } }
/>

### Text Classification

Find out more about Text Classification [here](../tasks/text-classification).

<InferenceSnippet
    pipeline=text-classification
    providersMapping={ {"hf-inference":{"modelId":"meta-llama/Llama-Prompt-Guard-2-86M","providerModelId":"meta-llama/Llama-Prompt-Guard-2-86M"} } }
/>

### Text To Image

Find out more about Text To Image [here](../tasks/text-to-image).

<InferenceSnippet
    pipeline=text-to-image
    providersMapping={ {"hf-inference":{"modelId":"stabilityai/stable-diffusion-3-medium-diffusers","providerModelId":"stabilityai/stable-diffusion-3-medium-diffusers"} } }
/>

### Token Classification

Find out more about Token Classification [here](../tasks/token-classification).

<InferenceSnippet
    pipeline=token-classification
    providersMapping={ {"hf-inference":{"modelId":"dslim/bert-base-NER","providerModelId":"dslim/bert-base-NER"} } }
/>

### Translation

Find out more about Translation [here](../tasks/translation).

<InferenceSnippet
    pipeline=translation
    providersMapping={ {"hf-inference":{"modelId":"google-t5/t5-small","providerModelId":"google-t5/t5-small"} } }
/>

### Zero Shot Classification

Find out more about Zero Shot Classification [here](../tasks/zero-shot-classification).

<InferenceSnippet
    pipeline=zero-shot-classification
    providersMapping={ {"hf-inference":{"modelId":"jeff-legacy/Jeff-Qwen3.5-0.8B","providerModelId":"jeff-legacy/Jeff-Qwen3.5-0.8B"} } }
/>

### Template
https://huggingface.co/docs/inference-providers/providers/fal-ai.md

### Template

If you want to update the content related to fal-ai's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/fal-ai.handlebars`.

### Logos

If you want to update fal-ai's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `fal-ai-light.png` and `fal-ai-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Fal

> [!TIP]
> All supported Fal models can be found [here](https://huggingface.co/models?inference_provider=fal-ai&sort=trending)

    
        
        
    

    
        
        
    

[fal](https://fal.ai/) is the generative media inference platform: call 1,000+ models by API, or deploy your own models, pipelines and apps on fal's autoscaling GPUs with [fal Serverless](https://fal.ai/docs/documentation/serverless). Serverless deployments scale from zero to thousands of runners and back, are billed per second of runner lifetime ([pricing](https://fal.ai/docs/documentation/serverless/pricing)), and can be deployed from a Python `fal.App`, a custom Docker container, or an existing HTTP server.

Founded in 2021 by [Burkay Gur](https://huggingface.co/burkaygur) and [Gorkem Yurtseven](https://huggingface.co/gorkemyurt), fal.ai was born out of a shared passion for AI and a desire to address the challenges in AI infrastructure observed during their tenures at Coinbase and Amazon.

## Supported tasks

### Automatic Speech Recognition

Find out more about Automatic Speech Recognition [here](../tasks/automatic-speech-recognition).

<InferenceSnippet
    pipeline=automatic-speech-recognition
    providersMapping={ {"fal-ai":{"modelId":"openai/whisper-large-v3","providerModelId":"fal-ai/whisper"} } }
/>

### Image Segmentation

Find out more about Image Segmentation [here](../tasks/image-segmentation).

<InferenceSnippet
    pipeline=image-segmentation
    providersMapping={ {"fal-ai":{"modelId":"briaai/RMBG-2.0","providerModelId":"fal-ai/bria/background/remove"} } }
/>

### Image To Image

Find out more about Image To Image [here](../tasks/image-to-image).

<InferenceSnippet
    pipeline=image-to-image
    providersMapping={ {"fal-ai":{"modelId":"black-forest-labs/FLUX.2-klein-9B","providerModelId":"fal-ai/flux-2/klein/9b/edit"} } }
/>

### Text To Image

Find out more about Text To Image [here](../tasks/text-to-image).

<InferenceSnippet
    pipeline=text-to-image
    providersMapping={ {"fal-ai":{"modelId":"krea/Krea-2-Turbo","providerModelId":"fal-ai/krea-2/turbo"} } }
/>

### Text To Video

Find out more about Text To Video [here](../tasks/text-to-video).

<InferenceSnippet
    pipeline=text-to-video
    providersMapping={ {"fal-ai":{"modelId":"Wan-AI/Wan2.2-TI2V-5B","providerModelId":"fal-ai/wan/v2.2-5b/text-to-video"} } }
/>

### Template
https://huggingface.co/docs/inference-providers/providers/wavespeed.md

### Template

If you want to update the content related to wavespeed's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/wavespeed.handlebars`.

### Logos

If you want to update wavespeed's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `wavespeed-light.png` and `wavespeed-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# WaveSpeed

> [!TIP]
> All supported WaveSpeed models can be found [here](https://huggingface.co/models?inference_provider=wavespeed&sort=trending)

    
        
        
    

    
        
        
    

[WaveSpeedAI](https://wavespeed.ai/) is a high-performance AI inference platform specializing in image and video generation. Built with cutting-edge infrastructure and optimization techniques, [WaveSpeedAI](https://wavespeed.ai/) provides fast, scalable, and cost-effective model serving for creative AI applications.

## Supported tasks

### Image To Image

Find out more about Image To Image [here](../tasks/image-to-image).

<InferenceSnippet
    pipeline=image-to-image
    providersMapping={ {"wavespeed":{"modelId":"briaai/Fibo-2.0-edit-turbo","providerModelId":"bria/fibo-2.0/edit"} } }
/>

### Text To Image

Find out more about Text To Image [here](../tasks/text-to-image).

<InferenceSnippet
    pipeline=text-to-image
    providersMapping={ {"wavespeed":{"modelId":"krea/Krea-2-Turbo","providerModelId":"wavespeed-ai/krea-v2-medium-turbo/text-to-image"} } }
/>

### Text To Video

Find out more about Text To Video [here](../tasks/text-to-video).

<InferenceSnippet
    pipeline=text-to-video
    providersMapping={ {"wavespeed":{"modelId":"ZhengmingYu/DMAD","providerModelId":"wavespeed-ai/minimax-h3/text-to-video-lora"} } }
/>

### Template
https://huggingface.co/docs/inference-providers/providers/deepinfra.md

### Template

If you want to update the content related to deepinfra's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/deepinfra.handlebars`.

### Logos

If you want to update deepinfra's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `deepinfra-light.png` and `deepinfra-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# DeepInfra

> [!TIP]
> All supported DeepInfra models can be found [here](https://huggingface.co/models?inference_provider=deepinfra&sort=trending)

    
        
        
    

    
        
        
    

DeepInfra is a serverless AI inference platform offering one of the most cost-effective pricing per token in the industry. With a catalog of over 100 models spanning LLMs, text-to-image, text-to-speech, speech-to-text, video generation, OCR, and more, DeepInfra makes it easy for developers to integrate a wide range of AI capabilities into their applications with minimal setup.

## Resources
- **Website**: https://deepinfra.com/
- **Documentation**: https://docs.deepinfra.com

## Supported tasks

### Automatic Speech Recognition

Find out more about Automatic Speech Recognition [here](../tasks/automatic-speech-recognition).

<InferenceSnippet
    pipeline=automatic-speech-recognition
    providersMapping={ {"deepinfra":{"modelId":"nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B","providerModelId":"nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"} } }
/>

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"deepinfra":{"modelId":"zai-org/GLM-5.3","providerModelId":"zai-org/GLM-5.3"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"deepinfra":{"modelId":"Qwen/Qwen3.8-27B","providerModelId":"Qwen/Qwen3.8-27B"} } }
conversational />

### Feature Extraction

Find out more about Feature Extraction [here](../tasks/feature-extraction).

<InferenceSnippet
    pipeline=feature-extraction
    providersMapping={ {"deepinfra":{"modelId":"Qwen/Qwen3-Embedding-0.6B","providerModelId":"Qwen/Qwen3-Embedding-0.6B"} } }
/>

### Template
https://huggingface.co/docs/inference-providers/providers/cerebras.md

### Template

If you want to update the content related to cerebras's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/cerebras.handlebars`.

### Logos

If you want to update cerebras's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `cerebras-light.png` and `cerebras-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Cerebras

> [!TIP]
> All supported Cerebras models can be found [here](https://huggingface.co/models?inference_provider=cerebras&sort=trending)

    
        
        
    

    
        
        
    

Cerebras stands alone as the world’s fastest AI inference and training platform. Organizations across fields like medical research, cryptography, energy, and agentic AI use our CS-2 and CS-3 systems to build on-premise supercomputers, while developers and enterprises everywhere can access the power of Cerebras through our pay-as-you-go cloud offerings.

## Supported tasks

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"cerebras":{"modelId":"openai/gpt-oss-120b","providerModelId":"gpt-oss-120b"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"cerebras":{"modelId":"Qwen/Qwen3.8-27B","providerModelId":"qwen-3.8-27b"} } }
conversational />

### Template
https://huggingface.co/docs/inference-providers/providers/together.md

### Template

If you want to update the content related to together's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/together.handlebars`.

### Logos

If you want to update together's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `together-light.png` and `together-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Together

> [!TIP]
> All supported Together models can be found [here](https://huggingface.co/models?inference_provider=together&sort=trending)

    
        
        
    

    
        
        
    

Together decentralized cloud services empower developers and researchers at organizations of all sizes to train, fine-tune, and deploy generative AI models.

## Supported tasks

### Automatic Speech Recognition

Find out more about Automatic Speech Recognition [here](../tasks/automatic-speech-recognition).

<InferenceSnippet
    pipeline=automatic-speech-recognition
    providersMapping={ {"together":{"modelId":"nvidia/nemotron-3.5-asr-streaming-0.6b","providerModelId":"nvidia/nemotron-3.5-asr-streaming-0.6b"} } }
/>

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"together":{"modelId":"zai-org/GLM-5.3","providerModelId":"zai-org/GLM-5.3"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"together":{"modelId":"deepseek-ai/DeepSeek-V4.1-Flash-1017","providerModelId":"deepseek-ai/DeepSeek-V4.1-Flash-1017"} } }
conversational />

### Template
https://huggingface.co/docs/inference-providers/providers/nscale.md

### Template

If you want to update the content related to nscale's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/nscale.handlebars`.

### Logos

If you want to update nscale's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `nscale-light.png` and `nscale-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Nscale

> [!TIP]
> All supported Nscale models can be found [here](https://huggingface.co/models?inference_provider=nscale&sort=trending)

    
        
        
    

    
        
        
    

Nscale is a vertically integrated AI cloud that delivers bespoke, sovereign AI infrastructure at scale.

Built on this foundation, Nscale's inference service empowers developers with a wide range of models and ready-to-use inference services that integrate into workflows without the need to manage the underlying infrastructure.

## Supported tasks

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"nscale":{"modelId":"zai-org/GLM-5.3","providerModelId":"zai-org/GLM-5.3"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"nscale":{"modelId":"Qwen/Qwen3.8-27B","providerModelId":"Qwen/Qwen3.8-27B"} } }
conversational />

### Text To Image

Find out more about Text To Image [here](../tasks/text-to-image).

<InferenceSnippet
    pipeline=text-to-image
    providersMapping={ {"nscale":{"modelId":"black-forest-labs/FLUX.1-schnell","providerModelId":"black-forest-labs/FLUX.1-schnell"} } }
/>

### Template
https://huggingface.co/docs/inference-providers/providers/replicate.md

### Template

If you want to update the content related to replicate's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/replicate.handlebars`.

### Logos

If you want to update replicate's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `replicate-light.png` and `replicate-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Replicate

> [!TIP]
> All supported Replicate models can be found [here](https://huggingface.co/models?inference_provider=replicate&sort=trending)

    
        
        
    

    
        
        
    

Replicate is building tools so all software engineers can use AI as if it were normal software. You should be able to import an image generator the same way you import an npm package. You should be able to customize a model as easily as you can fork something on GitHub.

## Supported tasks

### Automatic Speech Recognition

Find out more about Automatic Speech Recognition [here](../tasks/automatic-speech-recognition).

<InferenceSnippet
    pipeline=automatic-speech-recognition
    providersMapping={ {"replicate":{"modelId":"openai/whisper-large-v3","providerModelId":"openai/whisper:8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e"} } }
/>

### Image To Image

Find out more about Image To Image [here](../tasks/image-to-image).

<InferenceSnippet
    pipeline=image-to-image
    providersMapping={ {"replicate":{"modelId":"black-forest-labs/FLUX.2-klein-9B","providerModelId":"black-forest-labs/flux-2-klein-9b"} } }
/>

### Text To Image

Find out more about Text To Image [here](../tasks/text-to-image).

<InferenceSnippet
    pipeline=text-to-image
    providersMapping={ {"replicate":{"modelId":"black-forest-labs/FLUX.1-dev","providerModelId":"black-forest-labs/flux-dev"} } }
/>

### Text To Video

Find out more about Text To Video [here](../tasks/text-to-video).

<InferenceSnippet
    pipeline=text-to-video
    providersMapping={ {"replicate":{"modelId":"Wan-AI/Wan2.2-TI2V-5B","providerModelId":"wan-video/wan-2.2-5b-fast"} } }
/>

### Template
https://huggingface.co/docs/inference-providers/providers/groq.md

### Template

If you want to update the content related to groq's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/groq.handlebars`.

### Logos

If you want to update groq's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `groq-light.png` and `groq-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Groq

> [!TIP]
> All supported Groq models can be found [here](https://huggingface.co/models?inference_provider=groq&sort=trending)

    
        
        
    

    
        
        
    

Groq is fast AI inference. Their groundbreaking LPU technology delivers record-setting performance and efficiency for GenAI models. With custom chips specifically designed for AI inference workloads and a deterministic, software-first approach, Groq eliminates the bottlenecks of conventional hardware to enable real-time AI applications with predictable latency and exceptional throughput so developers can build fast.

For latest pricing, visit our [pricing page](https://groq.com/pricing/).

## Resources
 - **Website**: https://groq.com/
 - **Documentation**: https://console.groq.com/docs
 - **Community Forum**: https://community.groq.com/
 - **X**: [@GroqInc](https://x.com/GroqInc)
 - **LinkedIn**: [Groq](https://www.linkedin.com/company/groq/)
 - **YouTube**: [Groq](https://www.youtube.com/@GroqInc)

## Supported tasks

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"groq":{"modelId":"openai/gpt-oss-120b","providerModelId":"openai/gpt-oss-120b"} } }
conversational />

### Template
https://huggingface.co/docs/inference-providers/providers/zai-org.md

### Template

If you want to update the content related to zai-org's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/zai-org.handlebars`.

### Logos

If you want to update zai-org's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `zai-org-light.png` and `zai-org-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Z.ai

> [!TIP]
> All supported Z.ai models can be found [here](https://huggingface.co/models?inference_provider=zai-org&sort=trending)

    
        
        
    

    
        
        
    

Z.ai is an AI platform that provides cutting-edge large language models powered by GLM series. Their flagship models feature Mixture-of-Experts (MoE) architecture with advanced reasoning, coding, and agentic capabilities.

For latest pricing, visit the [pricing page](https://docs.z.ai/guides/overview/pricing).

## Resources
 - **Website**: https://z.ai/
 - **Documentation**: https://docs.z.ai/
 - **API Documentation**: https://docs.z.ai/api-reference/introduction
 - **GitHub**: https://github.com/zai-org
 - **Hugging Face**: https://huggingface.co/zai-org

## Supported tasks

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"zai-org":{"modelId":"zai-org/GLM-5.3","providerModelId":"glm-5.3"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"zai-org":{"modelId":"zai-org/GLM-5.3-Flash","providerModelId":"glm-5.3-flash"} } }
conversational />

### Template
https://huggingface.co/docs/inference-providers/providers/baseten.md

### Template

If you want to update the content related to baseten's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/baseten.handlebars`.

### Logos

If you want to update baseten's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `baseten-light.png` and `baseten-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Baseten

> [!TIP]
> All supported Baseten models can be found [here](https://huggingface.co/models?inference_provider=baseten&sort=trending)

    
        
        
    

    
        
        
    

Baseten provides on-demand frontier model APIs designed for production applications, not just experimentation. Built on the Baseten Inference Stack, these APIs deliver enterprise-grade performance and reliability with optimized inference for leading open-source models.

## Supported tasks

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"baseten":{"modelId":"zai-org/GLM-5.3","providerModelId":"zai-org/GLM-5.3"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"baseten":{"modelId":"deepseek-ai/DeepSeek-V4.1-Flash-1017","providerModelId":"deepseek-ai/DeepSeek-V4.1-Flash-1017"} } }
conversational />

### Template
https://huggingface.co/docs/inference-providers/providers/featherless-ai.md

### Template

If you want to update the content related to featherless-ai's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/featherless-ai.handlebars`.

### Logos

If you want to update featherless-ai's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `featherless-ai-light.png` and `featherless-ai-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Featherless AI

> [!TIP]
> All supported Featherless AI models can be found [here](https://huggingface.co/models?inference_provider=featherless-ai&sort=trending)

    
        
        
    

    
        
        
    

[Featherless AI](https://featherless.ai) is a serverless AI inference platform that offers access to thousands of open-source models. 

Our goal is to make all AI models available for serverless inference. We provide inference via API to a continually expanding library of open-weight models.

## Supported tasks

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"featherless-ai":{"modelId":"zai-org/GLM-5.3","providerModelId":"zai-org/GLM-5.3"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"featherless-ai":{"modelId":"Qwen/Qwen3.8-27B","providerModelId":"Qwen/Qwen3.8-27B"} } }
conversational />

### Template
https://huggingface.co/docs/inference-providers/providers/fireworks-ai.md

### Template

If you want to update the content related to fireworks-ai's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/fireworks-ai.handlebars`.

### Logos

If you want to update fireworks-ai's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `fireworks-ai-light.png` and `fireworks-ai-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Fireworks AI

> [!TIP]
> All supported Fireworks AI models can be found [here](https://huggingface.co/models?inference_provider=fireworks-ai&sort=trending)

    
        
        
    

    
        
        
    

Fireworks AI is a developer-centric platform that delivers high-performance generative AI solutions, enabling efficient deployment and fine-tuning of large language models (LLMs) and image models.
## Supported tasks

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"fireworks-ai":{"modelId":"zai-org/GLM-5.3","providerModelId":"accounts/fireworks/models/glm-5p3"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"fireworks-ai":{"modelId":"deepseek-ai/DeepSeek-V4.1-Flash","providerModelId":"accounts/fireworks/models/deepseek-v4p1-flash"} } }
conversational />

### Template
https://huggingface.co/docs/inference-providers/providers/scaleway.md

### Template

If you want to update the content related to scaleway's description, please edit the template file under `https://github.com/huggingface/hub-docs/tree/main/scripts/inference-providers/templates/providers/scaleway.handlebars`.

### Logos

If you want to update scaleway's logo, upload a file by opening a PR on https://huggingface.co/datasets/huggingface/documentation-images/tree/main/inference-providers/logos. Ping @wauplin and @celinah on the PR to let them know you uploaded a new logo.
Logos must be in .png format and be named `scaleway-light.png` and `scaleway-dark.png`. Visit https://huggingface.co/settings/theme to switch between light and dark mode and check that the logos are displayed correctly.

### Generation script

For more details, check out the `generate.ts` script: https://github.com/huggingface/hub-docs/blob/main/scripts/inference-providers/scripts/generate.ts.
--->

# Scaleway

> [!TIP]
> All supported Scaleway models can be found [here](https://huggingface.co/models?inference_provider=scaleway&sort=trending)

    
        
        
    

    
        
        
    

Scaleway is a European cloud provider, serving latest LLM models through its [Generative APIs](https://www.scaleway.com/en/generative-apis/) alongside a complete cloud ecosystem.

## Supported tasks

### Chat Completion (LLM)

Find out more about Chat Completion (LLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=text-generation
    providersMapping={ {"scaleway":{"modelId":"openai/gpt-oss-120b","providerModelId":"gpt-oss-120b"} } }
conversational />

### Chat Completion (VLM)

Find out more about Chat Completion (VLM) [here](../tasks/chat-completion).

<InferenceSnippet
    pipeline=image-text-to-text
    providersMapping={ {"scaleway":{"modelId":"Qwen/Qwen3.6-35B-A3B","providerModelId":"qwen3.6-35b-a3b"} } }
conversational />

### Feature Extraction

Find out more about Feature Extraction [here](../tasks/feature-extraction).

<InferenceSnippet
    pipeline=feature-extraction
    providersMapping={ {"scaleway":{"modelId":"Qwen/Qwen3-Embedding-8B","providerModelId":"qwen3-embedding-8b"} } }
/>

### API Reference
https://huggingface.co/docs/inference-providers/tasks/index.md
