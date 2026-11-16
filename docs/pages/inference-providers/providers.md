# Providers

Inference Providers sends each request to a provider serving the model. You pay that provider's price with no markup, and you can turn off the providers listed on this page in your [settings](https://huggingface.co/settings/inference-providers).

## Language models

Chat, vision and agent requests go through [OpenRouter](https://openrouter.ai), which routes each one to a provider serving the model, including the ones below. See [Routing and Providers](./routing) to choose how a provider is picked. To send a request to one provider only, append the suffix shown on its card to the model id, for example `deepseek-ai/DeepSeek-V4.1-Flash:together`.

Baseten:baseten · On the Hub · Privacy policy

Cerebras:cerebras · On the Hub · Privacy policy

Cohere:cohere · On the Hub · Privacy policy

CoreWeave:coreweave · On the Hub · Privacy policy

DeepInfra:deepinfra · On the Hub · Privacy policy

DigitalOcean:digitalocean · On the Hub · Privacy policy

Featherless AI:featherless · On the Hub · Privacy policy

Fireworks:fireworks · On the Hub · Privacy policy

Liquid AI:liquid · On the Hub · Privacy policy

Mistral:mistral · On the Hub · Privacy policy

Moonshot AI:moonshotai · On the Hub · Privacy policy

Nebius:nebius · On the Hub · Privacy policy

Novita:novita · On the Hub · Privacy policy

Parasail:parasail · On the Hub · Privacy policy

Prime Intellect:primeintellect · On the Hub

Reka:reka · On the Hub · Privacy policy

Tencent:tencent · On the Hub · Privacy policy

Together AI:together · On the Hub · Privacy policy

Zai:z-ai · On the Hub · Privacy policy

### Which providers serve a language model

A model's page on the Hub lists its providers. To compare their price, speed and context length, use the [models page](https://huggingface.co/inference/models) or `GET https://router.huggingface.co/v1/models` (see [Hub API](./hub-api#list-openai-compatible-models)).

## Image, video, audio, embeddings and more

Other tasks go to the following providers through the [Hugging Face SDKs](./tasks/index). Each link lists the models a provider serves on the Hub. To pick a provider, pass the id shown on its card as `provider`. The list of providers is also available on each model's page.

falfal-ai · Image, video and speech · On the Hub

HF Inference APIhf-inference · Embeddings, classification and other tasks · On the Hub

Replicatereplicate · Image, video and speech · On the Hub

WaveSpeedwavespeed · Image and video · On the Hub

### Prompt Caching
https://huggingface.co/docs/inference-providers/prompt-caching.md
