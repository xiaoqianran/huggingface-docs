# Analytics and Metrics

The Analytics page is like the control center for your deployed models. It tells you in real-time what's going on, how many users are
calling your models, about hardware usage, latencies, and much more. In this documentation we'll dive into what each metric means and
how to analyze the graphs.

![intro](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/analytics/1-intro.png)

In the top bar, you can configure the high level view:

- Which replica to view metrics from: either an individual replica or all.
- If you want to view metrics related to requests, hardware, or timeline of replicas.
- Which time frame you'll inspect the metrics, and this setting affects all graphs on the page. You can choose between any of the existing settings from the dropdown, or click-and-drag over any graph for a custom timeframe. You can also enable/disable
auto refresh or view the metrics per replica or all.

![config](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/analytics/2-config.png)

## Understanding the graphs

### Number of (HTTP) Requests 

The first graph at the top left shows you how many requests your Inference Endpoint has received. By default they are grouped by HTTP response
classes, but by switching the toggle you can view them by individual status. As a reminder the HTTP response classes are:

- **Informational responses (100-199)**: The server has received your request and is working on it. For example, `102 Processing` means the server is still handling your request.
- **Successful responses (200-299)**: Your request was received and completed successfully. For example, `200 OK` means everything worked as expected.
- **Redirection messages (300-399)**: The server is telling your client to look somewhere else for the information or to take another action. For example, `301 Moved Permanently` means the resource has a new address.
- **Client error responses (400-499)**: There was a problem with the request sent by your client (like a typo in the URL or missing data). For example, `404 Not Found` means the server couldn't find what you asked for.
- **Server error responses (500-599)**: The server ran into an issue while trying to process your request. For example, `502 Bad Gateway` means the server got an invalid response from another server it tried to contact.

We recommend checking the [MDN web docs](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status) for more information on individual
status codes.

The boxes above the graph also show the % of requests in the respective response class.

![http](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/analytics/3-http-reqs.png)

### Pending Requests

Pending requests are requests that have not yet received an HTTP status, meaning they include in-flight requests and requests currently
being processed. If this metric increases too much, it means that your requests are queuing up, and your users have to wait for requests
to finish. In this case you should consider increasing your number of replicas or alternatively use autoscaling, you can read more about
it in the [autoscaling guide](./autoscaling#scalingbasedonpendingrequests(betafeature))

![pending](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/analytics/4-pending-reqs.png)

### Latency Distribution

From this graph you'll be able to see how long it takes for your Inference Endpoint to generate a response. Latency is reported as:

- **p99**: meaning that 99% of all requests were faster than this value
- **p95**: meaning that 95% of all requests were faster than this value
- **p90**: meaning that 90% of all requests were faster than this value
- **median**: meaning that 50% of all requests were faster than this value

Usually a good metric is also to look at how big the difference is between the median and p99. The closer the values are to each other, the more
uniform the latency is, whereas if the difference is large, it means that the users of your Inference Endpoint have in general a fast response but
the worst case latencies can be long.

![latency](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/analytics/5-latency.png)

### Running Replicas

In the running replica graph, you'll see how many running replicas you have during a point in time. The red line shows
your current maximum replicas setting. 

![status](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/analytics/6-running.png)

For a more advanced view of different statuses for individual replicas, going from *pending* all the way
to *running*, you can toggle to the Timeline section. This is very useful to get a sense of how long it takes an Endpoint to become ready to serve requests.

![advanced](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/analytics/7-timeline.png)

### Compute 

These four graphs are dedicated to hardware usage. You'll find:

- CPU usage: How much processing power is being used.
- Memory usage: How much RAM is being used.
- GPU usage: How much of the GPU's processing power is being used.
- GPU Memory (VRAM) usage: How much GPU memory is being used.

![usage](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/analytics/8-usage.png)

By toggling "details" you can either view the average or per replica value for the metric in question.

If you have autoscaling based on hardware utilization enabled, these are the metrics that determine your autoscaling behaviour. You can
read more about autoscaling [here](./autoscaling#scalingbasedonhardwareutilization)

## Create an integration with the Inference Endpoints OpenMetrics API

**This feature is currently in Beta. You will need to be subscribed to [Team or Enterprise](https://huggingface.co/pricing) to take advantage of this feature.**

You can export real-time metrics from your Inference Endpoints into your own monitoring stack. The Metrics API exposes metrics in the OpenMetrics format, which is widely supported by observability tools such as Prometheus, Grafana, and Datadog.

This allows you to monitor in near real-time:
- Requests grouped by replica
- Latency distributions (p50, p95, etc.)
- Hardware metrics (CPU, GPU, memory, accelerator utilization)

### Query metrics manually

You can use `curl` to query the metrics endpoint directly and inspect the raw data:
```bash
curl -X GET "https://api.endpoints.huggingface.cloud/v2/endpoint/{namespace}/{endpoint-name}/open-metrics" \
  -H "Authorization: Bearer YOUR_AUTH_TOKEN"
```

This will return metrics in OpenMetrics text format:
```bash
# HELP latency_distribution Latency distribution
# TYPE latency_distribution summary
latency_distribution{quantile="0.5"} 0.006339203
latency_distribution{quantile="0.9"} 0.007574241
latency_distribution{quantile="0.95"} 0.007994495
latency_distribution{quantile="0.99"} 0.020140918
latency_distribution_count 4
latency_distribution_sum 0.042048857
# HELP http_requests HTTP requests by code and replicas
# TYPE http_requests counter
http_requests{replica_id="fqwg7eri-hskoj",status_code="200"} 1152
http_requests{replica_id="q9cv26ut-3vo4s",status_code="200"} 1
# HELP cpu_usage_percent CPU percent
# TYPE cpu_usage_percent gauge
# UNIT cpu_usage_percent percent
```

### Connect with your observability tools

OpenMetrics is widely supported across monitoring ecosystems. A few common options:
- [Datadog OpenMetrics integration](https://docs.datadoghq.com/integrations/openmetrics/)
- [Grafana Prometheus datasource](https://tinyurl.com/e4fypk5m)

From there, you can set up dashboards, alerts, and reports to monitor endpoint performance.

### Subscribe to Team or Enterprise

Your organization can sign up for the Team or Enterprise plan [here](https://huggingface.co/enterprise?subscribe=true) 🚀 
For any questions or feature requests, please email us at api-enterprise@huggingface.co

### Configuration
https://huggingface.co/docs/inference-endpoints/guides/configuration.md

# Configuration

This section describes the configuration options available when creating a new inference endpoint. Each section of
the interface allows fine-grained control over how the model is deployed, accessed, and scaled.

## Endpoint name, model and organization

In the top left you can:
- change the name of the inference endpoint
- verify to which organization you're deploying this model
- verify which model you are deploying
- and which Hugging Face Hub repo you are deploying this model from

![name-org-model](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/configuration/1-name-org-model.png)

## Hardware Configuration
The Hardware Configuration section allows you to choose the compute backend used to host the model.
You can select from three major cloud providers:
- Amazon Web Services (AWS)
- Microsoft Azure
- Google Cloud Platform

![hardware](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/configuration/2-hardware.png)

You must also choose an accelerator type:
- CPU
- GPU
- INF2 (AWS Inferentia)

Additionally, you can select the deployment region (e.g., East US) using the dropdown menu. Once the
provider, accelerator, and region are chosen, a list of available instance types is displayed. Each instance tile includes:

- GPU Type and Count
- Memory (e.g., 48 GB)
- vCPUs and RAM
- Hourly Pricing (e.g., $1.80 / h)

You can select a tile to choose that instance type for your deployment. Instances that are incompatible or unavailable in the
selected region are grayed out and unclickable.

## Authentication

This section determines who can access your deployed endpoint. Available options are:
- **Private (default)**: Accessible only to you, or members of your Hugging Face organization, using a personal HF access token.
- **Public**: Anyone can access your endpoint, without authentication.
- **Authenticated**: Anyone with a Hugging Face account can access it, using their personal HF access tokens.

Additionally, if you deploy your Inference Endpoint in AWS, you can use **AWS PrivateLink** for an intra-region secured connection to your AWS VPN.

![auth](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/configuration/11-auth.png)

## Autoscaling

The Autoscaling section configures how many replicas of your model run and whether the system scales down to zero during periods of inactivity. For more
information we recommend reading the [in-depth guide on autoscaling](./autoscaling).

![autoscaling](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/configuration/4-autoscaling.png)

- **Automatic Scale-to-Zero**: A dropdown lets you choose how long the system should wait after the last request before
scaling down to zero. Default is after 1 hour with no activity.
- **Number of Replicas**:
    - Min: Minimum number of replicas to keep running. Note that enabling automatic scale-to-zero requires setting this to 0.
    - Max: Maximum number of replicas allowed (e.g., 1)
- **Autoscaling strategy**:
    - Based on hardware usage: For example, a scale up will be triggered if the average hardware utilisation (%) exceeds this threshold for more than 20 seconds.
    - Pending requests: A scale up event will be triggered if the average number of pending requests exceeds this threshold for more than 20 seconds.

## Inference Engine Configuration
This section allows you to specify how the container hosting your model behaves. This setting depends on the selected inference engine.
For configuration details, please read the Inference Engine section.
![inference-engine](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/configuration/9-inference-engine.png)

## Container Configuration
Here you can edit the container arguments and container command.
![container-configs](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/configuration/8-container-config.png)

## Environment Variables
Environment variables can be provided to customize container behavior or pass secrets.
- **Default Env**: Key-value pairs passed as plain environment variables.
- **Secret Env**: Key-value pairs stored securely and injected at runtime.

Each section allows you to add multiple entries using the Add button.

![env-vars](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/configuration/5-env-vars.png)

## Endpoint Tags
You can label endpoints with tags (e.g., for-testing) to help organize and manage deployments across environments or teams. In the dashboard
you will be able to filter and sort endpoints based on these tags.
Tags are plain text labels added via the Add button.

![tags](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/configuration/6-tags.png)

## Network
This section determines from where your deployed endpoint can be accessed. 

By default, your endpoint is accessible from the Internet, and secured with TLS/SSL. Endpoints deployed on an AWS instance can use AWS PrivateLink to restrict access to a specific VPC.

To configure it you need to:
1. check the box to activate AWS PrivateLink for your endpoint.
2. Add your AWS Account ID: You need to provide the AWS ID of the account that owns the VPC you want to restrict access to.

Optionally you can enable PrivateLink Sharing. This will enable sharing of the same PrivateLink between different endpoints.

![network](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/configuration/10-network.png)

## Advanced Settings
Advanced Settings offer more fine-grained control over deployment.

![advanced](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/configuration/7-advanced.png)

- **Commit Revision**: Optionally specify a commit hash to which revision of the model repository on the Hugging Face Hub
you want to download the model artifacts from
- **Task**: Defines the type of model task. This is usually inferred from the model repository.
- **Container Arguments**: Pass CLI-style arguments to the container entrypoint.
- **Container Command**: Override the container entrypoint entirely.
- **Download Pattern**: Defines which model files are downloaded.

### Security & Compliance
https://huggingface.co/docs/inference-endpoints/guides/security.md
