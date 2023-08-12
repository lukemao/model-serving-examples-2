# model-serving-examples using GCP cloud function 

**[GCP Documentation](https://cloud.google.com/functions/docs)** 
| **[Pricing](https://cloud.google.com/functions/pricing)** 
| **[Quotas](https://cloud.google.com/functions/quotas)**

## Cloud Functions

An event-driven compute platform to easily connect and extend Google and third-party cloud services and build applications that scale from zero to planet scale.

- Fully managed, serverless, 
- Automatically scale based on the load
- Pay only for what you use
- Supported Python versions: 3.7/3.8

### Pricing 

According to: 
- how long your function runs 
- how many times it's invoked 
- and how many resources you provision for the function

#### Invocations pricing

Function invocations are charged at a flat rate.

| Invocations per Month	| Price/Million|
| -----|:-----: |
|First 2 million	| Free |
|Beyond 2 million	| $0.40 |

#### Compute Time pricing

Compute time is measured from the time your function receives a request to the time it completes.

Cloud Functions can be provisioned as one of 5 types:

| Memory	|CPU	|Price/100ms (Tier 1 Price) |
| :-----:|:-----: |:--------------: |
| 128MB	| 200MHz	| $0.000000231 |
| 256MB	| 400MHz	| $0.000000463 |
| 512MB	| 800MHz	| $0.000000925 |
| 1024MB	| 1.4 GHz	| $0.000001650 |
| 2048MB	| 2.4 GHz	| $0.000002900 |

Tier 1 regions in Europe: 
- europe-west1 (Belgium)
- europe-west2 (London)

#### Networking pricing 

| Type	| Price/GB |
| -----|:-----: |
| Outbound Data (Egress)	| $0.12 |
| Outbound Data per month	| 5GB Free |
| Inbound Data (Ingress)	| Free |
| Outbound Data to Google APIs in the same region	| Free |


### Quotas & Limits

| Quota	| Description	| Limit	| Can be increased	| Scope |
| -----|------------------------------ |:------------------------------: |:-----: |:-----: |
| Number of functions	| The total number of functions that can be deployed per project	| 1,000	| No	| per project |
| Max deployment size	| The maximum size of a single function deployment	| 100MB (compressed) for sources. <br>500MB (uncompressed) for <br>sources plus modules.	| No	| per function |
| Max uncompressed HTTP request size	| Data sent to HTTP Functions in an HTTP request	| 10MB	| No	| per invocation |
| Max uncompressed HTTP response size	| Data sent from HTTP functions in an HTTP response	| 10MB	| No	| per invocation |
| Max event size for background functions	| Data sent in events to background functions	| 10MB	| No	| per event |
| Max function memory	| Amount of memory a function can use	| 2048MB	| No	| per function |