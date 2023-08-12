# model-serving-examples using flask framework

**[GCP Documentation](https://cloud.google.com/appengine/docs)** 
| **[Pricing](https://cloud.google.com/appengine/pricing)** 
| **[Quotas](https://cloud.google.com/appengine/quotas)** 


## Deploy a Flexible Flask Python App

Provide code with a yaml file called `app.yaml`, for example:

```yaml
runtime: python
env: flex
entrypoint: gunicorn -b :$PORT main:app

runtime_config:
  python_version: 3

manual_scaling:
  instances: 1
resources:
  cpu: 1
  memory_gb: 4
  disk_size_gb: 20
```

Include a `main.py` file as the execution starting call.

For required dependencies, all libraries need to be specified in the `requirements.txt` file.

## Deploy and View App Logs

You can use gcloud function to deploy the app. 


Make sure you have [`gcloud` sdk](https://cloud.google.com/sdk/docs/quickstarts) is installed. 

`cd` to the application directory (the one that `main.py` and `app.yaml` files are created in), and run the following command to deploy the app:

```
gcloud app deploy 
```

After the app is deployed, you can view app log locally using command:

```
gcloud app logs tail
```

## Standard

Application instances run in a sandbox and need to deal with rapid scaling.

Supported Python: 2.7/3.7/3.8.

Pricing scheme: pay only for what you need and when you need it. For example, it can scale to 0 instances when there is no traffic.

#### Instance Classes

| Instance Class	| Memory Limit	| CPU Limit	| Supported Scaling Types |
| -----|----- |----- |----- |
| F1 (default)	| 256 MB	| 600 MHz	| automatic |
| F2	| 512 MB	| 1.2 GHz	| automatic |
| F4	| 1024 MB	| 2.4 GHz	| automatic |
| F4_1G	| 2048 MB	| 2.4 GHz	| automatic |
| B1	| 256 MB	| 600 MHz	| manual, basic |
| B2 (default)	| 512 MB	| 1.2 GHz	| manual, basic |
| B4	| 1024 MB	| 2.4 GHz	| manual, basic |
| B4_1G	| 2048 MB	| 2.4 GHz	| manual, basic |
| B8	| 2048 MB	| 4.8 GHz	| manual, basic |

#### Pricing 

For London(europe-west2)

| Instance class	| Cost per hour per instance
| -----|----- |
| B1	| $0.06|
| B2	| $0.12|
| B4	| $0.24|
| B4_1G	| $0.36|
| B8	| $0.48|
| F1	| $0.06|
| F2	| $0.12|
| F4	| $0.24|
| F4_1G	| $0.36|

#### Quotas

The App Engine standard environment gives you 1 GB of data storage and traffic for free, which can be increased by enabling paid applications. 

## Flexible

Application instances run within Docker containers on Compute Engine virtual machines (VM).

Applications that receive **consistent traffic**, experience regular traffic fluctuations, or meet the parameters for scaling up and down gradually.

Runs in a Docker container that includes a custom runtime or source code written in other programming languages.


#### Pricing 

For London(europe-west2)

| Resource	| Unit	| Unit cost |
| -----|----- |----- |
| vCPU	| per core hour	| $0.063 |
| Memory	| per GB hour	| $0.009 |
| Persistent disk	| Priced as Compute Engine persistent disk, which is called "Storage PD Capacity" on your bill. | |
| Outgoing network traffic	| Priced as Compute Engine internet egress. | |
| Incoming network traffic	| Gigabytes	| Free |


##### Netword pricing:

| Resource	| Unit	| Unit cost (in US $) |
| -----|----- |----- |
| Outgoing network traffic	| Gigabytes	| $0.12 |
| Incoming network traffic	| Gigabytes	| Free |

#### Quotas

##### Cloud Storage:

| Resource	| Default limit |
| -----|----- |
| Default Cloud Storage Bucket Stored Data	| irst 5 GB free; no maximum |
| Default Cloud Storage Bucket Class A Operations	| First 20,000 ops/day free; no maximum |
| Default Cloud Storage Bucket Class B Operations	| First 50,000 ops/day free; no maximum |
| Default Cloud Storage Bucket Network Egress	| Up to the Outgoing Bandwidth quota free; no maximum |

##### Code storage:

| Resource	| Cost |
| -----|----- |
| Code & Static Data Storage - First 1 GB	| Free |
| Code & Static Data Storage - Exceeding 1 GB	| $0.026/GB/month |

##### Deployments:

In each App Engine application, you can deploy up to 10,000 times per day.

##### Requests:

| Resource	| Daily limit	| Maximum rate |
| -----|----- |----- |
| Outgoing bandwidth (billable, includes HTTPS)	| 1 GB free; 14,400 GB maximum	| 10 GB/minute |
| Incoming bandwidth (includes HTTPS)	| None	| None |

## Comparison of Standard and Flexible

| Feature	| Standard environment	| Flexible environment|
| -----|----- |----- |
| Instance startup time	| Seconds	| Minutes|
| Maximum request timeout	| Depends on the runtime and type of scaling.	| 60 minutes|
| Scaling	| Manual, Basic, Automatic	| Manual, Automatic|
| Scale to zero	| Yes	| No, minimum 1 instance|
| Writing to local disk	| Python 3 have read and write access to the `/tmp` directory. <br>Python 2.7 don't have write access to the disk. | Yes, ephemeral (disk initialized on each VM startup)|
| Modifying the runtime	| No	| Yes (through Dockerfile)|
| Deployment time	| Seconds	| Minutes|
| Automatic in-place security patches	| Yes	| Yes (excludes container image runtime)|
| Location	| North America, Asia Pacific, or Europe	| North America, Asia Pacific, or Europe|

