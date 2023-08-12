# Creating A Redis MemoryStore

Whenever we create a Memorystore instance, it will be given a private IP address that will be visible to a single VPC network. Further, Memorystore only permits connections from resources that are contained within the same region. 

To access Memorystore from Cloud Run requires some additional steps. This is because Cloud Run is a managed serverless environment and isn’t associated with any specific VPCs such as the VPC with which Memorystore is peered. The solution to this is to use serverless VPC access.

We create a Serverless VPC Access connector in the same region as both Cloud Run and Memorstore and associated with the same VPC as Memorystore. With this in place, we can then instruct Cloud Run to leverage that connector to access Memorystore.


To create a memory store

```
gcloud redis instances create redis1 --region europe-west2
```


When the instance has been created, we will want to make note of the IP address that was allocated for us.
```
gcloud redis instances describe redis1 --region europe-west2
```

```

authorizedNetwork: projects/gft-bike-insurance/global/networks/default
connectMode: DIRECT_PEERING
createTime: '2021-02-23T09:22:12.615710780Z'
currentLocationId: europe-west2-b
>>> host: 10.105.53.19
locationId: europe-west2-b
memorySizeGb: 1
name: projects/gft-bike-insurance/locations/europe-west2/instances/redis1
persistenceIamIdentity: serviceAccount:304300108037-compute@developer.gserviceaccount.com
port: 6379
redisVersion: REDIS_4_0
reservedIpRange: 10.105.53.16/29
state: READY
tier: BASIC
transitEncryptionMode: DISABLED
```

Now we can create the VPC access connector:

```
gcloud compute networks vpc-access connectors create my-vpc-connector --network default --region europe-west2 --range 10.8.0.0/28
```


This will create an access connector called “my-vpc-connector” associated with the default VPC network and the us-central1 region. The IP addresses for the connector will be taken from the CIDR range 10.8.0.0/28.


To create a Cloud Run you can use this command:

```
gcloud run deploy cloud-run-app \
  --image gcr.io/[PROJECT_ID]/cloud-run-app \
  --max-instances 1 \
  --platform managed \
  --region us-central1 \
  --vpc-connector my-vpc-connector \
  --allow-unauthenticated \
  --set-env-vars "REDIS_IP=[IP_ADDRESS]"

```