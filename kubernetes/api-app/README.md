# model-serving-examples using Kubernetes engine

Please use the set of commands in commands_cheet_sheet.txt file for
 deployment of this code. 
 
 You can also follow the instruction in 
 'https://cloud.google.com/kubernetes-engine/docs/quickstarts/deploying-a-language-specific-app#python' and code
  in this repo to install the model prediction api in GKE.
  
 Using curl or python the API can be called.
 
 ```
curl 35.246.50.251/score     --request POST     --header "Content-Type: application/json"     --data '{"label": ["unhappy","happy","posit
ive","negative","neutral"],"sentence": "Hi, I had very good experience in using the service.I have recently made use of Virtual Shopping to help me to rearrang home deliv
ery for my order.Because I want to change the delivery date, can you share the instructions.Kindly regards, Lu"}'

{"score":"happy \t similarity: 0.13\npositive \t similarity: 0.04\nunhappy \t similarity: -0.06\nneutral \t similarity: -0.12\nnegative \t similarity: -0.17\n"}
```
  
 