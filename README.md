updated 2020/3/8

## Re-implement random walk models. 

The dataset split and evaluate is consistent with other method

## Re-implement GCN models

The code is from the repo `Intellifusion-graph` which is a tensorflow version (dgl and pyg cannot obtain normal output).
The epoch in different models is adjusted to the same value.
The dataset split was also adjusted as the same method. Others is same with the original version.

Notes: The epoch in GAE is 200, and other model_1/2/3/4 is 2000. When increasing training epoch in GAE, the performance drop large
Also, decresing the epoch in model_1/2/3/4 their performance are also unsatisfied.

Besides, the learning rate in GAE is 0.01 while in other four models is 0.0005

Random walk methods are same with my previous implementation and difference is in the evaluation method

The variance is a little big, the following results just as reference

![Recall out in citeseer](./citeseer_recall_out.png)
![Recall in in citeseer](./citeseer_recall_in.png)
![Recall out in cora](./cora_recall_out.png)
![Recall in in cora](./cora_recall_in.png)

 

