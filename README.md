# CrowdKit
CrowdKit - Exploring and Exploiting the Wisdom of Crowds.  

CrowdKit is a powerful toolkit for knowledge learning with crowdsourcing. Comparing with machine learning, knowledge learning extends its connotation to the whole process of knowledge discovery and utilization.

---

## Label Aggregation (Truth Inference)
### Single Label / Choice
1. **Marjority Voting (MV)** [Ceka]
2. **Dawid & Skene's (DS)** [Ceka]
3. **GLAD** [Ceka, by Whitehill, CPP source code and exe file are in ZoneEx]
4. **Karger, Oh, & Shah's (KOS)**[Ceka]
5. **Raykar, Yu, et al.'s (RY)** [Ceka, by Sheshadri in [SQUARE](http://ir.ischool.utexas.edu/square/)]
6. **ZenCrowd** [Ceka, by Sheshadri in [SQUARE](http://ir.ischool.utexas.edu/square/)]
7. **Iterative Weigted Marjority Voting (IWMV)** [Ceka]
8. **Ground Truth Inferece by Clustering (GTIC)** [Ceka]
9. **Positive LAbel freqency Thresholding (PLAT)** [Ceka]
10. **WTCMM** [Ceka]
### Muli-Label
11. **Multi-Class Multi-Label Independent (MCMLI)** [InferenceML]
12. **Multi-Class Multi-Label Dependent (MCMLD)** [InferenceML]
13. **Multi-Class Multi-Label Independent One-Coin model (MCMLI-OC)** [InferenceML]
14. **Multi-Class Multi-Label Dependent One-Coin model (MCMLD-OC)** [InferenceML]
15. **Majority Voting (MV, extended to multi-label scenario)** [InferenceML]
16. **Dawid and Skene's model (DS, extended to mulit-label scenario)** [InferenceML]
17. **Independent Bayesian Classifier Combination (iBCC, extended to mulit-label scenario)** [InferenceML]
18. **Multi-Class One-Coin model (MCOC, extended to multi-label scenario)** [InferenceML]

---

>## Ceka-1.0.2 *[Independent Project]*
>>Ceka-1.0.2 refers to Crowd Environment and its Knowledge Analysis, which is a toolkit for learning with crowdsourcing written in Java, developed by Jing Zhang et al. It was originally hosted at https://sourceforge.net/projects/ceka/. We moved it to Github and the code at SourceForge was no longer maintained.

>>The usages and development of Ceka can be found in [CekaProgrammingGuide-1.0.pdf](https://github.com/wisdomofcrowds/CrowdKit/blob/main/docs/CekaProgrammingGuide-1.0.pdf). Although the code of Ceka is recommended to work with [Java-15](https://www.oracle.com/java/technologies/javase-jdk15-downloads.html) and [Weka-3.8.5](https://prdownloads.sourceforge.net/weka/weka-3-8-5.zip), the CekaProgrammingGuide is still a good reference that can help set up an Eclipse IDE development environment.

>>Ceka has its own licenses.

---

>## InferenceML-1.0.1 *[Independent Project]*
>>InferenceML includes some multi-label inference models for multi-label annotation tasks in crowdsourcing. It was orignally developped by Jing Zhang. For more information about the models, refer to [Multi-Label Truth Inference for Crowdsourcing Using Mixture Models](https://github.com/wisdomofcrowds/CrowdKit/blob/main/Docs/TKDE2019zhang.pdf).
>>>### Input Files and Their Formats
>>>1. **.resp** file: the file containing the responses of crowdsourced workers to the questions
>>>>- format for *single-label* `worker_id` `instance_id` `label_value`
>>>>- format for *multi-label* &nbsp; `worker_id` `instance_id` `label_id` `label_value`
>>>2. **.gold** file: the file containing the ground truth for performance evaluation
>>>>- format for *single-label* `instance_id` `label_value`
>>>>- format for *multi-label*  &nbsp; `instance_id` `label_id` `label_value`
>>>### Identifiers for Instances, Workers, Labels and Class Values
>>>>All internal identifiers for instances, workers, labels and class values start from one. Zero means an empty annotation.

---

>## ZoneEX (Zone External) *[Independent Projects]*
>>We have collected some code written by other researchers, which is put in directory [ZoneEx](https://github.com/wisdomofcrowds/CrowdKit/tree/main/ZoneEx). We do not guarantee that these programs will work well.