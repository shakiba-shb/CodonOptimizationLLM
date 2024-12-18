One of the main challenges in mRNA therapeutics is designing an optimized mRNA sequence with structural stability and optimized codon usage. A single protein sequence can be translated from an astronomically large number of possible mRNA sequences due to synonymous codons, and this vast combinatorial space makes identifying the ideal mRNA sequence both complex and difficult. Here, we use large language models to find th optimized codon combination for a protein. 

![alt text](https://github.com/shakiba-shb/CodonOptimizationLLM/blob/main/images/Untitled.png)

We used CodonBERT, a large language model pre-trained on 10 million mRNA sequences. CodonBERT can do mRNA prediction tasks, but does not generate sequences. Here, we make CodonBERT a generative model and fine-tune it on a sample_dataset that contains low and high quaility mRNAs.

To run this code first download the pre-trained CodonBERT model here. 
Then, clone this repository and install the computing environment using the following:

```
git clone https://github.com/shakiba-shb/CodonOptimizationLLM.git
pip install requirements.txt
```
