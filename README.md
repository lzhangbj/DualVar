## Official implementation for BMVC 2021 poster paper "Inter-intra Variant Dual Representations for Self-supervised Video Recognition"
Contrastive learning applied to self-supervised representation learning has seen a resurgence in deep models. In this paper, we find that existing contrastive learning based solutions for self-supervised video recognition focus on inter-variance encoding but ignore the intra-variance existing in clips within the same video. We thus propose to learn dual representations for each clip which (i) encode intra-variance through a shuffle-rank pretext task; (ii) encode inter-variance through a temporal coherent contrastive loss. Experiment results show that our method plays an essential role in balancing inter and intra variances and brings consistent performance gains on multiple backbones and contrastive learning frameworks. Integrated with SimCLR and pretrained on Kinetics-400, our method achieves **82.0%** and **51.2%** downstream classification accuracy on UCF101 and HMDB51 test sets respectively and **46.1%** video retrieval accuracy on UCF101, outperforming both pretext-task based and contrastive learning based counterparts.

![Overview](asset/overview.png)

# Set up
The code is based on Python3.8 and PyToch 1.8.1.

### Data preparation
1. Download UCF101, HMDB51 and Kinetics400 dataset.
2. Slice videos into frames using ```process_data/src/extract_frame.py```
3. Write sliced frame into csv file using ```process_data/src/write_csv.py```

### Running experiment
Before running code
1. Replace dataset directory in  ```dataset/local_dataset.py```

2. Prepare your own running settings ```paper_scripts/*/*/sh``` 

We provide pretrain, finetune, test and test retrieval command in ``paper_scripts``. We also pack up all commands into one file in ```run``` mode.
Select you running mode ```{mode}``` and experiment name ```{exp}``` , and run

```angular2html
bash paper_scripts/{mode}/{exp}.sh
```

### Peformance comparison

![Finetuning Comparison](asset/finetune_acc.png)

![Retrieval Comparison](asset/retrieval_acc.png)

### Feature vistribution visualization

![Feat_Distribution](asset/feat_dist.png)


### Acknowledgement
We refer the code frameworks of [CoCLR](https://github.com/TengdaHan/CoCLR). Thank Tengda for his excellent work!

If you find our work useful, please cite it:

```
@article{lin2021dualvar
 author = {Lin, Zhang and Qi, She and Zhengyang, Shen and Changhu, Wang},
 booktitle = {BMVC},
 title = {Inter-intra Variant Dual Representations forSelf-supervised Video Recognition},
 year = {2021}
}
```










 
