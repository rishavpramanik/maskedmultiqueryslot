# Masked Multi-Query Slot Attention
"**Masked Multi-Query Slot Attention for Unsupervised Object Discovery**" - accepted for presentation in [2024 International Joint Conference on Neural Networks](https://2024.ieeewcci.org/).

Access the paper: [Arxiv](https://arxiv.org/abs/2404.19654)
```
@INPROCEEDINGS{pramanik2024masked,
AUTHOR="Rishav Pramanik and Jos{\'e}-Fabian {Villa-V{\'a}squez} and Marco Pedersoli",
TITLE="Masked {Multi-Query} Slot Attention for Unsupervised Object Discovery",
BOOKTITLE="2024 International Joint Conference on Neural Networks (IJCNN) (IJCNN 2024)",
ADDRESS="Yokohama, Japan",
DAYS=28,
MONTH=jun,
YEAR=2024,
}
```


## Datasets Used:
PASCAL VOC 2012 [Click here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)
## Requirements

- Python >= 3.8
- PyTorch >= 1.7.1
- Pytorch Lightning >= 1.1.4
- CUDA enabled computing device
  
## Instructions to run the code:
1. Download the repository and install the required packages:
```
pip3 install -r requirements.txt
```
2. Unzip the data in a folder of your choice
```
tar -xf yourdirectory/VOCtrainval_11-May-2012.tar -C $SLURM_TMPDIR/yourdirectory
```
3. The train2 file is sufficent to run the code
```
torchrun --nproc_per_node=4 --nnodes=1 scripts/train2.py
```

Edit the parameters before you start in params.py

## Acknowledgements
We greatly thank the authors of https://github.com/amazon-science/object-centric-learning-framework/tree/main and https://github.com/imbue-ai/slot_attention/tree/master for their code which had helped us in our work
