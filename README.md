# SMCA-DETR


# Usage
There are no extra compiled components in SMCA DETR and package dependencies are minimal,
so the code is very simple to use. We provide instructions how to install dependencies via conda.
First, clone the repository locally:
```
git clone https://github.com/facebookresearch/detr.git
```
Then, install PyTorch 1.5+ and torchvision 0.6+:
```
conda install -c pytorch pytorch torchvision
```
Install pycocotools (for evaluation on COCO) and scipy (for training):
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
That's it, should be good to train and evaluate detection models.

(optional) to work with panoptic install panopticapi:
```
pip install git+https://github.com/cocodataset/panopticapi.git
```

## Data preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

## Training
To train Single Scale SMCA on a single node with 8 gpus for 300 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/coco --batch_size 2 --lr_drop 40 --num_queries 300 --epochs 50 --dynamic_scale type3 --output_dir smca_single_scale


```
A single epoch takes 30 minutes, so 50 epoch training
takes around 25 hours on a single machine with 8 V100 cards.

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>backbone</th>
      <th>schedule</th>
      <th>box AP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SMCA(single scale)</td>
      <td>R50</td>
      <td>50</td>
      <td>41.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SMCA-Container(single scale)</td>
      <td>Container-S-Light</td>
      <td>50</td>
      <td>44.2</td>
    </tr>
     <tr>
      <th>1</th>
      <td>SMCA-Container(single scale)</td>
      <td>Container-M</td>
      <td>50</td>
      <td> N/A </td>
    </tr>
    <tr>
      <th>2</th>
      <td>SMCA(single scale)</td>
      <td>R50</td>
      <td>108</td>
      <td>42.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SMCA(single scale)</td>
      <td>R50</td>
      <td>250</td>
      <td>43.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SMCA(multi scale)</td>
      <td>R50</td>
      <td>50</td>
      <td>43.7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SMCA(New multi scale)</td>
      <td>R50</td>
      <td>50</td>
      <td>44.4</td>
    </tr>
  </tbody>
</table>

## SMCA has been accepted by ICCV 2021. 

## Original SMCA code submission during ICCV review period. 
https://github.com/abc403/SMCA-replication


## Release Steps1
1. Single-scale SMCA 
2. Single-scale SMCA with Container-Small
3. Single-scale SMCA with Container-Medium
4. New Multi-scale SMCA

## Citation
If you find this repository useful, please consider citing our work:
```
@article{gao2021fast,
  title={Fast convergence of detr with spatially modulated co-attention},
  author={Gao, Peng and Zheng, Minghang and Wang, Xiaogang and Dai, Jifeng and Li, Hongsheng},
  journal={arXiv preprint arXiv:2101.07448},
  year={2021}
}
```
```
@article{gao2021container,
  title={Container: Context Aggregation Network},
  author={Gao, Peng and Lu, Jiasen and Li, Hongsheng and Mottaghi, Roozbeh and Kembhavi, Aniruddha},
  journal={arXiv preprint arXiv:2106.01401},
  year={2021}
}
```
