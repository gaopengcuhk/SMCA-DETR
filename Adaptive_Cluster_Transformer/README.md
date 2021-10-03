# End-to-End Object Detection with Adaptive Clustering Transformer

The implementation of Adaptive Clustering Transformer (ACT). ACT can reduce the computation and memory costs of DETR without any re-training. 

## Dependencies

The ACT model has the following dependencies:
- PyTorch
- CUDA toolchain

Our experimental environment is PyTorch 1.6.0 + CUDA 9.2

## Installation

First, install PyTorch and torchvision with CUDA.

Then, install pycocotools for evaluation:
```
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Finally, install the ACT library:
```
python setup.py install
```

## Usage

Replace `torch.nn.MultiheadAttention` with `ACT.AdaMultiheadAttention`:

```python
class AdaMultiheadAttention(embed_dim, num_heads, group_Q=True, group_K=False, 
                            q_hashes=32, k_hashes=32, **kwargs):
```

**Parameters:**

- **embed_dim** – total dimension of the model.
- **num_heads** – parallel attention heads.
- **group_Q** – whether apply adaptive clustering to queries.
- **group_K** – whether apply adaptive clustering to keys.
- **q_hashes** – number of hash rounds for queries. Only available when group_Q is true. 
- **k_hashes** – number of hash rounds for keys. Only available when group_K is true. 
- **kwargs** – other arguments in ``torch.nn.MultiheadAttention``. 

## Experiments on DETR

### Data preparation

Download the COCO 2017 val images with annotations from http://cocodataset.org. The directory structure is as follows:
```
DataPath/
    annotations/
    val2017/
```

### Evaluation

We have modified the DETR model in the **experiments** folder. We replace the attention module in the encoder with our adaptive clustering attention. ACT can directly use the parameters training by DETR and reduce the FLOPs.

To evaluate the ACT with L = 32 and the dilated ResNet-50 backbone, run:
```
cd experiments
python main.py --no_aux_loss \
    --eval \
    --batch_size 1 \
    --dilation \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth \
    --coco_path DataPath \
    --group_Q \
    --q_hashes 32 \
```

Changing the parameters of **q_hashes** can test models with different FLOPs. We recommend setting **n_hashes** to 32, 24, 20, or 16.

To evaluate the DETR with the dilated ResNet-50 backbone for comparison, run:

```
cd experiments
python main.py --no_aux_loss \
    --eval \
    --batch_size 1 \
    --dilation \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth \
    --coco_path DataPath \
```


## Acknowledgement

This project is motivated by DETR, Cluster Attention, Reformer and SMYRF. 
