# Preparation
```bash
git clone https://github.com/open-mmlab/mmdetection.git
```
Follow the [installation guidance](https://mmdetection.readthedocs.io/en/latest/get_started.html) from mmdetection.

# Train-Exp1
Run `bash exp1.sh`

or
```bash
for c in 1 0.1 0.01 0.001; do
    for o in 0.01 1e-4 0; do
        for bs in 8 32 128; do
            CUDA_VISIBLE_DEVICES=0 python src/main.py --cls-lr ${c} --other-lr ${o} --bs ${bs}
            CUDA_VISIBLE_DEVICES=0 python src/main.py --cls-lr ${c} --other-lr ${o} --bs ${bs} --noinit
        done
    done
done
```

# Train-Exp2
```bash
cd mmdetection
bash tools/dist_train.sh config/CUSTOM/mask-rcnn.py 4
bash tools/dist_train.sh config/CUSTOM/sparse.py 4
```

# Visualization-Exp2
```bash
CUDA_VISIBLE_DEVICES=0 python '/root/NNDL/mmdetection/tools/visualize_infer_sparse.py'
CUDA_VISIBLE_DEVICES=0 python '/root/NNDL/mmdetection/tools/visualize_infer_mask.py'
CUDA_VISIBLE_DEVICES=0 python '/root/NNDL/mmdetection/tools/vis_proposal.py'
```
