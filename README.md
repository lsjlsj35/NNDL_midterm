# Preparation
```bash
git clone https://github.com/open-mmlab/mmdetection.git
```
Then follow the [installation guidance](https://mmdetection.readthedocs.io/en/latest/get_started.html) from mmdetection to create the environment.

**IMPORTANT**
- To visualize proposal bounding boxes, we hack the original `rpn_head.py` to manually create a "hook", so that we can get the predicted bounding boxes.
```bash
patch mmdetection/mmdet/models/dense_heads/rpn_head.py < rpn_hook.patch
``` 

# Exp1
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

Run `python script/exp_result1.py` to get experiment results.

# Exp2
```bash
cd mmdetection
```
The second experiment is conducted in *mmdetection* directory.
## Train
```bash
cd mmdetection
bash tools/dist_train.sh config/CUSTOM/mask-rcnn.py 4
bash tools/dist_train.sh config/CUSTOM/sparse.py 4
```

## Visualization
```bash
CUDA_VISIBLE_DEVICES=0 python tools/visualize_infer_sparse.py
CUDA_VISIBLE_DEVICES=0 python tools/visualize_infer_mask.py
CUDA_VISIBLE_DEVICES=0 python tools/vis_proposal.py
```
