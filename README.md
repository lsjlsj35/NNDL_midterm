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
bash tools/dist_train.sh 
```
