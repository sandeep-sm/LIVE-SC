## PyContrast

### Eval command:
```
python3 main_contrast_eval.py --method MoCov2 --head mlp
```
### Training command:
```
python3 main_contrast_IQA.py --method MoCov2 --cosine --head mlp --multiprocessing-distributed --world-size 1 --rank 0
```
### Eval features with NR model:
```
python3 train_linear_regression_NR.py --use_parallel
```

This repo is built on top of PyContrast(author: HobbitLong) available [here](https://github.com/HobbitLong/PyContrast).
