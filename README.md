# BDA_projcet

# Environment
```
docker pull tjsanf0606/paper:xdino
```

# Run
```
python tran_lenet_mnist.py
```

# Evaluation
```
from sklearn import metrics

metrics.roc_auc_score(target_array, predic_array)
metrics.average_precision_score(target_array[:,1], predic_array[:,1])
```
