# pyTensor

## Overview
This is sample usage TensorFlow in python

### Memo
Model
```
W = Weight
b = bias
x = input
y = output

evidense = Σ W * x + b
y = softmax(evidense)

-> y = softmax(W * x + b)
```

Cross-entropy
```
y = predicted probability distribution
y' = true distribution (label)

H_y'(y) = - Σ y'_i * log(y_i)
```
