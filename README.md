# self-pruning-neural-network


A deep learning project that implements **dynamic weight pruning** using learnable gates, enabling the model to automatically identify and remove less important connections during training.

---

## Project Overview

Traditional neural networks are often over-parameterized, leading to unnecessary computation and memory usage.
This project introduces a **self-pruning mechanism** where each weight is controlled by a learnable gate that determines its importance.

During training:

* Important connections are retained
* Weak connections are suppressed and pruned

This results in a **more efficient and compact model** without manual pruning.

---

##  Key Features

*  Custom `PrunableLinear` layer with learnable gating mechanism
*  L1 regularization on gates to induce sparsity
*  Automatic pruning during training (no post-processing required)
*  Trade-off analysis between **accuracy vs sparsity**
*  Visualization of gate distributions and pruning behavior

---

## Architecture

```
Input (CIFAR-10 Images)
   ↓
Flatten
   ↓
FC (1024) → BatchNorm → ReLU
   ↓
FC (512)  → BatchNorm → ReLU
   ↓
FC (256)  → BatchNorm → ReLU
   ↓
FC (10)   → Output
```

Each fully connected layer is replaced with a **PrunableLinear layer**.

---

##  How It Works

Each weight has an associated learnable parameter:

```
gate = sigmoid(gate_score)
pruned_weight = weight × gate
```

* Gates close to **0 → pruned connections**
* Gates close to **1 → active connections**

Sparsity is encouraged using:

```
Loss = Classification Loss + λ × L1(gates)
```

---

##  Results

* Achieved competitive accuracy on CIFAR-10
* Demonstrated **emerging sparsity during training**
* Observed clear trade-off between:

  * Model performance
  * Model sparsity

---

## Tech Stack

* Python
* PyTorch
* NumPy
* Matplotlib

---

## How to Run

```bash
pip install -r requirements.txt
python train.py
```

---

##  Key Learning Outcomes

* Understanding of **model compression techniques**
* Practical implementation of **learnable sparsity**
* Experience with **custom neural network layers in PyTorch**
* Trade-off analysis in deep learning systems

---

## Future Improvements

* Extend to CNN architectures for higher accuracy
* Implement structured pruning (channel/filter level)
* Compare with standard pruning techniques

---

