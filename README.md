# **Efficient Learning for Linear Properties of Bounded-Gate Quantum Circuits**
This repository provides the source code for our work “Efficient Learning for Linear Properties of Bounded-Gate Quantum Circuits” [![arXiv](https://img.shields.io/badge/arXiv-2408.12199-blue)](https://arxiv.org/pdf/2408.12199). Our approach leverages **classical shadows** and **compact representations of input configurations** to balance **prediction accuracy and computational efficiency**. We validate the effectiveness of this framework across diverse quantum applications, including **quantum information processing, Hamiltonian simulation, and variational quantum algorithms (VQAs)** with up to **60 qubits**.


![Image Description](assets/scheme.png)


## **Overview**
Predicting the behavior of large-scale quantum circuits is essential for advancing quantum computing and understanding quantum systems. However, the extent to which **classical learners** can infer linear properties of quantum circuits from measurement data remains an open question. This repository implements:
- **A kernel-based learning framework** designed for efficient inference of bounded-gate quantum circuits.
- **Theoretical guarantees on sample complexity**, demonstrating linear scaling with tunable gate count while computational costs may grow exponentially.
- **Optimized pretraining strategies** for **variational quantum eigensolvers (VQEs)**, enabling resource-efficient quantum system certification.


## **Usage**
### **1. Training the Learning Model** (Fig 2)

The source code is avaiable at the folder 
```python
from model import QuantumLearningModel

# Define parameters
num_qubits = 10
num_samples = 500
model = QuantumLearningModel(num_qubits=num_qubits)

# Train the model
model.train(num_samples)
```

### **2. Evaluating the Model**
```python
accuracy = model.evaluate()
print(f"Model accuracy: {accuracy:.4f}")
```

### **3. Pretraining VQEs**
```python
from vqe_pretrain import VQEPipeline

vqe = VQEPipeline()
vqe.pretrain()
```

## **Benchmarks**
We compare our method against **LOWESA** and **other classical simulators**, demonstrating advantages in **runtime efficiency** and **robustness to quantum noise**. Detailed benchmarking results are available in the **Benchmarking** section of our paper.

## **Citation**
If you find this work useful, please consider citing:
```bibtex
@article{du2024efficient,
  title={Efficient learning for linear properties of bounded-gate quantum circuits},
  author={Du, Yuxuan and Hsieh, Min-Hsiu and Tao, Dacheng},
  journal={arXiv preprint arXiv:2408.12199},
  year={2024}
}
```

## **License**
This project is released under the **MIT License**.
