# PGCNNGeometry
# PGCNN

**Polynomial Group-Convolutional Neural Networks (PGCNN)**

A Python/Sage package for algebraic analysis of **Polynomial Group-Convolutional Neural Networks**, including:

* symbolic computation of network parametrization maps
* image dimension computation
* generic fiber size computation
* algebraic geometry analysis via **Macaulay2**

The package integrates:

* **SageMath** for symbolic computation and group algebra
* **Macaulay2** (with the `Cremona` package) for algebraic geometry computations.

---

# Requirements

The package requires the following software:

* **Python ≥ 3.9**
* **SageMath**
* **Macaulay2**
* Macaulay2 package **Cremona**

Because SageMath is not a standard pip dependency, this package **must be installed using Sage’s Python environment**.

---

# Installation

## 1. Install SageMath

Follow the instructions for your system:

https://doc.sagemath.org/html/en/installation/index.html

Verify installation:

```bash
sage --version
```

---
## 2. Install Macaulay2

Instructions:

https://macaulay2.com/

Verify installation:

M2

Inside Macaulay2 install the Cremona package:

installPackage "Cremona"

---

## 3. Install the PGCNN package


```bash
sage -python -m pip install pgcnn
```


---

# Verify Installation

Start Sage Python:

```bash
sage -python
```

Run the following test:

```python
from sage.all import CyclicPermutationGroup
from pgcnn import PGCNN

G = CyclicPermutationGroup(2)

net = PGCNN(G, [1])

print(net.compute_dim_image_Phi_sage())
```

If this runs without errors, the installation is successful.

---

# Example Usage

```python
from sage.all import CyclicPermutationGroup
from pgcnn import PGCNN

# Define a group
G = CyclicPermutationGroup(3)

# Define activation exponents
r = [1, 2]

# Create network
net = PGCNN(G, r)

# Compute image dimension
dim = net.compute_dim_image_Phi_sage()

print("Image dimension:", dim)
```
See example.sage for more examples.

---

# Citation

If you use this package in research, please cite our paper. 

@misc{hendi2026geometrypolynomialgroupconvolutional,
      title={The Geometry of Polynomial Group Convolutional Neural Networks}, 
      author={Yacoub Hendi and Daniel Persson and Magdalena Larfors},
      year={2026},
      eprint={2603.29566},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.29566}, 
}

---

# License

MIT License.


