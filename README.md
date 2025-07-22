# TetrahedralBases

# Tetrahedral Bases and Elegant Joint Measurements

This repository contains the core code used in two research projects on **multi-qubit tetrahedral measurement bases** and their classification under **Clifford hierarchy and local equivalence**. It includes all scripts used in the numerical and symbolic analyses presented in:

- **[Geometry and Localizability of Pauli-Orbit Measurements](https://arxiv.org/abs/XXXX.XXXXX)** 
- **[The Multiqubit Elegant Joint Measurement](https://arxiv.org/abs/YYYY.YYYYY)**

---

## Contents

### 📘 General Tetrahedral Bases

These scripts classifies all tetrahedral bases according to local geometry at some specified level of the Clifford hierarchy:

- **`2qubit_classification.py`**  
  For **2 qubits**: given a Clifford level \( k \), this script outputs all **unique Bloch vector geometries** for tetrahedral bases localizable at level \( k \). Also tests Clifford hierarchy levels of the corresponding basis matrices and diagonal gates.

- **`3qubit_classification.py`**  
  Analogous to above, but for **3-qubit** tetrahedral bases.

---

### 📗 Short Paper: Elegant Joint Measurements (EJM)

These scripts focus on characterizing and classifying the **Elegant Joint Measurement** and its multi-qubit generalizations.

- **`EJM_finder.py`**  
  Given \( n \) qubits and Clifford level \( k \), finds **all locally isotropic tetrahedral bases**.

- **`LCE_3qubit.py`**  
  Performs a **complete classification** of 3-qubit EJM bases **up to local Clifford equivalence**, using symbolic methods and symmetry checks.

- **`PPIsimple.nb`** *(Mathematica)*  
  Verifies the **analytic form** of the canonical 3-qubit EJM state derived in Appendix B of The Multiqubit Elegant Joint Measurement, including its polynomial phase and tensor product structure.

- **`Cyril_family.nb`** *(Mathematica)*  
  Analyzes the localizability of a family of measurements (https://arxiv.org/abs/2006.16694) that interpolates between the Bell basis and the EJM

## Data

This folder contains plain-text output files from the Python codes. These include:
- Classification results for 2-qubit tetrahedral bases.
- Outputs from `EJM_finder.py`, which searches for new EJM-type fiducials. This is the input for LCE_3qubit.py

---

## Dependencies

All Python scripts require:

- Python ≥ 3.8  
- Libraries:
  - `numpy`
  - `itertools` (standard library)
  - `math` (standard library)
  - `sympy` (for symbolic processing, used in some scripts)
