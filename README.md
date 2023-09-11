# Hartree-Fock-torch
A simple implemetation of solving a batch of Hartree-Fock equations (of the same molecule) using the SCF iteration method.

The algorithm is from Szabo and Ostlund, 3.4.6, p. 146. and the implementation also references [the one of yangdatou](github.com/yangdatou/hf-tutorial).

Files:

- `scf.py` 

    The main implementation of SCF methods.

- `math_utils.py`

    Including the functions related to solving generalized eigenvalue problems.

- `data_utils.py`

    Including the functions of constructing and transforming data.

- `HartreeFock.ipynb`

    A demo.

Dependencies:

-   `torch`
-   `pyscf`

