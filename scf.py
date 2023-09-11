import torch
from math_utils import *

def solve_rhf(n_ele, e_nuc, h_core, overlap, eri, check_converge, init_P = None, max_iter = 500):
    n_occ = n_ele // 2

    X = eigen_sqrt_nosym(overlap)
    
    if init_P is None:
        energy_mo, coeff_mo = transformed_eigh(A=h_core, X=X)
        coeff_mo = coeff_mo[:, :, :n_occ]
        P_old = coeff_mo @ coeff_mo.transpose(-1,-2) * 2
    else:
        P_old = init_P

    P = P_old
    fock = None
    energy_old = None

    for i in range(max_iter):
        coul = torch.einsum('Npqrs,Nrs->Npq', eri, P)
        exch = - torch.einsum('Nprqs,Nrs->Npq', eri, P) / 2
        fock = h_core + coul + exch

        energy_mo, coeff_mo = transformed_eigh(A=fock, X=X)
        coeff_mo = coeff_mo[:, :, :n_occ]
        P = coeff_mo @ coeff_mo.transpose(-1,-2) * 2
        energy = torch.einsum('Npq, Npq -> N', h_core + fock, P) / 2

        if energy_old is not None:
            if check_converge(i, (P-P_old).norm(dim=(-1,-2)), (energy-energy_old).abs()):
                energy_rhf = energy + e_nuc
                return energy_rhf
        P_old = P
        energy_old = energy

    energy_rhf = energy + e_nuc
    print(f'not converged in {max_iter} iterations')
    return energy_rhf