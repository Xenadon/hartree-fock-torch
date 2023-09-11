from collections import namedtuple
from copy import deepcopy
from functools import partial
import torch
from utils import size_repr
import pyscf

class HartreeFockData(
    namedtuple('HartreeFockData', ['N', 
                                   'e_nuc', 'h_core', 'overlap', 'ele_repul',
                                   'n_ao', 'n_ele', 'n_occ']),
    ):

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = [size_repr(k, v) for k, v in self._asdict().items()]
        return '{}({})'.format(cls, ', '.join(info))


def noised_batch(mol_data, noise_intensity, batch_size):
    '''
    Generate a batch of molecular geometries from a single geometry by adding Gaussian noise to the atom-positions. 
    '''
    ba = []
    for i in range(batch_size):
        noised = deepcopy(mol_data)
        noised.pos[:] += torch.randn_like(noised.pos) * noise_intensity
        ba.append(noised)
    return ba

def HF_transform(basis, need_mol = False):
    '''
    Returns a transform function that transforms a molecular geometry to the qualities needed in HF calculation under given basis.
    '''
    def HF_transform_fn(mol_data):
        (pos, z) = (mol_data.pos, mol_data.z)
        N = z.shape[0]
        mol = pyscf.gto.Mole()
        for i in range(N):
            mol.atom.append((z[i], pos[i]))
        mol.basis = basis
        mol.build()
        e_nuc = torch.tensor(mol.energy_nuc())
        hcore = torch.tensor(mol.intor('int1e_nuc') + mol.intor('int1e_kin'))
        ovlp = torch.tensor(mol.intor('int1e_ovlp'))
        eri = torch.tensor(mol.intor('int2e'))
        n_ao = hcore.shape[0]
        n_ele = z.sum().item()
        n_occ = n_ele//2
        if not need_mol:
            return HartreeFockData(N, e_nuc, hcore, ovlp, eri, n_ao, n_ele, n_occ)
        else:
            return HartreeFockData(N, e_nuc, hcore, ovlp, eri, n_ao, n_ele, n_occ), mol
    return HF_transform_fn

def HF_transform_batch(mol_datas, basis, need_mols = False):
    '''
    Transform a batch of molecular geometries to the HFData
    '''
    t = HF_transform(basis, need_mol=need_mols)
    if not need_mols:
        return [t(mol_data) for mol_data in mol_datas]
    else:
        data = []
        mols = []
        for mol_data in mol_datas:
            m_d, mol = t(mol_data)
            data.append(m_d)
            mols.append(mol)
        return data, mols
    
def get_demo_geometry(pos=None, z=None, from_dataset=False):
    if pos is not None and z is not None:
        Data = namedtuple('Data', ['pos', 'z'])
        return Data(pos, z)
    if from_dataset:
        from torch_geometric.datasets import QM9
        dataset = QM9(root='./dataset/QM9/')
        mol_data = dataset[0]
        return mol_data
    else:
        Data = namedtuple('Data', ['pos', 'z'])
        pos = torch.tensor([[-1.2700e-02,  1.0858e+00,  8.0000e-03],
                            [ 2.2000e-03, -6.0000e-03,  2.0000e-03],
                            [ 1.0117e+00,  1.4638e+00,  3.0000e-04],
                            [-5.4080e-01,  1.4475e+00, -8.7660e-01],
                            [-5.2380e-01,  1.4379e+00,  9.0640e-01]])
        z = torch.tensor([6, 1, 1, 1, 1])
        return Data(pos, z)
    
    