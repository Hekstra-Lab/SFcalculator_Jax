import pytest

import numpy as np
import reciprocalspaceship as rs
import jax.numpy as jnp

from SFC_Jax.Fmodel import SFcalculator
from SFC_Jax.utils import vdw_rad_tensor, unitcell_grid_center
from SFC_Jax.voxel import voxelvalue_jax_p1, voxelvalue_jax_p1_savememory
from SFC_Jax.voxel import voxel_1dto3d_np, voxel_1dto3d_jnp


@pytest.mark.parametrize("binary", [True, False])
def test_voxelvalue_torch_p1_sm(data_pdb, data_mtz_exp, binary):
    sfcalculator = SFcalculator(
        data_pdb, mtzfile_dir=data_mtz_exp, set_experiment=True)
    vdw_rad = vdw_rad_tensor(sfcalculator.atom_name)
    uc_grid_orth_tensor = unitcell_grid_center(sfcalculator.unit_cell,
                                               spacing=4.5, return_tensor=True)
    CUTOFF = 0.0001
    N_grid = len(uc_grid_orth_tensor[:, 0])
    spacing = jnp.max(uc_grid_orth_tensor[1] - uc_grid_orth_tensor[0])
    # s ~ log(1/c -1) / (d/2 - r)
    spacing = jnp.maximum(spacing, jnp.array(3.5))
    steepness = np.log(1.0/CUTOFF - 1.0)/(spacing/2.0 - 1.5)

    voxel_map_p1 = voxelvalue_jax_p1(uc_grid_orth_tensor, sfcalculator.atom_pos_orth,
                                       sfcalculator.unit_cell, sfcalculator.space_group, vdw_rad,
                                       s=steepness, binary=binary, cutoff=CUTOFF)
    voxel_map_p1_sm = voxelvalue_jax_p1_savememory(uc_grid_orth_tensor, sfcalculator.atom_pos_orth,
                                                     sfcalculator.unit_cell, sfcalculator.space_group, vdw_rad,
                                                     s=steepness, binary=binary, cutoff=CUTOFF)

    if binary:
        assert np.all(np.array(voxel_map_p1) ==
                      np.array(voxel_map_p1_sm))
    else:
        assert np.all(np.isclose(np.array(voxel_map_p1),
                      np.array(voxel_map_p1_sm)))


def test_voxel_1dto3d(data_pdb, data_mtz_exp):
    sfcalculator = SFcalculator(
        data_pdb, mtzfile_dir=data_mtz_exp, set_experiment=True)
    vdw_rad = vdw_rad_tensor(sfcalculator.atom_name)
    uc_grid_orth_tensor = unitcell_grid_center(sfcalculator.unit_cell,
                                               spacing=4.5, return_tensor=True)
    CUTOFF = 0.0001
    N_grid = len(uc_grid_orth_tensor[:, 0])
    spacing = jnp.max(uc_grid_orth_tensor[1] - uc_grid_orth_tensor[0])
    # s ~ log(1/c -1) / (d/2 - r)
    spacing = jnp.maximum(spacing, jnp.array(3.5))
    steepness = np.log(1.0/CUTOFF - 1.0)/(spacing/2.0 - 1.5)

    voxel_map_p1 = voxelvalue_jax_p1(uc_grid_orth_tensor, sfcalculator.atom_pos_orth,
                                       sfcalculator.unit_cell, sfcalculator.space_group, vdw_rad,
                                       s=steepness, binary=True, cutoff=CUTOFF)
    na, nb, nc = 6, 9, 11
    map_1 = voxel_1dto3d_np(np.array(voxel_map_p1), na, nb, nc)
    map_2 = np.array(voxel_1dto3d_jnp(voxel_map_p1, na, nb, nc))
    assert np.all(map_1 == map_2)
