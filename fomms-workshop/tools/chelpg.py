## CHarges from ELectrostatic Potentials using a Grid-based method ## 
## CHELPG charges:
## Breneman and Wiberg (J. Comp. Chem. 1990, 11, 361)
## Defaults for grid spacing and size are taken from this paper. 
## For options see function definition below
#  
## Description: compute ESP on grid points around the molecule. 
#  grid points are arranged in a cuboid,
#  with any points falling inside the VDW radii being removed.
#  Hence VWD defs are important.
#
import pyscf
import numpy as np
import scipy
import ctypes
from pyscf import lib, gto
from pyscf.scf import _vhf
from pyscf.lib.parameters import BOHR
from pyscf import data

def chelpg_charges(mf, options=None):
    """Calculate chelpg charges

    Args:
        mf: mean field object in pyscf
        options (optional):

            deltaR (float, optional): the interval in the cube. Defaults to 0.3.
            Rhead (float, optional): the head length of the cube. Defaults to 2.8.
            ifqchem (bool, optional): whether to use the modification in qchem. Defaults to True.
            VDW_SCHEME(str,optional): what VDW radii scheme to use. 
                                      For details see https://github.com/pyscf/pyscf/blob/master/pyscf/data/radii.py
                                      Choices: VDW - vdw from ASE a.k.a. Bondi 
                                               VDW_mod - (default) VDW but with H modified to 1.1A. (recommended)   
                                               UFF - Universal Force Field 
                                               MM3 - Allinger's MM3
                                               BRAGG - # JCP 41, 3199 (1964)
                                               COVALENT - covalent radius
            VDW_RADII(dict,optional): dictionary of user defined vdw radii to use, in Angstrom
            printq (bool, optional): whether to print charges and scheme at end of this function. Default=True
    Returns:
        numpy.array: charges
    """
#    # Check input options:
    if options is None:
        options = {}
    #print(options)

    # check print
    if 'printq' not in options:
        options['printq'] = False

    # VDW surface options
    if 'deltaR' not in options:
        deltaR = 0.3
    else:
        deltaR = options['deltaR']

    if 'Rhead' not in options:
        Rhead = 2.8
    else:
        Rhead = options['Rhead']

    if 'ifqchem' not in options:
        ifqchem = True
    else:
        Rhead = options['ifqchem']

    radii = {}
    if 'VDW_RADII' in options:
        options['VDW_SCHEME'] = 'USER'
    # Use PySCF built-in VDW schemes from pyscf.data.radii
    if 'VDW_SCHEME' not in options:
        options['VDW_SCHEME'] = 'VDW_mod'

    if( options['VDW_SCHEME'] == 'VDW_mod' ):
        # modified Bondi
        vdw_array = data.radii.VDW
        vdw_array[1] = 1.1/BOHR
    if( options['VDW_SCHEME'] == 'VDW' ):
        vdw_array = data.radii.VDW
    if( options['VDW_SCHEME'] == 'UFF' ):
        vdw_array = data.radii.UFF
    if( options['VDW_SCHEME'] == 'MM3' ):
        vdw_array = data.radii.MM3
    if( options['VDW_SCHEME'] == 'BRAGG' ):
        vdw_array = data.radii.BRAGG
    if( options['VDW_SCHEME'] == 'COVALENT' ):
        vdw_array = data.radii.COVALENT

    if( options['VDW_SCHEME'] != 'USER' ):
        # in all pyscf data element 0 is not useful. H is at 1
        vdw_array = vdw_array[1:]
        # convert to Dict
        radii = dict(enumerate(vdw_array.flatten(), 1))

    # or a user defined scheme?
    if options['VDW_SCHEME'] == 'USER':
        radii = options['VDW_RADII']
        # convert to Bohr
        for key in radii:
            radii[key] /= BOHR

    ##

    # define extra params
    Roff = Rhead/BOHR
    Deltar = 0.1

    # smoothing function
    def tau_f(R, Rcut, Roff):
        return (R - Rcut)**2 * (3*Roff - Rcut - 2*R) / (Roff - Rcut)**3

    #### Check the atoms have a defined VDW in the scheme
    elements = np.array(mf.mol._atm[:, 0])
    for i in elements:
        if i not in radii.keys() or radii[i] == 0.0:
            raise KeyError('%s is not a supported element; ' %i
                         + 'use the "VDW_RADII" option to add '
                         + 'its van der Waals radius.')
    #### atom check complete

    atomcoords = mf.mol.atom_coords(unit='B')
    dm = np.array(mf.make_rdm1())

    Rshort = np.array([radii[iatom] for iatom in mf.mol._atm[:, 0]])
    idxxmin = np.argmin(atomcoords[:, 0] - Rshort)
    idxxmax = np.argmax(atomcoords[:, 0] + Rshort)
    idxymin = np.argmin(atomcoords[:, 1] - Rshort)
    idxymax = np.argmax(atomcoords[:, 1] + Rshort)
    idxzmin = np.argmin(atomcoords[:, 2] - Rshort)
    idxzmax = np.argmax(atomcoords[:, 2] + Rshort)
    atomtypes = np.array(mf.mol._atm[:, 0])
    # Generate the grids in the cube
    xmin = atomcoords[:, 0].min() - Rhead/BOHR - radii[atomtypes[idxxmin]]
    xmax = atomcoords[:, 0].max() + Rhead/BOHR + radii[atomtypes[idxxmax]]
    ymin = atomcoords[:, 1].min() - Rhead/BOHR - radii[atomtypes[idxymin]]
    ymax = atomcoords[:, 1].max() + Rhead/BOHR + radii[atomtypes[idxymax]]
    zmin = atomcoords[:, 2].min() - Rhead/BOHR - radii[atomtypes[idxzmin]]
    zmax = atomcoords[:, 2].max() + Rhead/BOHR + radii[atomtypes[idxzmax]]
    x = np.arange(xmin, xmax, deltaR/BOHR)
    y = np.arange(ymin, ymax, deltaR/BOHR)
    z = np.arange(zmin, zmax, deltaR/BOHR)
    gridcoords = np.meshgrid(x, y, z)
    gridcoords = np.vstack(list(map(np.ravel, gridcoords))).T

    # [natom, ngrids] distance between an atom and a grid
    r_pX = scipy.spatial.distance.cdist(atomcoords, gridcoords)
    # delete the grids in the vdw surface and out the Rhead surface.
    # the minimum distance to any atom
    Rkmin = (r_pX - np.expand_dims(Rshort, axis=1)).min(axis=0)
    Ron = Rshort + Deltar
    Rlong = Roff - Deltar
    AJk = np.ones(r_pX.shape)  # the short-range weight
    idx = r_pX < np.expand_dims(Rshort, axis=1)
    AJk[idx] = 0
    if ifqchem:
        idx2 = (r_pX < np.expand_dims(Ron, axis=1)) * \
            (r_pX >= np.expand_dims(Rshort, axis=1))
        AJk[idx2] = tau_f(r_pX, np.expand_dims(Rshort, axis=1),
                          np.expand_dims(Ron, axis=1))[idx2]
        wLR = 1 - tau_f(Rkmin, Rlong, Roff)  # the long-range weight
        idx1 = Rkmin < Rlong
        idx2 = Rkmin > Roff
        wLR[idx1] = 1
        wLR[idx2] = 0
    else:
        wLR = np.ones(r_pX.shape[-1])  # the long-range weight
        idx = Rkmin > Roff
        wLR[idx] = 0
    w = wLR*np.prod(AJk, axis=0)  # weight for a specific point
    idx = w <= 1.0E-14
    w = np.delete(w, idx)
    r_pX = np.delete(r_pX, idx, axis=1)
    gridcoords = np.delete(gridcoords, idx, axis=0)

    ngrids = gridcoords.shape[0]
    r_pX = np.array(r_pX)
    r_pX_potential = 1/r_pX
    # nuclear part of electrostatic potential (ESP)
    potential_real = np.dot(np.array(
        mf.mol.atom_charges()), r_pX_potential)

    # add in the electronic part of ESP..need to do in batches if large
    # amount of memory required: ngrdis*(NBasis*NBasis)*8, divide by (1024**3) for GB
    ## non-batched:
    ##  Vele = np.einsum('pij,ij->p', mf.mol.intor('int1e_grids', grids=gridcoords), mf.make_rdm1())
    ##  potential_real -= Vele
    ## batched:
    try:
        ### define batch size based on available memory - if psutil installed
        import psutil
        NBasis = int(mf.mol.nao_nr())
        mem_avail = psutil.virtual_memory()[1]
        grid_avail = int(mem_avail/(8*(NBasis)*(NBasis)))
        # batch size, use 90% of available memory
        nbatch = min( int( grid_avail*0.90) , ngrids)
    except:
        # hard coded batch size - may fail, just adjust nbatch
        nbatch = 128*128

    # need density matrix
    dm = mf.make_rdm1()

    try:
        for ibatch in range(0, ngrids, nbatch):
            max_grid = min(ibatch+nbatch, ngrids)
            num_grids = max_grid - ibatch
            grid_bit = gridcoords[ibatch:max_grid] 
            potential_real[ibatch:max_grid] -= np.einsum('pij,ij->p', mf.mol.intor('int1e_grids', grids=grid_bit), dm)
    except:
        print("Out of memory in func chelpg_charges. Make batch size (nbatch) smaller")
        return()    

    w = np.array(w)
    r_pX_potential_omega = r_pX_potential*w
    GXA = r_pX_potential_omega@r_pX_potential.T
    eX = r_pX_potential_omega@potential_real
    GXA_inv = np.linalg.inv(GXA)
    g = GXA_inv@eX
    alpha = (g.sum() - mf.mol.charge)/(GXA_inv.sum())
    q = g - alpha*GXA_inv@np.ones((mf.mol.natm))


    # print output
    if(options['printq']):
        print('VDW SCHEME', options['VDW_SCHEME'])
        print("charges:", q)

    return q

