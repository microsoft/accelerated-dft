## Computation of Dipole Moment.
## Loads QCSchema format json and hdf5 result from Accelerated DFT
## and computes dipole moment.
## Accelerated DFT task can be energy, gradient, hessian, 
##  or geometry optimization
##  with the wavefunction output e.g. "write_wavefunction": "last"
import pyscf
from pyscf import gto, dft
import json
import numpy as np
from tools.libqcschema import *
from tools.wavefunction_hdf5_to_qcschema import *

def main():
    chkfile = ""
    qcschema_json = "examples/aspirin_neutral_output.json" 
    qcwavefunction_h5 = "examples/aspirin_neutral_output.h5"

    # Load Accelerated DFT output json
    qcschema_dict = load_qcschema_json(qcschema_json)
    
    # Load wavefunction from hdf5
    qcwavefunction = {}
    qcwavefunction['wavefunction'] = read_hdf5_wavefunction(qcwavefunction_h5)

    # add wfn info to the total qcschema output
    qcschema_dict.update(qcwavefunction)
    
    # Create DFT object
    mol, ks = recreate_scf_obj(qcschema_dict)

    #### Compute Molecular Dipole Moment ####
    # load density matrix
    dm = qcwavefunction['wavefunction']['scf_density_a']
    # OR compute density matrix from MOs
    #mo_occ = ks.mo_occ
    #mo_coeff = ks.mo_coeff
    #dm = ks.make_rdm1(mo_coeff, mo_occ)
    # compute dipole
    DipMom = ks.dip_moment(ks.mol, dm, unit='Debye', verbose=3)
    # note: instead one can also use using single 'analyze' will compute dipole moment and mulliken population 
    #an = pyscf.scf.hf.analyze(ks)
    #############

if __name__ == "__main__":
    main()
