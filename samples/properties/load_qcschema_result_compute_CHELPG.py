## Computation of CHELPG charges.
## Loads QCSchema format json and hdf5 result from Accelerated DFT
## and computes dipole moment.
## Accelerated DFT task can be energy, gradient, hessian,
##  or geometry optimization
##  with the wavefunction output e.g. "write_wavefunction": "last"
import pyscf
from pyscf import gto, dft
import json
import numpy as np
from tools.chelpg import chelpg_charges
from tools.libqcschema import *
from tools.wavefunction_hdf5_to_qcschema import *

def main():

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

    print("")

    #### Compute CHELPG charges ####
    # we have multiple choices for Van der Waals radii schemes and settings. 
    # Some examples uses:

    # 1. simplest use, does default VDW scheme
    q = chelpg_charges(ks)
    print("method 1, charges: ")
    print(q)

    # 2. Use 'VDW' radii scheme and set grid settings
    options = {'VDW_SCHEME': 'VDW', 'deltaR':0.3, 'Rhead':2.8}
    q = chelpg_charges(ks,options)
    print("method 2, charges: ")
    print(q)

    # 3. Use 'UFF' radii scheme 
    options = {'VDW_SCHEME': 'UFF'}
    q = chelpg_charges(ks,options)
    print("method 3, charges: ")
    print(q)

    # 4. user defined radii scheme (a dictionary) 
    RVDW_bondi = {1: 1.1, 2: 1.40,
                  3: 1.82, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 10: 1.54,
                  11: 2.27, 12: 1.73, 14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75, 18: 1.88,
                  19: 2.75, 35: 1.85}
    options = {'VDW_RADII': RVDW_bondi}
    q = chelpg_charges(ks,options)
    print("method 4, charges: ")
    print(q)
    #############
    
    #### compare to Mulliken charges ####
    # First compute density matrix
    mo_occ = ks.mo_occ
    mo_coeff = ks.mo_coeff
    dm = ks.make_rdm1(mo_coeff, mo_occ)
    # add verbose=0 to turn off printing of pop stuff
    mpop = pyscf.scf.hf.mulliken_pop(mol, dm, s=None,verbose=0)[-1]
    print("Mulliken Population charges:",mpop)
    print("")
    #############



if __name__ == "__main__":
    main()
