## Generation fo Localized orbitals
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
import argparse

def main():
    parser = argparse.ArgumentParser(description="Compute localized orbitals from QCSchema and wavefunction files.")
    parser.add_argument("qcschema_json", help="Path to the QCSchema JSON file.")
    parser.add_argument("qcwavefunction_h5", help="Path to the wavefunction HDF5 file.")
    args = parser.parse_args()

    qcschema_json = args.qcschema_json
    qcwavefunction_h5 = args.qcwavefunction_h5

    # Load Accelerated DFT output json
    qcschema_dict = load_qcschema_json(qcschema_json)

    # Load wavefunction from hdf5
    qcwavefunction = {}
    qcwavefunction['wavefunction'] = read_hdf5_wavefunction(qcwavefunction_h5)

    # add wfn info to the total qcschema output
    qcschema_dict.update(qcwavefunction)

    # Create DFT object
    mol, ks = recreate_scf_obj(qcschema_dict)

    #### Localize Orbitals ####
    from pyscf import lo
    from pyscf.tools import molden
    # Localize occupied orbitals only
    select_mo = ks.mo_occ>0
    # Localize ALL orbitals (note the significant number of virtual orbitals)
    #select_mo = len(ks.mo_occ)

    # Examples of 3 Different Localization methods
    # 1. Boys Localization
    loc_orb = lo.Boys(mol, ks.mo_coeff[:,select_mo]).kernel()
    # save orbitals in molden format
    #molden.from_mo(mol, 'boys.molden', loc_orb)

    # 2. Edmiston-Ruedenberg localization
    loc_orb = lo.ER(mol, ks.mo_coeff[:,select_mo]).kernel()
    # save orbitals in molden format
    #molden.from_mo(mol, 'edmiston.molden', loc_orb)

    # 3. Pipek-Mezey localization
    loc_orb = lo.PM(mol, ks.mo_coeff[:,select_mo]).kernel()
    # save orbitals in molden format
    #molden.from_mo(mol, 'pm.molden', loc_orb)

    # There are other localization options in PySCF!
    # see https://pyscf.org/pyscf_api_docs/pyscf.lo.html
    #############

if __name__ == "__main__":
    main()
