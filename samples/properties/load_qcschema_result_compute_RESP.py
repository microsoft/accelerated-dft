## Computation of RESP Charges.
## Loads QCSchema format json and hdf5 result from Accelerated DFT
## and computes dipole moment.
## Accelerated DFT task can be energy, gradient, hessian,
##  or geometry optimization
##  with the wavefunction output e.g. "write_wavefunction": "last"
import pyscf
from tools.libqcschema import *
from pyscf import gto, dft
import json
import numpy as np
import tools.resp
from tools.resp import resp
from tools.libqcschema import *
from tools.wavefunction_hdf5_to_qcschema import *
import argparse

def main():
    parser = argparse.ArgumentParser(description="Compute RESP charges from QCSchema and wavefunction files.")
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

    #### Compute RESP Charges ####
    # Assigns charge to each atom by fitting the electrostatic potential evaulated on a grid.
    # Function returns a list of charges.
    # The sum of the charges is constrained to the molecular charge given.
    # Charges are restrained using a penlaty function so they don't grow too large .
    # By default, Hydrogen is the only atom unrestrained (i.e. it is FREE), but this can be changed by setting 'IHFREE': False.
    # Below are 4 separate examples of computing RESP charges.

    # 1. simplest case, default options.
    print("")
    print('1. partial charge with RESP')
    options = {}
    q = resp(ks,options)
    print(q) # default

    # 2. Adding options. Set VDW radii scheme. and save the results to file.
    # Note: generates new files.
    # Also turn off printing
    print("")
    print('2. partial charge with RESP')
    options = {'VDW_SCHEME': 'VDW_mod', 'SAVE': True}
    q = resp(ks,options)
    print(q) 

    # 3. Hydrogens charge is not restricted by default, restricting here.
    #    Also turning off printing in resp code and print here instead 
    print("")
    print('3. partial charge with RESP')
    options = {'VDW_SCHEME': 'VDW', 'IHFREE': False, 'PRINTQ': False}
    q = resp(ks,options)
    print(q)
    print("")
    #############


if __name__ == "__main__":
    main()
