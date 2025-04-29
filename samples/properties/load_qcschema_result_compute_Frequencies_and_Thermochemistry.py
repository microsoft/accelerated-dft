## Computation of Vibrational Frequencies 
## Loads QCSchema format json and hdf5 result from Accelerated DFT
## and computes dipole moment.
## Accelerated DFT task must be 'hessian',
##  with the wavefunction output e.g. "write_wavefunction": "last"
import pyscf
from pyscf import gto, dft
from pyscf import hessian
from pyscf.hessian.thermo import *
import json
import numpy as np
from tools.libqcschema import *
from tools.wavefunction_hdf5_to_qcschema import *
import argparse

def main():
    parser = argparse.ArgumentParser(description="Compute frequencies and thermochemistry from QCSchema and wavefunction files.")
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

    # Load Accelerated DFT qschema_output json and Create DFT object
    mol, ks = recreate_scf_obj(qcschema_dict)

    # Load Hessian from QcSchema output
    h = load_qcschema_hessian(qcschema_dict)
    
    # Compute Vibrational Frequencies
    freq = harmonic_analysis(mol,h)
    dump_normal_mode(mol,freq)

    # Compute Thermochemistry
    thermochem = thermo(ks,freq['freq_au'], 298.15)
    print("Thermochem:",thermochem)

if __name__ == "__main__":
    main()
