## Computation of RESP Charges.
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

    qcschema_json = "examples/heh_output.json"
    qcwavefunction_h5 = "examples/heh_output.h5"

    # Load Accelerated DFT output json
    qcschema_dict = load_qcschema_json(qcschema_json)

    # Load wavefunction from hdf5
    qcwavefunction = {}
    qcwavefunction['wavefunction'] = read_hdf5_wavefunction(qcwavefunction_h5)

    # add wfn info to the total qcschema output
    qcschema_dict.update(qcwavefunction)

    # Create DFT object
    mol, ks = recreate_scf_obj(qcschema_dict)

    # Compute Milliken charges, spin density and Dipole moment
    # note: need uhf or hf in analyze call
    an = pyscf.scf.uhf.analyze(ks)

    # Compute < \hat{S}^2 > amd 2S+1 using occupied MOs
    # note hf or uhf in call
    mo = (ks.mo_coeff[0][:,ks.mo_occ[0]>0], ks.mo_coeff[1][:,ks.mo_occ[1]>0])
    print('S^2 = %.7f, 2S+1 = %.7f' % pyscf.scf.uhf.spin_square(mo, mol.intor('int1e_ovlp_sph')))

if __name__ == "__main__":
    main()
