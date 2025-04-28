## Compute g-Tensor.
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

def main():

    qcschema_json = "examples/aspirin_charged_output.json"
    qcwavefunction_h5 = "examples/aspirin_charged_output.h5"

    # Load Accelerated DFT output json
    qcschema_dict = load_qcschema_json(qcschema_json)

    # Load wavefunction from hdf5
    qcwavefunction = {}
    qcwavefunction['wavefunction'] = read_hdf5_wavefunction(qcwavefunction_h5)

    # add wfn info to the total qcschema output
    qcschema_dict.update(qcwavefunction)

    # Create DFT object
    mol, ks = recreate_scf_obj(qcschema_dict)

    #### G-Tensor ####
    # first populate Grid that scf would normally populate
    ks.grids = dft.gen_grid.Grids(mol)
    ks.grids.build(with_non0tab=True)
    print("*** G-Tensor Ouput ***")
    from pyscf.prop import gtensor
    # create object
    gobj = gtensor.uks.GTensor(ks).set(verbose=4)
    # settings may be manipulated, for details see
    # https://github.com/pyscf/properties/blob/master/examples/02-g_tensor.py
    gobj.kernel()
    #############

if __name__ == "__main__":
    main()
