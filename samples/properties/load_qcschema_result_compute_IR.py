## Computation of Vibrational Frequencies and IR Spectrum.
## Loads QCSchema format json and hdf5 result from Accelerated DFT
## and computes dipole moment.
## Accelerated DFT task must be hessian,
##  with the wavefunction output e.g. "write_wavefunction": "last"
import pyscf
from pyscf import gto, dft
from pyscf import hessian
from pyscf.hessian.thermo import *
from pyscf.prop import infrared
import json
import numpy as np
from tools.libqcschema import *
from tools.wavefunction_hdf5_to_qcschema import *

def main():

    qcschema_json = "examples/c2h4_fh_output.json"
    qcwavefunction_h5 = "examples/c2h4_fh_output.h5"

    # Load Accelerated DFT output json
    qcschema_dict = load_qcschema_json(qcschema_json)

    # Load wavefunction from hdf5
    qcwavefunction = {}
    qcwavefunction['wavefunction'] = read_hdf5_wavefunction(qcwavefunction_h5)

    # add wfn info to the total qcschema output
    qcschema_dict.update(qcwavefunction)

    # Load Accelerated DFT qschema_output json and Create DFT object
    mol, ks = recreate_scf_obj(qcschema_dict)

    # Form Hessian object and Load Hessian from QCSchema json
    hessian = ks.Hessian()
    hessian.de = load_qcschema_hessian(qcschema_dict)
    print("ks.hessian.de",hessian.de)

    # Compute Vibrational Frequencies
    freq = harmonic_analysis(mol,hessian.de)
    dump_normal_mode(mol,freq)
    
    # Compute Thermochemistry if desired
    #thermochem = thermo(ks,freq['freq_au'], 298.15)
    #print("Thermochem:",thermochem)

    ###############################
    ##### Compute IR Spectrum #####
    ###############################
    # make IR object and populate with info
    ks_ir = prepare_ir(ks,hessian,freq)

    # compute IR intensities
    infrared.rhf.kernel_dipderiv(ks_ir)
    ir_intensity = infrared.rhf.kernel_ir(ks_ir)
    # summarise
    ks_ir.summary()
    # plot - uncomment to show
    #fig = ks_ir.plot_ir()[0]
    #fig.show()
    #fig.savefig("ir_spectrum_C2H4.png")
    ###############################

if __name__ == "__main__":
    main()
