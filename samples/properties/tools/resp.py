"""
Driver for the RESP code.
"""
# Adapted and Extended from Psi4NumPy by Microsoft under BSD-3.
# Original credit:
#__authors__   =  "Asem Alenaizan"
#__credits__   =  ["Asem Alenaizan"]
#
#__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
#__license__   = "BSD-3-Clause"
#__date__      = "2018-04-28"
#
import os

import numpy as np
import scipy
from . import espfit
from . import vdw_surface

import pyscf
from pyscf.lib.parameters import BOHR
from pyscf import data

bohr_to_angstrom = BOHR #0.52917721092


def resp(mf, options=None):
    """RESP code driver.
    
    see: Bayly, J.Phys.Chem,97,10271 (1993)

    Assigns charge to each atom by fitting the electrostatic potential evaulated on a grid.
    Function returns a list of charges.
    The sum of the charges is constrained to the molecular charge given.
    Charges are restrained using a penlaty function so they don't grow too large .
    By default, Hydrogen is the only atom unrestrained (i.e. it is FREE), but this can be changed by setting 'IHFREE': False.

    Parameters
    ---------- 
    molecules : list
        list of psi4.Molecule instances
    options_list : dict, optional
        a dictionary of user's defined options
        RESP_A : float
                 restraint scale a
        RESP_B : float
                 restraint parabola tightness b
        IHFREE : bool
                 whether hydrogens are excluded or included in restraint
        PRINTQ : bool
                 Default=False. Whether to print charges and scheme at end of this function.
        SAVE   : bool
                 Default=False. Whether to save a result.dat and grid points and ESP values (at the grid points) to file.
        VDW_SCALE_FACTORS : list of floats
                            Default = [1.2]. Scales the VDW radii by this factor, and puts grid on it.
        VDW_POINT_DENSITY : float
                            Default=1.0. Controls grid point density.  Warning: computation time will increase with more grid points.

    Returns
    -------
    charges : list
        list of RESP atomic charges

    Note
    ----
    output files : (if 'SAVE' == True)
                   results.dat: fitting results
                   grid.dat: grid points in Bohr
                   grid_esp.dat: QM esp valuese in a.u. 
    """
    import pyscf
    from pyscf.lib.parameters import BOHR
    from pyscf import data

    if options is None:
        options = {}

    # Check options
    # RESP options have large case keys
    options = {k.upper(): v for k, v in sorted(options.items())}

    # VDW surface options
    if 'ESP' not in options:
        options['ESP'] = []
    if 'GRID' not in options:
        options['GRID'] = []
    if 'VDW_SCALE_FACTORS' not in options:
        options['VDW_SCALE_FACTORS'] = [1.2] #[1.2, 1.4, 1.6, 1.8, 2.0]
    if 'VDW_POINT_DENSITY' not in options:
        options['VDW_POINT_DENSITY'] = 1.0
    # Hyperbolic restraint options
    if 'RESTRAINT' not in options:
        options['RESTRAINT'] = True
    if options['RESTRAINT']:
        if 'RESP_A' not in options:
            options['RESP_A'] = 0.0005
        if 'RESP_B' not in options:
            options['RESP_B'] = 0.1
        if 'IHFREE' not in options:
            options['IHFREE'] = True
        if 'TOLER' not in options:
            options['TOLER'] = 1e-5
        if 'MAX_IT' not in options:
            options['MAX_IT'] = 25

    # check print
    if 'PRINTQ' not in options:
        options['PRINTQ'] = False
    # Check save
    if 'SAVE' not in options:
        options['SAVE'] = False

    # VDW surface options
    # all VDw scheme in PyCSF data are in Bohr, so need to covert to Angstrom
    # For user defined radii: User should directly supply Angstrom.
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
        # they are also all defined in Bohr, so convert to Angstrom
        vdw_array = vdw_array[1:]#*bohr_to_angstrom
        # convert to Dict
        radii = dict(enumerate(vdw_array.flatten(), 1))

    # or a user defined scheme?
    if options['VDW_SCHEME'] == 'USER':
        radii = options['VDW_RADII']
        # convert to Bohr - NO, this resp uses Angstrom. User should input angstrom.
        for key in radii:
            radii[key] /= BOHR

    
    #print("Using VDW Scheme: ", options['VDW_SCHEME'])

    # Constraint options
    if 'CONSTRAINT_CHARGE' not in options:
        options['CONSTRAINT_CHARGE'] = []
    if 'CONSTRAINT_GROUP' not in options:
        options['CONSTRAINT_GROUP'] = []

    data = {}
    data['natoms'] = mf.mol.natm #molecules[0].natom()
    data['symbols'] = []
    data['symbols'] = mf.mol.elements # do i need unique ones? like make this a set?
    data['atom_types'] = np.array(mf.mol._atm[:, 0])
    data['mol_charge'] = mf.mol.charge #molecules[0].molecular_charge()

    data['coordinates'] = []
    data['esp_values'] = []
    data['invr'] = []

    data['coordinates'] = mf.mol.atom_coords(unit='B') #(mf.mol.atom_coords(unit='B'))*bohr_to_angstrom
    coordinates = mf.mol.atom_coords(unit='B') #(mf.mol.atom_coords(unit='B'))*bohr_to_angstrom
    if options['GRID']:
        # Read grid points
        points = []
        points = np.loadtxt('grid.dat')
        #if 'Bohr' in str(molecules[imol].units()):
        #    points *= bohr_to_angstrom
        # read Points in Bohr
        print("Reading Grid points (in Bohr) from file grid.dat")
    else:
        # Get the points at which we're going to calculate the ESP
        points = []
        for scale_factor in options['VDW_SCALE_FACTORS']:
            # pass atomic number
            shell, radii_scaled = vdw_surface.vdw_surface(coordinates, data['atom_types'], scale_factor,
                                options['VDW_POINT_DENSITY'], radii)
            points.append(shell)
        points = np.concatenate(points)
        #print("points",points)
        #if 'Bohr' in str(molecules[imol].units()):
        #    points /= bohr_to_angstrom
        #    np.savetxt('grid.dat', points, fmt='%15.10f')
        #    points *= bohr_to_angstrom
        #else:
        #    np.savetxt('grid.dat', points, fmt='%15.10f')
        if(options['SAVE'] == True):
                np.savetxt('grid.dat', points, fmt='%15.10f')

        #coordinates = coordinates/BOHR

    # Calculate ESP values at the grid
    if options['ESP']:
        # Read electrostatic potential values
        #data['esp_values'].append(np.loadtxt(options['ESP'][imol]))
        #np.savetxt('grid_esp.dat', data['esp_values'][-1], fmt='%15.10f')
        print("Read ESP from file: Not Yet Implemented")
        return
    else:
        from pyscf import gto, scf, lib
        # ESP nuclear contribution:
        r_pX = scipy.spatial.distance.cdist(coordinates, points)
        r_pX = np.array(r_pX)
        r_pX_potential = 1/r_pX
        # nuclear part of electrostatic potential (ESP)
        potential_real = np.dot(np.array(mf.mol.atom_charges()), r_pX_potential)

        # add in the electronic part of ESP..need to do in batches if large
        # amount of memory required: ngrdis*(NBasis*NBasis)*8, divide by (1024**3) for GB
        ## non-batched:
        ##  Vele = np.einsum('pij,ij->p', mf.mol.intor('int1e_grids', grids=gridcoords), mf.make_rdm1())
        ##  potential_real -= Vele
        ## batched:
        ngrids = len(points)
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
                grid_bit = points[ibatch:max_grid]
                potential_real[ibatch:max_grid] -= np.einsum('pij,ij->p', mf.mol.intor('int1e_grids', grids=grid_bit), dm)
        except:
            print("Out of memory in func resp. Make batch size (nbatch) smaller")
            return()

        data['esp_values'].append(potential_real)

        if(options['SAVE'] == True):
                np.savetxt('grid_esp.dat', data['esp_values'], fmt='%15.10f')

    # Build a matrix of the inverse distance from each ESP point to each nucleus
    invr = np.zeros((len(points), len(coordinates)))
    for i in range(invr.shape[0]):
        for j in range(invr.shape[1]):
            invr[i, j] = 1/np.linalg.norm(points[i]-coordinates[j])
    data['invr'].append(invr)
    #data['invr'].append(invr*bohr_to_angstrom) # convert to atomic units
    #data['coordinates'][-1] /= bohr_to_angstrom # convert to angstroms

    # Calculate charges
    qf, labelf, notes = espfit.fit(options, data)
   
    # Write the results to disk
    if(options['SAVE'] == True):
        with open("results.out", "w") as f:
            f.write("Electrostatic potential parameters\n")
            if not options['GRID']:
                f.write("    van der Waals radii (Bohr):\n")
                for i, j in radii.items():
                    f.write("                               %8s%8.3f\n" %(i, j/scale_factor))
                f.write("    VDW scale factors:              ")
                for i in options["VDW_SCALE_FACTORS"]:
                    f.write('%6.2f' %i)
                f.write('\n')
                f.write("    VDW point density:                %.3f\n" %(options["VDW_POINT_DENSITY"]))

            f.write("\nGrid information (see %i_%s_grid.dat in %s)\n")
            f.write("    Number of grid points:            %d\n" %len(data['esp_values']))
            f.write("\nQuantum electrostatic potential (see grid_esp.dat)\n")

            f.write("\nConstraints\n")
            if options['CONSTRAINT_CHARGE']:
                f.write("    Charge constraints\n")
                for i in options['CONSTRAINT_CHARGE']:
                    f.write("        Total charge of %12.8f on the set" %i[0])
                    for j in i[1]:
                        f.write("%4d" %j)
                    f.write("\n")
            if options['CONSTRAINT_GROUP']:
                f.write("    Equality constraints\n")
                f.write("        Equal charges on atoms\n")
                for i in options['CONSTRAINT_GROUP']:
                    f.write("                               ")
                    for j in i:
                        f.write("%4d" %j)
                    f.write("\n")

            f.write("\nRestraint\n")
            if options['RESTRAINT']:
                f.write("    Hyperbolic restraint to a charge of zero\n")
                if options['IHFREE']:
                    f.write("    Hydrogen atoms are not restrained\n")
                f.write("    resp_a:                           %.4f\n" %(options["RESP_A"]))
                f.write("    resp_b:                           %.4f\n" %(options["RESP_B"]))

            f.write("\nFit\n")
            f.write(notes)
            f.write("\nElectrostatic Potential Charges\n")
            f.write("   Center  Symbol")
            for i in labelf:
                f.write("%10s" %i)
            f.write("\n")
            for i in range(data['natoms']):
                f.write("  %5d    %s     " %(i+1, data['symbols'][i]))
                for j in qf:
                    f.write("%12.8f" %j[i])
                f.write("\n")
            f.write("Total Charge:    ")
            for i in qf:
                f.write("%12.8f" %np.sum(i))
            f.write('\n')

    # print output
    if(options['PRINTQ']):
        print('VDW SCHEME', options['VDW_SCHEME'])
        print("charges:", qf[1])

    #np.savetxt('grid.dat', points, fmt='%15.10f')
    # return charges. qf[0] are ESP values. qf[1] are the charges
    return qf[1]
