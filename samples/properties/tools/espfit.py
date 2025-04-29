"""
Fitting procedure for RESP charges.

Reference:
Equations taken from Bayly, J.Phys.Chem,97,10271 (1993)
"""
# Adapted and Extended from Psi4NumPy by Microsoft under BSD-3.
# Original credit:
#__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
#__license__   = "BSD-3-Clause"
#
import copy
import warnings
import numpy as np


def esp_solve(A, B):
    """Solves for point charges: A*q = B

    Parameters
    ----------
    A : ndarray
        array of matrix A
    B : ndarray
        array of matrix B

    Return
    ------
    q : ndarray
        array of charges

    """

    q = np.linalg.solve(A, B)
    # Warning for near singular matrix
    # in case np.linalg.solve does not detect singularity
    if np.linalg.cond(A) > 1/np.finfo(A.dtype).eps:
        warnings.warn("Possible fit problem; singular matrix")

    return q


def restraint(q, A_unrestrained, resp_a, resp_b, ihfree, symbols):
    """Adds hyperbolic restraint to matrix A

    Parameters
    ----------
    q : ndarray
        array of charges
    A_unrestrained : ndarray
        array of unrestrained A matrix
    resp_a : float
        restraint scale a
    resp_b : float
        restraint parabola tightness b
    ihfree : bool
        whether hydrogens are excluded or included in restraint
    symbols : ndarray
        array of element symbols

    Returns
    -------
    a : ndarray
        restrained A array
    """

    # hyperbolic Restraint
    # Bayly, J.Phys.Chem,97,10271 (1993) (Eqs. 10,13)
    A = copy.deepcopy(A_unrestrained)
    for i in range(len(symbols)):
        # if an element is not hydrogen or if hydrogens are to be restrained
        if not ihfree or symbols[i] != '1':
            A[i, i] = A_unrestrained[i, i] + resp_a/np.sqrt(q[i]**2 + resp_b**2) 

    return A


def iterate(q, A_unrestrained, B, resp_a, resp_b, ihfree, symbols, toler, maxit):
    """Iterates the RESP fitting procedure

    Parameters
    ----------
    q : ndarray
        array of initial charges
    A_unrestrained : ndarray
        array of unrestrained A matrix
    B : ndarray
        array of matrix B
    resp_a : float
        restraint scale a
    resp_b : float
        restraint parabola tightness b
    ihfree : bool
        whether hydrogens are excluded or included in restraint
    symbols : ndarray
        array of element symbols
    toler : float
        tolerance for charges in the fitting
    maxit : int
        maximum number of iterations

    Returns
    -------
    q : ndarray
        array of the fitted charges

    """
    q_last = copy.deepcopy(q)
    niter, dif, note = 0, 2*toler, ''
    while dif > toler and niter < maxit:
        niter += 1
        A = restraint(q, A_unrestrained, resp_a, resp_b, ihfree, symbols)
        q = esp_solve(A, B)
        # Extract vector elements that correspond to charges
        dif = np.sqrt(np.max((q[:len(symbols)] - q_last[:len(symbols)])**2))
        q_last = copy.deepcopy(q)

    if dif > toler:
        note += ('\nCharge fitting did not converge; ' + 
               'try increasing the maximum number of iterations to ' +
               '> %i.' %maxit)
    return q[:len(symbols)], note


def intramolecular_constraints(constraint_charge, constraint_groups):
    """Extracts intramolecular constraints from user constraint input

    Parameters
    ----------
    constraint_charge : list
        list of lists of charges and atom indices list
        e.g. [[0, [1, 2]], [1, [3, 4]]]
        The sum of charges on 1 and 2 will equal 0
        The sum of charges on 3 and 4 will equal 1
    constraint_group : list
        list of lists of indices of atoms to have equal charge
        e.g. [[1, 2], [3, 4]]
        atoms 1 and 2 will have equal charge
        atoms 3 and 4 will have equal charge

    Returns
    -------
    constrained_charges : list
        list of fixed charges
    constrained_indices : list
        list of lists of indices of atoms in a constraint
        negative number before an index means
        the charge of that atom will be subtracted.

    Notes
    -----
    Atom indices starts with 1 not 0.
    Total charge constraint is added by default for the first molecule.

    """
    constrained_charges = []
    constrained_indices = []
    for i in constraint_charge:
        constrained_charges.append(i[0])
        group = []
        for k in i[1]:
            group.append(k)
        constrained_indices.append(group)

    for i in constraint_groups:
        for j in range(1, len(i)):
            group = []
            constrained_charges.append(0)
            group.append(-i[j-1])
            group.append(i[j])
            constrained_indices.append(group)

    return constrained_charges, constrained_indices


def fit(options, data):
    """Performs ESP and RESP fits.

    Parameters
    ----------
    options : list
        list of dictionaries of fitting options and internal data

    Returns
    -------
    qf : list
        list of ndarrays of fitted charges
    labelf : list
        list of strings of fitting methods i.e. ESP and RESP
    note : str
        string of notes on the fitting

    """
    qf = []
    labelf = []
    constraint_charges, constraint_indices = intramolecular_constraints(options['CONSTRAINT_CHARGE'],
                                                                        options['CONSTRAINT_GROUP'])
    natoms = data['natoms']
    ndim = natoms + 1 + len(constraint_charges) 
    A = np.zeros((ndim, ndim))
    B = np.zeros(ndim)

    # Bayly, J.Phys.Chem,97,10271 (1993) (Eqs. 12-14)
    for mol in range(len(data['invr'])):
        r_inverse, V = data['invr'][mol], data['esp_values'][mol]

        # Lower case a and b are the A matrix and B vector for one molecule
        # and without the addition of constraints

        # Construct a: a_jk = sum_i [(1/r_ij)*(1/r_ik)]
        a = np.einsum("ij, ik -> jk", r_inverse, r_inverse)

        # Construct b: b_j = sum_i (V_i/r_ij)
        b = np.einsum('i, ij->j', V, r_inverse)

        A[:natoms, :natoms] += a
        B[:natoms] += b

    # Add total charge constraint
    A[:natoms, natoms] = 1
    A[natoms, :natoms] = 1
    B[natoms] = data['mol_charge']

    # Add constraints to matrices A and B
    for i in range(len(constraint_charges)):
        B[natoms + 1 + i] = constraint_charges[i]
        for k in constraint_indices[i]:
            if k > 0:
                A[natoms + 1 + i, k - 1] = 1
                A[k - 1, natoms + 1 + i] = 1
            else:
                A[natoms + 1 + i, -k - 1] = -1
                A[-k - 1, natoms + 1 + i] = -1

    labelf.append('ESP')
    q = esp_solve(A, B)
    qf.append(q[:natoms])
    if not options['RESTRAINT']:
        return qf, labelf, ''
    else:
        # Restrained ESP
        labelf.append('RESP')
        q, note = iterate(q, A, B, options['RESP_A'], options['RESP_B'], options['IHFREE'], data['symbols'], options['TOLER'], options['MAX_IT'])
        qf.append(q)
        return qf, labelf, note
