#!/usr/bin/env python

import numpy as np
from scipy.special import comb
import itertools
from functools import reduce
from distutils.util import strtobool
from pyscf import gto, scf, ao2mo, tools, symm, fci, mcscf
from pyscf.tools import molden
from pyscf.fci.direct_spin0 import make_rdm12
from pyscf.fci.direct_spin1 import absorb_h1e,contract_2e,energy
from pyscf.mcscf.casci import h1e_for_cas

def permute( mat, perms ):
	for p in perms:
		mat = np.dot( np.dot( p, mat ), p )
	return mat

# reads input file or uses default values
def get_user_input(filename='count.inp'):
	# default values
	d = {	'geometry'	: None,
		'Cguess'	: None,
		'Xguess'	: None,
		'eris'		: 'fcidump.txt',
		'basis'		: None,
		'core_type'	: 'noCore',
		'nelec_active'	: None,
		'norb_active'	: None,
		'active_list'	: None,
		'target_state'	: 1,
		'target_spin'	: 0,
		'omega'		: -100,
		'lambda'	: 1e-6,
		'Xrelax'	: True,
		'bfgs_Hess'	: False,
		'bfgs_thresh_initial'	: 1e-4,
		'bfgs_thresh_final'	: 1e-6,
		'bfgs_print'	: -1,
		'macro_maxiters': 500,
		'chi0'		: False,
		'debug'		: False,
		'molden'	: True,
		'output_dir'	: None
		 }

	with open(filename, 'r') as f:
		line = f.readline()
		line_number = 0
		while line:
			line_number += 1
			stripped = line.split('#')[0]  # removes commented lines
			#split_line = stripped.split(maxsplit=1)
			split_line = stripped.split()
			if len(split_line) == 1:
				raise RuntimeError("Only one word in input line %d" % line_number)
			elif len(split_line) == 2:
				d[split_line[0]] = split_line[1].strip()
			elif len(split_line) > 2:
				raise RuntimeError("Unexpectedly found more than two pieces in input line %d" % line_number)
			line = f.readline()

	# check for required parameters
	if d['output_dir'] is None: raise RuntimeError ("User must specify an output directory to write the optimized parameters into")
	if d['geometry'] is None: raise RuntimeError ("User must specify the file containing the molecular geometry")
	if d['Cguess'] is None: raise RuntimeError ("User must specify the file containing the input guess for the CI coefficients")
	if d['basis'] is None: raise RuntimeError ("User must specify the basis")
	if d['nelec_active'] is None: raise RuntimeError ("User must specify the number of active electrons")
	if d['norb_active'] is None: raise RuntimeError ("User must specify the number of active orbitals")

	# convert from strings to usable variable types
	d['debug'] = strtobool(d['debug'])
	d['chi0'] = strtobool(d['chi0'])
	d['molden'] = strtobool(d['molden'])
	d['nelec_active'] = int(d['nelec_active'])
	d['norb_active'] = int(d['norb_active'])
	d['target_state'] = int(d['target_state'])
	d['target_spin'] = int(d['target_spin'])
	d['omega'] = float(d['omega'])
	d['lambda'] = float(d['lambda'])
	d['Xrelax'] = strtobool(d['Xrelax'])
	d['bfgs_Hess'] = strtobool(d['bfgs_Hess'])
	d['bfgs_thresh_initial'] = float(d['bfgs_thresh_initial'])
	d['bfgs_thresh_final'] = float(d['bfgs_thresh_final'])
	d['bfgs_print'] = int(d['bfgs_print'])
	d['macro_maxiters'] = int(d['macro_maxiters'])

	return d


def read_fcidump( my_filename, coreList, active_list=None ):
	f = open(my_filename,'r')
	data = f.readlines()
	f.close()
	data = [ dat.strip() for dat in data ]
	norb_tot = data[0]
	norb_tot = norb_tot.split()
	norb_tot = int(norb_tot[2].split(',')[0])
	HF_oints = np.zeros((norb_tot,norb_tot))
	HF_tints = np.zeros((norb_tot,norb_tot,norb_tot,norb_tot))
	for line in data[4:-1]:
		x,i,j,k,l = line.split()
		i = int( int(i)-1 )
		j = int( int(j)-1 )
		k = int( int(k)-1 )
		l = int( int(l)-1 )
		if k==-1 and l==-1:
			HF_oints[int(i)][int(j)] = float(x)
			HF_oints[int(j)][int(i)] = float(x)
		else:
			HF_tints[int(i)][int(j)][int(k)][int(l)] = float(x)
			HF_tints[int(j)][int(i)][int(k)][int(l)] = float(x)
			HF_tints[int(i)][int(j)][int(l)][int(k)] = float(x)
			HF_tints[int(j)][int(i)][int(l)][int(k)] = float(x)
			HF_tints[int(k)][int(l)][int(i)][int(j)] = float(x)
			HF_tints[int(k)][int(l)][int(j)][int(i)] = float(x)
			HF_tints[int(l)][int(k)][int(i)][int(j)] = float(x)
			HF_tints[int(l)][int(k)][int(j)][int(i)] = float(x)

	if active_list == None:
		return HF_oints, HF_tints

	else:		# sort MOs to place core and active orbitals sequentially
		full_orb_list = [int(i-1) for i in active_list]
		for i in range(norb_tot):
			if i in coreList: full_orb_list.insert(i,i)
			if i not in full_orb_list and i not in coreList: full_orb_list.append(i)
		HF_oints = HF_oints[:,full_orb_list]
		HF_oints = HF_oints[full_orb_list,:]
		HF_tints = HF_tints[:,:,:,full_orb_list]
		HF_tints = HF_tints[:,:,full_orb_list,:]
		HF_tints = HF_tints[:,full_orb_list,:,:]
		HF_tints = HF_tints[full_orb_list,:,:,:]
		return HF_oints, HF_tints

def my_pyscf( molecule, basis, nelec_act, norb_act, active_list=None ):
	# reformat  molecule geometry for pyscf
	molecule_list = []
	for atom in molecule:
		atom_list = []
		atom_list.append(str(atom[0]))
		atom_list.append([atom[1],atom[2],atom[3]])
		molecule_list.append(atom_list)

	# build molecule in pyscf
	mol = gto.Mole()
	mol.atom = molecule_list
	mol.basis = basis
	mol.build()

	# run RHF
	mol_hf = scf.RHF(mol)
	mol_hf.kernel()

	# calculate parameters
	c_hf = mol_hf.mo_coeff
	nelec_tot = mol.nelectron	# total number of electrons
	norb_tot  = c_hf.shape[1]	# total number of MO's
	nelec_core = int(nelec_tot - nelec_act)
	norb_core = int(nelec_core/2.)
	energy_nuc = mol.energy_nuc()

	# get Cguess from CASCI calc
	print ("\nPyscf energies, CASCI:")
	if active_list == None:
		mc = mcscf.CASCI(mol_hf, norb_act, nelec_act, ncore=norb_core)
		mc.kernel()
	else:
		mc = mcscf.CASCI(mol_hf, norb_act, nelec_act, ncore=norb_core)
		mo_coeff_fix = mcscf.sort_mo(mc, mc.mo_coeff, active_list)
		mc.kernel(mo_coeff_fix)
	
	# get core energy (includes nuclear energy)
	oints_Veff, energy_nuc_core = h1e_for_cas(mc)
	del oints_Veff		# effective potential oints, don't currently need for frozen core approximation
	
	return nelec_tot, norb_tot, energy_nuc_core, energy_nuc
	
# account for frozen core in remaining oints
# add frozen core components of tints to oints
def oints_2MFcore(oints, tints, norb_core, norb_occ):
	oints_MF = np.zeros((norb_occ,norb_occ))
	for i in range(norb_core,norb_occ):
		for j in range(norb_core,norb_occ):
			oints_MF[i,j] = oints[i,j]
			for r in range(norb_core):
				oints_MF[i,j] += ( 2.0 * tints[i,j,r,r] ) - ( 1.0 * tints[i,r,r,j] ) 
	return oints_MF

# Build occupied rdms by adding in core components (pq=core)
def rdm12_act2occ(rdm1_act, rdm2_act, norb_core, norb_occ):
	idx_core = np.arange(norb_core)
	rdm1_occ = np.zeros((norb_occ,norb_occ))
	rdm1_occ[idx_core,idx_core] = 2.
	rdm1_occ[norb_core:norb_occ,norb_core:norb_occ] = rdm1_act

	rdm2_occ = np.zeros((norb_occ,norb_occ,norb_occ,norb_occ))
	rdm2_occ[norb_core:norb_occ,norb_core:norb_occ,norb_core:norb_occ,norb_core:norb_occ] = rdm2_act
	for p in range(norb_core):	# loop one index (p) over the core orbitals
		for q in range(norb_core):	# loop one index (p) over the core orbitals
			rdm2_occ[p,p,q,q] += 4.		# core^4
			rdm2_occ[p,q,q,p] += -2.	# core^4
		rdm2_occ[p,p,norb_core:norb_occ,norb_core:norb_occ] = rdm2_occ[norb_core:norb_occ,norb_core:norb_occ,p,p] = 2. * rdm1_act
		rdm2_occ[p,norb_core:norb_occ,norb_core:norb_occ,p] = rdm2_occ[norb_core:norb_occ,p,p,norb_core:norb_occ] = - rdm1_act
		
	return rdm1_occ, rdm2_occ

def my_absorb_h1e(oints, tints, norb, nelec):
	#f1e = oints
	#for p in range(norb):
	#	for q in range(norb):
	#		for r in range(norb):
	#			f1e[p,q] -= 0.5 * tints[p,r,r,q]
	f1e = oints - np.einsum('jiik->jk',tints) * 0.5
	f1e = f1e * (1./(nelec+1e-100))
	for k in range(norb):
		tints[k,k,:,:] += f1e
		tints[:,:,k,k] += f1e
	return tints * 0.5

def vis_orbs_molden( U, filename, molecule, basis ):
	# reformat  molecule geometry for pyscf
	molecule_list = []
	for atom in molecule:
		atom_list = []
		atom_list.append(str(atom[0]))
		atom_list.append([atom[1],atom[2],atom[3]])
		molecule_list.append(atom_list)

	# build molecule in pyscf
	mol = gto.Mole()
	mol.atom = molecule_list
	mol.basis = basis
	mol.build()

	# run RHF
	mol_hf = scf.RHF(mol)
	mol_hf.kernel()

	# create molden file
	with open( filename, 'w' ) as f:
		molden.header( mol, f )
		molden.orbital_coeff( mol, f, np.dot( U, mol_hf.mo_coeff ), ene=mol_hf.mo_energy, occ=mol_hf.mo_occ )

	return None

#_______________________________
# Approximate Hessian functions
#_______________________________

def prep_ham_diagonal_slow( nstr, nelec_act, norb_act, var_h2e, j ):

	jI = np.eye(nstr**2,k=j)[0,:]
	Hc = contract_2e( var_h2e, np.reshape(jI,(nstr,nstr)), norb_act, nelec_act )
	Hc = np.reshape(Hc,(-1))
	Hdiag = np.dot( jI, Hc )

	return Hdiag

def prep_ham_diagonal_fast( nstr, nelec_act, norb_act, oints, tints, ci_ind ):

	Hdiag = 0
	Ia_occ, Ib_occ = ci2strings( nstr, nelec_act, norb_act, ci_ind )
	Ia_virt = [a for a in range(norb_act) if a not in Ia_occ]
	Ib_virt = [a for a in range(norb_act) if a not in Ib_occ]
	
	# contributions from alpha orbitals
	for ia in Ia_occ:
		Hdiag += oints[ia,ia]
		for ra in range(norb_act):
			Hdiag -= 0.5 * tints[ia,ra,ra,ia]
		for ja in Ia_occ:
			Hdiag += 0.5 * tints[ia,ia,ja,ja]
		for aa in Ia_virt:
			Hdiag += 0.5 * tints[ia,aa,aa,ia]
		for jb in Ib_occ:
			Hdiag += 0.5 * tints[ia,ia,jb,jb]
	# contributions from beta orbitals
	for ib in Ib_occ:
		Hdiag += oints[ib,ib]
		for rb in range(norb_act):
			Hdiag -= 0.5 * tints[ib,rb,rb,ib]
		for jb in Ib_occ:
			Hdiag += 0.5 * tints[ib,ib,jb,jb]
		for ab in Ib_virt:
			Hdiag += 0.5 * tints[ib,ab,ab,ib]
		for ja in Ia_occ:
			Hdiag += 0.5 * tints[ib,ib,ja,ja]

	return Hdiag


# Effective 1-eri
#	g_pq = h_pq + sum_{r}^{Nspacial} ( 2*(pq|rr) - (pr|rq) )
#	Nspacial (noCore,frozenCore) = active
#	Nspacial (closedCore) = occupied
def prep_eff_oint( i, j, MO_oints_tot, MO_tints_tot, norb_cut_start, norb_occ ):
	oint_MF = MO_oints_tot[i,j]
	for p in range(norb_cut_start,norb_occ):
		oint_MF += ( 2.0 * MO_tints_tot[i,j,p,p] ) - ( 1.0 * MO_tints_tot[i,p,p,j] ) 
	return oint_MF

# Orbital energy hessian, approximate as diagonal and with Fock operator
# 	Hess_orb = (1/2) * (1+Pijkl) * <psi| [Eij, [Ekl, F]] |psi> 
# ...approximate as diagonal, where ij is a compound index
# 	Hess_orb_{ij,ij} = (1/2) * (1 + P_{ij,ij}) * <psi| [Eij, [Eij, F]] |psi> 
# 			 = <psi| [Eij, [Eij, F]] |psi> 
def build_Hess_E_orb( Xlen, Xindices, coreList, actvList, virtList, eff_oint, rdm1, norb_cut_start):

	Hess_E_orb = np.zeros((Xlen))
	for indHess in range(Xlen):
		i = Xindices[indHess][0]
		j = Xindices[indHess][1]
	
		# i=core, j=active (closedCore only)
		if i in coreList and j in actvList:
			Hess_E_orb[indHess] +=  2.0 * eff_oint(i,i) * rdm1[j-norb_cut_start,j-norb_cut_start]
			Hess_E_orb[indHess] +=  2.0 * eff_oint(j,j) * rdm1[i-norb_cut_start,i-norb_cut_start]
			Hess_E_orb[indHess] += -2.0 * eff_oint(i,i) * rdm1[i-norb_cut_start,i-norb_cut_start]
			for w in actvList:
				Hess_E_orb[indHess] += -2.0 * eff_oint(j,w) * rdm1[j-norb_cut_start,w-norb_cut_start]
		# i=core, j=virtual (closedCore only)
		if i in coreList and j in virtList:
			Hess_E_orb[indHess] +=  2.0 * eff_oint(j,j) * rdm1[i-norb_cut_start,i-norb_cut_start]
			Hess_E_orb[indHess] += -2.0 * eff_oint(i,i) * rdm1[i-norb_cut_start,i-norb_cut_start]
		# i=active, j=virtual (all core types)
		if i in actvList and j in virtList:
			Hess_E_orb[indHess] +=  2.0 * eff_oint(j,j) * rdm1[i-norb_cut_start,i-norb_cut_start]
			for w in actvList:
				Hess_E_orb[indHess] += -2.0 * eff_oint(i,w) * rdm1[i-norb_cut_start,w-norb_cut_start]

	return Hess_E_orb


#_______________________________
# String functions
#_______________________________

"""Function that compares strings for reverse colexical ordering
input: two strings (arrays)
output: True or False
"""
def colexical(I,J):
	diff = [J.index(j) for j in J if j not in I]
	if J[diff[-1]] > I[diff[-1]]:
		sort = True
	else:
		sort = False
	return sort

"""Function that sorts array in reverse colexical ordering using a bubble sort algorithm
input: array of strings in any order
output: array of strings in reverse colexical order
"""
def colexicalSort(arr):
	sort = False
	while sort == False:
		swaps = 0
		for j in range(len(arr)-1):     # traverse through all array elements
			# Swap if the element found non-colexical with the next element
			if colexical(arr[j],arr[j+1]) == False:
				swaps += 1
				arr[j], arr[j+1] = arr[j+1], arr[j]
		if swaps == 0:
			sort = True
	return arr

'''strings: array of all possible strings sorted in reverse colexical ordering
'''
def strings( nelec, norb ):

	nelec_a = int(nelec/2.)
	combos = itertools.combinations(range(0,norb), nelec_a)
	strings = []
	for word in combos:
		split = [char for char in word]
		strings.append(split)
	colexicalSort(strings)

	return strings

'''Tracks ci index back to the associated alpha/beta strings
input: index in C_vec (nstr**2)
output: alpha string, beta string
'''
def ci2strings( nstr, nelec, norb, ind_vec ):
	ind_mat = np.unravel_index(ind_vec,(nstr,nstr))
	a_str = strings( nelec, norb )[ind_mat[0]]
	b_str = strings( nelec, norb )[ind_mat[1]]

	return a_str, b_str
