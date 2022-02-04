#!/usr/bin/env python

import sys
import time
import numpy as np
from numpy import concatenate, reshape, zeros
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

def prep_Xmat2flat( Xmat, norb_core, norb_act, norb_virt, coreType ):
	if coreType == 'closedCore':
		Xflat = concatenate( np.flatten( Xmat[:norb_core,norb_core:] ), np.flatten( Xmat[norb_core:norb_occ,norb_occ:] ) )
		Xcaav = concatenate((reshape(Xflat[:(norb_core*(norb_act+norb_virt))],[norb_core,norb_act+norb_virt]),concatenate((np.zeros((norb_act,norb_act)), reshape(Xflat[(norb_core*(norb_act+norb_virt)):], [norb_act,norb_virt]) ), axis=1) ), axis=0)
	else:
		Xcaav = concatenate((zeros((norb_core,norb_act+norb_virt)),concatenate((zeros((norb_act,norb_act)),reshape(Xflat,(norb_act,norb_virt))), axis=1)), axis=0)

# splits flat array into X (vector) and C (matrix) variables
def prep_split_array( Vflat, Xlen, nstr ): #, norb_core, norb_act, norb_virt, coreType ):
	Vflat = reshape(Vflat,[-1])
	Cflat = reshape(Vflat[Xlen:], [nstr**2])
	Xflat = Vflat[:Xlen]
	return Xflat, Cflat

# reads input file or uses default values
def get_user_input(filename='count.inp'):
	# default values
	d = {	'geometry'		: None,
		'Cguess'		: None,
		'Xguess'		: None,
		'eris'			: 'fcidump.txt',
		'basis'			: None,
		'core_type'		: 'noCore',
		'nelec_active'		: None,
		'norb_active'		: None,
		'active_list'		: None,
		'symmetry_list'		: None,
		'target_state'		: 1,
		'target_spin'		: 0,
		'omega'			: -100,
		'lambda'		: 1e-6,
		'hess_shift'		: 0.5,
		'hess_shift_signed_mu'	: -1,
		'Xrelax'		: True,
		'Xrelax_bfgs_hess'	: False,
		'bfgs_hess'		: False,
		'bfgs_thresh_initial'	: 1e-4,
		'bfgs_thresh_final'	: 1e-6,
		'bfgs_thresh_Xrelax'	: 1e-5,
		'bfgs_print'		: -1,
		'macro_maxiters'	: 500,
		'micro_maxiters'	: 5000,
		'chi0'			: False,
		'debug'			: False,
		'molden'		: True,
		'output_dir'		: None
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
	d['hess_shift'] = float(d['hess_shift'])
	d['hess_shift_signed_mu'] = float(d['hess_shift_signed_mu'])
	d['Xrelax'] = strtobool(d['Xrelax'])
	d['Xrelax_bfgs_hess'] = strtobool(d['Xrelax_bfgs_hess'])
	d['bfgs_hess'] = strtobool(d['bfgs_hess'])
	d['bfgs_thresh_initial'] = float(d['bfgs_thresh_initial'])
	d['bfgs_thresh_final'] = float(d['bfgs_thresh_final'])
	d['bfgs_thresh_Xrelax'] = float(d['bfgs_thresh_Xrelax'])
	d['bfgs_print'] = int(d['bfgs_print'])
	d['macro_maxiters'] = int(d['macro_maxiters'])
	d['micro_maxiters'] = int(d['micro_maxiters'])

	return d

def build_molecule( user_inputs ):

	# import molecule from user inputs
	molecule = []
	with open( user_inputs['geometry'],'r') as f:
		for line in f:
			molecule.append( line )
	molecule = molecule[1:]
	natoms = len(molecule)
	if user_inputs['active_list'] == None:
		orb_list = None
	else:
		orb_list = [int(s) for s in user_inputs['active_list'].split(',')]
	
	# print molecule geometry
	tmp = "MOLECULE"
	print ( "\n%s" %tmp + " "*(abs(8-len(tmp)))  + " %6s" %("X") + " %16s" %("Y") + " %16s" %("Z") )
	for i in range(0,natoms):
		atom = molecule[i]
		atom = atom.split(',')
		atom[1] = float(atom[1])
		atom[2] = float(atom[2])
		atom[3] = float(atom[3])
		molecule[i] = atom
		print ( "%s" %atom[0] + " "*(abs(8-len(atom[0]))) + " %16s" %("%.10f	%.10f	%.10f" %(atom[1],atom[2],atom[3]) ))
	print ('\n\n')

	return molecule, orb_list

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


	#if active_list == None:
	#	print ("didn't find active_list")
	#	exit()
	#	return HF_oints, HF_tints

	#else:		# sort MOs to place core and active orbitals sequentially
	#	full_orb_list = [int(i-1) for i in active_list]
	#	for i in range(norb_tot):
	#		if i in coreList: full_orb_list.insert(i,i)
	#		if i not in full_orb_list and i not in coreList: full_orb_list.append(i)

	#	print (full_orb_list)
	#	HF_oints = HF_oints[:,full_orb_list]
	#	HF_oints = HF_oints[full_orb_list,:]
	#	HF_tints = HF_tints[:,:,:,full_orb_list]
	#	HF_tints = HF_tints[:,:,full_orb_list,:]
	#	HF_tints = HF_tints[:,full_orb_list,:,:]
	#	HF_tints = HF_tints[full_orb_list,:,:,:]
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
	#c_hf = np.loadtxt('HF_MOcoeff.txt')
	nelec_tot = mol.nelectron	# total number of electrons
	norb_tot  = c_hf.shape[1]	# total number of MO's
	nelec_core = int(nelec_tot - nelec_act)
	norb_core = int(nelec_core/2.)
	energy_nuc = mol.energy_nuc()

	#with open( 'HF.molden', 'w' ) as f:
	#	molden.header( mol, f )
	#	molden.orbital_coeff( mol, f, c_hf, ene=mol_hf.mo_energy, occ=mol_hf.mo_occ )

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

##def natural_orbitals( rdm1 ):
#def natural_orbitals( C, norb_act, nelec_act ):
#	## reformat  molecule geometry for pyscf
#	#molecule_list = []
#	#for atom in molecule:
#	#	atom_list = []
#	#	atom_list.append(str(atom[0]))
#	#	atom_list.append([atom[1],atom[2],atom[3]])
#	#	molecule_list.append(atom_list)
#	## build molecule in pyscf
#	#mol = gto.Mole()
#	#mol.atom = molecule_list
#	#mol.basis = basis
#	#mol.build()
#	## run RHF
#	#mol_hf = scf.RHF(mol)
#	#mol_hf.kernel()
#	#mocoeff = np.dot( mol_hf.mo_coeff, U )
#	#mc = mcscf.CASSCF( mol_hf, norb_act, nelec_act, ncore=norb_core, frozen=norb_core)
#	#nat_orbs = mcscf.casci.cas_natorb( mol_hf, mocoeff, sort=True, casdm1=rdm1, with_meta_lowdin=True )
#
#	C =/ np.linalg.norm( C )
#	rdm1,rdm2 = make_rdm12( C, norb_act, (int(nelec_act/2.),int(nelec_act/2.)) )
#	nat_orbs = np.eig( rdm1 )
#
#	#sort = np.argsort(nat_orbs.round(9), kind='mergesort')
#	#nat_orbs = nat_orbs[sort]
#	
#	return nat_orbs


def vis_orbs_molden( U, filename, molecule, basis, norb_core, norb_occ, norb_tot, nelec_act, eigvecs_cas=None, natorbs=None, active_list=None, core_list=None):
#def vis_orbs_molden( U, filename, molecule, basis, norb_core, norb_occ, eigvecs_cas=None, natorbs=None, active_list=None, core_list=None):
	# reformat  molecule geometry for pyscf
	molecule_list = []
	for atom in molecule:
		atom_list = []
		atom_list.append(str(atom[0]))
		atom_list.append([atom[1],atom[2],atom[3]])
		molecule_list.append(atom_list)

	# build molecule in pyscf
	mol = gto.Mole()
	mol.verbose = 5
	mol.atom = molecule_list
	mol.basis = basis
	mol.symmetry = 'c2v'
	mol.build()

	# run RHF
	mol_hf = scf.RHF(mol)
	mol_hf.kernel()

	#mo_coeff_hf = mol_hf.mo_coeff
	#if active_list != None:
	#	full_orb_list = [int(i-1) for i in active_list]
	#	for i in range(norb_tot):
	#		if i in core_list: full_orb_list.insert(i,i)
	#		if i not in full_orb_list and i not in core_list: full_orb_list.append(i)

	#	print (full_orb_list)
	#	mo_coeff_sort = mo_coeff_hf[:,full_orb_list]
	#else:
	#	print("didn't sort MOs!")
	#	exit()

	#mo_coeff_sort = np.loadtxt('LDA_MOcoeff.txt')
	#mo_coeff_sort = mol_hf.mo_coeff
	#np.savetxt('HF_MOcoeff_sort.txt',mo_coeff_sort)
	#np.savetxt('HF_MOcoeff.txt',mol_hf.mo_coeff)

	#mo_coeff_hf = np.loadtxt('LDA_MOcoeff.txt')
	mo_coeff_hf = np.loadtxt(sys.argv[2])
	print ('using MOs: ',sys.argv[2])
	mo_coeff = np.dot( mo_coeff_hf, U )

	# create molden file
	with open( filename+'.molden', 'w' ) as f:
		molden.header( mol, f )
		molden.orbital_coeff( mol, f, mo_coeff, ene=mol_hf.mo_energy, occ=mol_hf.mo_occ )

	if natorbs.any() != None:
		no_occ = np.zeros(mo_coeff.shape[1])
		no_occ[:norb_core] = 2
		no_occ[norb_core:norb_occ] = natorbs
		no_mocoeff = np.copy( mo_coeff )
		norb_act = norb_occ - norb_core
		no_mocoeff[:,norb_core:norb_occ] = np.dot( mo_coeff[:,norb_core:norb_occ], eigvecs_cas )
		#no_mc = mcscf.CASCI(mol_hf, norb_act, nelec_act, ncore=norb_core)
		#no_mc.mo_coeff[:,:] = no_mocoeff * 1.0
		#no_mc.fcisolver = fci.solver(mol_hf, singlet=True)
		#no_mc.fcisolver.nroots = 12
		#no_mc.kernel()
		#np.savetxt('C_gvp_no.txt',no_mc.ci)
		np.savetxt('MO_gvp_no.txt',no_mocoeff)
		with open( filename+'_natorbs.molden', 'w' ) as f:
			molden.header( mol, f )
			molden.orbital_coeff( mol, f, no_mocoeff, ene=mol_hf.mo_energy, occ=no_occ )

	return None

#def vis_orbs_molden( U, filename, molecule, basis, norb_core, norb_occ, norb_tot, eigvecs_cas=None, natorbs=None, active_list=None, core_list=None):
def vis_orbs_molden_U( U, filename, molecule, basis, norb_core, norb_occ, eigvecs_cas=None, natorbs=None, active_list=None, core_list=None):
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
	mol_hf = scf.RKS(mol)
	mol_hf.kernel()

	#mo_coeff_hf = mol_hf.mo_coeff
	#if active_list != None:
	#	full_orb_list = [int(i-1) for i in active_list]
	#	for i in range(norb_tot):
	#		if i in core_list: full_orb_list.insert(i,i)
	#		if i not in full_orb_list and i not in core_list: full_orb_list.append(i)

	#	print (full_orb_list)
	#	mo_coeff_sort = mo_coeff_hf[:,full_orb_list]
	#else:
	#	print("didn't sort MOs!")
	#	exit()

	#mo_coeff_sort = np.loadtxt('LDA_MOcoeff.txt')
	mo_coeff_sort = mol_hf.mo_coeff
	U_2step = np.loadtxt(sys.argv[2])
	mo_coeff_2step = np.dot( mo_coeff_sort, U_2step )
	mo_coeff = np.dot( mo_coeff_2step, U )

	# create molden file
	with open( filename+'.molden', 'w' ) as f:
		molden.header( mol, f )
		molden.orbital_coeff( mol, f, mo_coeff, ene=mol_hf.mo_energy, occ=mol_hf.mo_occ )

	if natorbs.any() != None:
		no_occ = np.zeros(mo_coeff.shape[1])
		no_occ[:norb_core] = 2
		no_occ[norb_core:norb_occ] = natorbs
		no_mocoeff = np.copy( mo_coeff )
		no_mocoeff[:,norb_core:norb_occ] = np.dot( mo_coeff[:,norb_core:norb_occ], eigvecs_cas )
		with open( filename+'_natorbs.molden', 'w' ) as f:
			molden.header( mol, f )
			molden.orbital_coeff( mol, f, no_mocoeff, ene=mol_hf.mo_energy, occ=no_occ )

	return None

# account for frozen core in remaining oints
# add frozen core components of tints to oints
def oints_2MFcore(oints, tints, norb_core, norb_occ):
	oints_MF = np.zeros((norb_occ,norb_occ))
	oints_MF[norb_core:norb_occ,norb_core:norb_occ] += oints[norb_core:norb_occ,norb_core:norb_occ] + (2.0 * np.trace( tints[norb_core:norb_occ,norb_core:norb_occ,:norb_core,:norb_core], axis1=2, axis2=3 )) - (1.0 * np.trace( tints[norb_core:norb_occ,:norb_core,:norb_core,norb_core:norb_occ], axis1=1, axis2=2 )) 
	#oints_MF = np.zeros((norb_occ,norb_occ))
	#for i in range(norb_core,norb_occ):
	#	for j in range(norb_core,norb_occ):
	#		oints_MF[i,j] = oints[i,j]
	#		for r in range(norb_core):
	#			oints_MF[i,j] += ( 2.0 * tints[i,j,r,r] ) - ( 1.0 * tints[i,r,r,j] ) 
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
def generate_strings( nelec, norb ):

	nelec_a = int(nelec/2.)
	combos = itertools.combinations(range(0,norb), nelec_a)
	strings = []
	for word in combos:
		split = [char for char in word]
		strings.append(split)
	colexicalSort(strings)

	return strings

''' Create indexing array of arc weights (CAS only)
paths: count of paths through each index (Norb x Nelec)
Wvert: vertex weights ((Norb+1) x (Nelec+1))
Warc: arc weights (Norb x Nelec)
'''
def prep_string2index(norb, nelec, strings):
	nelec_a = int(nelec/2.)
	paths = np.zeros((norb,nelec_a))
	for o in range(norb):	#loop through orb indices (starting at 0)
		for e in range(nelec_a):	#loop though electron indices (starting at 0)
			countStr = 0		#counter to skip index 0 of string list
			for I in strings:	#for each electron index, count the number of each occ orb
				if countStr > 0:
					if I[e] == (o+1):
						paths[o][e] += 1
				countStr += 1
	
	Wvert = np.zeros((norb+1,nelec_a+1))
	for e in range(norb-nelec_a+1):	#place 1's in 0th column
		Wvert[e][0] = 1
	for o in range(1,norb+1):	#loop through orb indices+1
		for e in range(1,nelec_a+1):	#loop through electron indices+1
			if paths[o-1][e-1] > 0:
				Wvert[o][e] = Wvert[o-1][e] + Wvert[o-1][e-1] 	#calculate vertex weight
	
	Warc = np.zeros((norb,nelec_a))
	for o in range(norb):
		for e in range(nelec_a):
			if paths[o][e] > 0:
				Warc[o][e] = Wvert[o][e+1] 

	return Warc

"""function that turns a list of occupied orbitals into a string index
input: list of occupied orbitals, in ascending order (element of strings array)
output: string index
"""
def string2index(orb_list, Warc):
	index = 1
	counter_elec = 0
	for i in orb_list:
		index += Warc[i-1][counter_elec]
		counter_elec += 1
	return index

'''Tracks ci index back to the associated alpha/beta strings
input: index in C_vec (nstr**2)
output: alpha string, beta string
'''
def ci2strings( nstr, nelec, norb, ind_vec ):
	ind_mat = np.unravel_index(ind_vec,(nstr,nstr))
	a_str = generate_strings( nelec, norb )[ind_mat[0]]
	b_str = generate_strings( nelec, norb )[ind_mat[1]]

	return a_str, b_str

