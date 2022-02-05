#!/usr/bin/env python3.8

import time
startTime = time.time()
print ('\n============================================')
print ('||                                        ||')
print ('||       ESS-CASSCF using the GVP         ||')
print ('||                                        ||')
print ('============================================')
print ('\nStarting execution at ', time.asctime(),'\n')

import os
import sys
import itertools
import numpy as np
from numpy import concatenate,reshape,transpose,matmul,trace,tensordot
from distutils.util import strtobool
from scipy.optimize import minimize
from scipy.special import comb
from scipy.linalg import expm,logm,eigh
from numpy.linalg import norm,eig
from functools import reduce
from pyscf import gto, scf, ao2mo 
from pyscf.fci.direct_spin0 import make_rdm12
from pyscf.fci.direct_spin1 import absorb_h1e,contract_2e

# custom tools 
from my_tools import *
#from approxHess import *
#from approxHess import prep_ham_diagonal_fast
from lbfgs import *

sys.stdout.flush()

# process input file into input dictionary
user_inputs = get_user_input(sys.argv[1])
print ('\nFinished imports, elapsed time in seconds = ', time.time() - startTime)
print("\nUser input dictionary: ")
for key in user_inputs:
	print("%30s : %30s" % (key, user_inputs[key]))
sys.stdout.flush()

# prepare output directory
if os.path.isfile(user_inputs['output_dir']): raise RuntimeError ("%s is a file and cannot be used as the output data directory" % user_inputs['outdir'] )
if not os.path.isdir(user_inputs['output_dir']): os.mkdir(user_inputs['output_dir'])
print ("\nData files will be placed in the following directory: %s" % user_inputs['output_dir'] )
sys.stdout.flush()

# import molecule
molecule, orb_list = build_molecule( user_inputs )

# generate integrals and core energy from RHF calc
nelec_tot, norb_tot, energy_nuc_core, energy_nuc = my_pyscf( molecule, user_inputs['basis'], user_inputs['nelec_active'], user_inputs['norb_active'], active_list=orb_list)
if user_inputs['core_type'] != 'closedCore': energy_nuc = energy_nuc_core

# set variables and parameters
nelec_act  = user_inputs['nelec_active']
norb_act   = user_inputs['norb_active']
nelec_core = int(nelec_tot - nelec_act)
norb_core  = int(nelec_core/2.)
norb_virt  = int(norb_tot - norb_act - norb_core)
norb_occ   = int(norb_act + norb_core)
nstr       = int(comb(norb_act,int(nelec_act/2.)))	# number of alpha (or beta) strings with the active space; norb choose nelec/2
lam        = user_inputs['lambda']	# magnitude of perturbation in finite difference
if user_inputs['core_type'] == 'closedCore':
	Xlen = int( (norb_act*norb_virt) + (norb_core*(norb_act+norb_virt)) )	# number of rotation coefficients in X (where U=e^X) included in the optimization
	norb_cut_start = 0			# initial slicing index for rotated integrals
	norb_cut_stop  = norb_occ		# final slicing index for rotated integrals
	norb_cut_dim   = norb_occ		# dimension of cut down integrals
else:
	Xlen = int(norb_act * norb_virt)			
	norb_cut_start = norb_core
	norb_cut_stop  = norb_occ
	norb_cut_dim   = norb_act

if norb_core == 0 and user_inputs['core_type'] != 'noCore': raise RuntimeError ("Core orbitals required for Frozen Core or Closed Core algorithms")

# make lists of orbital types
coreList = []
for n in range(norb_core):
	coreList.append(n)
actvList = []
for n in range(norb_core,norb_core+norb_act):
	actvList.append(n)
virtList = []
for n in range(norb_core+norb_act,norb_tot):
	virtList.append(n)

# Construct map of X matrix indices in Xflat vector
Xones = np.ones((Xlen))
if user_inputs['core_type'] == 'closedCore':
	Xcaav_ones = concatenate((reshape(Xones[:(norb_core*(norb_act+norb_virt))],[norb_core,norb_act+norb_virt]),concatenate((np.zeros((norb_act,norb_act)), reshape(Xones[(norb_core*(norb_act+norb_virt)):], [norb_act,norb_virt]) ), axis=1) ), axis=0)
else:
	Xcaav_ones = concatenate((np.zeros((norb_core,norb_act+norb_virt)),concatenate((np.zeros((norb_act,norb_act)),reshape(Xones,(norb_act,norb_virt))), axis=1)), axis=0)
Xupper_ones = concatenate((np.zeros((norb_core+norb_act+norb_virt,norb_core)),concatenate((Xcaav_ones,np.zeros((norb_virt,norb_act+norb_virt))), axis=0)), axis=1)
Xindices = []
for i in range(norb_tot):
	for j in range(norb_tot):
		if Xupper_ones[i,j] != 0: Xindices.append((i,j))

# Build dictionary of strings for the occupied space
if (user_inputs['Xrelax_bfgs_hess'] == True) or (user_inputs['bfgs_hess'] == True):
	print ("Building array of string information")
	occ_strings = []
	for i in range(nstr**2):
		Ia_occ, Ib_occ = ci2strings( nstr, nelec_act, norb_act, i )
		occ_strings.append([Ia_occ,Ib_occ])

print ("\nTotal Electrons and Orbitals:")
print ("nelec_tot  = %3d "  %nelec_tot   )
print ("norb_tot   = %3d "  %norb_tot  )
print ("\nActive Electrons and Orbitals:")
print ("nelec_act  = %3d "  %nelec_act   )
print ("norb_act   = %3d "  %norb_act  )
print ("norb_virt  = %3d "  %norb_virt   )
print ("\nCore Electrons and Orbitals:")
print ("nelec_core  = %3d "  %nelec_core   )
print ("norb_core   = %3d "  %norb_core  )
print ("core_type   = ", user_inputs['core_type'])
print ("\nParameter Counts:")
print ("# of ci strings = ",nstr)
print ("# of orb coeff  = ",Xlen)


# numpy settings
np.set_printoptions( linewidth=250, precision=6, suppress=True )
#np.set_printoptions( linewidth=250, suppress=False)

sys.stdout.flush()

# rotate electron integrals with a given rotation matrix
def prep_rotate_ints( varXflat, oints, tints ):

	# build full anti-hermitian matrix X of rotation coefficients from only the active-virtual block
	if user_inputs['core_type'] == 'closedCore':
		Xcaav = concatenate((reshape(varXflat[:(norb_core*(norb_act+norb_virt))],[norb_core,norb_act+norb_virt]),concatenate((np.zeros((norb_act,norb_act)), reshape(varXflat[(norb_core*(norb_act+norb_virt)):], [norb_act,norb_virt]) ), axis=1) ), axis=0)
	else:
		Xcaav = concatenate((np.zeros((norb_core,norb_act+norb_virt)),concatenate((np.zeros((norb_act,norb_act)),reshape(varXflat,(norb_act,norb_virt))), axis=1)), axis=0)

	Xupper = concatenate((np.zeros((norb_core+norb_act+norb_virt,norb_core)),concatenate((Xcaav,np.zeros((norb_virt,norb_act+norb_virt))), axis=0)), axis=1)
	X = Xupper - Xupper.T

	# build unitary rotation matrix U = e^X
	U = expm( X )

	# Rotate one electron integrals into updated MO basis and cut down to core and active orbitals
	MOtot_oints = matmul( transpose(U), matmul(oints, U) )
	
	# Rotate two electron integrals into updated MO basis
	## Full process with rank-4 tensor G: G_pqrs -> G_PQRS sum_{pqrs} U_pP.T U_qQ.T U_rR U_sS G_pqrs
	## G_pqrs -> G_pqr,s -> sum_s G_pqr,s U_sS -> G_pqr,S -> G_pqrS
	tmp_np = matmul( reshape( tints, [norb_tot*norb_tot*norb_tot,norb_tot]) , U )
	tmp_np = reshape( tmp_np, [norb_tot,norb_tot,norb_tot,norb_tot] )
	## G_pqrS -> G_pqSr -> G_pqS,r -> sum_r G_pqS,r U_rR -> G_pqS,R -> G_pqSR
	tmp_np = transpose( tmp_np, [0,1,3,2] )
	tmp_np = matmul( reshape( tmp_np, [norb_tot*norb_tot*norb_tot,norb_tot]) , U )
	tmp_np = reshape( tmp_np, [norb_tot,norb_tot,norb_tot,norb_tot] )
	## G_pqSR -> G_qpRS -> G_q,pRS -> sum_q U_qQ.T G_q,pRS -> G_Q,pRS -> G_QpRS
	tmp_np = transpose( tmp_np, [1,0,3,2] )
	tmp_np = matmul( transpose(U), reshape( tmp_np, [norb_tot, norb_tot*norb_tot*norb_tot]) )
	tmp_np = reshape( tmp_np, [norb_tot,norb_tot,norb_tot,norb_tot] )
	## G_QpRS -> G_pQRS -> G_p,QRS _> sum_p U_pP.T G_p,QRS -> G_PQRS
	tmp_np = transpose( tmp_np, [1,0,2,3] )
	tmp_np = matmul( transpose(U), reshape( tmp_np, [norb_tot, norb_tot*norb_tot*norb_tot]) )
	MOtot_tints = reshape( tmp_np, [norb_tot,norb_tot,norb_tot,norb_tot] )

	return MOtot_oints, MOtot_tints

# Orbital energy gradient
#   grad_orb = <psi| [Eij, H] |psi> 
def grad_E_orb( Xlen, coreList, actvList, virtList, rdm1, rdm2, oints, tints, norb_cut_start):

	grad_E_orb = np.zeros((norb_tot,norb_tot))

	# i=active, j=virtual
	grad_E_orb[norb_core:norb_occ,norb_occ:] += 2.0 * transpose( tensordot( oints[norb_occ:,norb_core:norb_occ], rdm1[norb_core-norb_cut_start:norb_occ-norb_cut_start,norb_core-norb_cut_start:norb_occ-norb_cut_start], axes=([1],[1]) ) ) \
	  + 2.0 * transpose( tensordot( tints[norb_occ:,norb_core:norb_occ,norb_core:norb_occ,norb_core:norb_occ], rdm2[norb_core-norb_cut_start:norb_occ-norb_cut_start,norb_core-norb_cut_start:norb_occ-norb_cut_start,norb_core-norb_cut_start:norb_occ-norb_cut_start,norb_core-norb_cut_start:norb_occ-norb_cut_start], axes=([1,2,3],[1,2,3]) ) ) \
	  + 4.0 * transpose( trace( tensordot( tints[norb_occ:,norb_core:norb_occ,:norb_core,:norb_core],rdm1[norb_core-norb_cut_start:norb_occ-norb_cut_start,norb_core-norb_cut_start:norb_occ-norb_cut_start],axes=([1],[1])), axis1=1, axis2=2 ) ) \
	  - 2.0 * transpose( trace( tensordot( tints[norb_occ:,:norb_core,norb_core:norb_occ,:norb_core],rdm1[norb_core-norb_cut_start:norb_occ-norb_cut_start,norb_core-norb_cut_start:norb_occ-norb_cut_start],axes=([2],[1])), axis1=1, axis2=2 ) )

	if user_inputs['core_type'] == 'closedCore':

		# i=core, j=active
		grad_E_orb[:norb_core,norb_core:norb_occ] += 4.0 * oints[:norb_core,norb_core:norb_occ] \
		  + 8.0 * trace( tints[:norb_core,norb_core:norb_occ,:norb_core,:norb_core], axis1=2, axis2=3 ) \
		  - 4.0 * trace( tints[:norb_core,:norb_core,norb_core:norb_occ,:norb_core], axis1=1, axis2=3 ) \
		  - 2.0 * tensordot( oints[:norb_core,norb_core:norb_occ], rdm1[norb_core:norb_occ,norb_core:norb_occ], axes=([1],[1]) ) \
		  + 4.0 * tensordot( tints[:norb_core,norb_core:norb_occ,norb_core:norb_occ,norb_core:norb_occ], rdm1[norb_core:norb_occ,norb_core:norb_occ], axes=([2,3],[0,1]) ) \
		  - 2.0 * tensordot( tints[:norb_core,norb_core:norb_occ,norb_core:norb_occ,norb_core:norb_occ], rdm1[norb_core:norb_occ,norb_core:norb_occ], axes=([1,3],[0,1]) ) \
		  - 2.0 * tensordot( tints[:norb_core,norb_core:norb_occ,norb_core:norb_occ,norb_core:norb_occ], rdm2[norb_core:norb_occ,norb_core:norb_occ,norb_core:norb_occ,norb_core:norb_occ], axes=([1,2,3],[1,2,3]) ) \
		  - 4.0 * trace( tensordot( tints[:norb_core,norb_core:norb_occ,:norb_core,:norb_core], rdm1[norb_core:norb_occ,norb_core:norb_occ], axes=([1],[1]) ), axis1=1, axis2=2 ) \
		  + 2.0 * trace( tensordot( tints[:norb_core,:norb_core,norb_core:norb_occ,:norb_core], rdm1[norb_core:norb_occ,norb_core:norb_occ], axes=([2],[1]) ), axis1=1, axis2=2 )

		# i=core, j=virtual
		grad_E_orb[:norb_core,norb_occ:] += 4.0 * oints[:norb_core,norb_occ:] \
		  + 8.0 * transpose( trace( tints[norb_occ:,:norb_core,:norb_core,:norb_core], axis1=2, axis2=3 ) )  \
		  - 4.0 * transpose( trace( tints[norb_occ:,:norb_core,:norb_core,:norb_core], axis1=1, axis2=3 ) )  \
		  + 4.0 * transpose( tensordot( tints[norb_occ:,:norb_core,norb_core:norb_occ,norb_core:norb_occ], rdm1[norb_core:norb_occ,norb_core:norb_occ], axes=([2,3],[0,1]) ) )  \
		  - 2.0 * transpose( tensordot( tints[norb_occ:,norb_core:norb_occ,:norb_core,norb_core:norb_occ], rdm1[norb_core:norb_occ,norb_core:norb_occ], axes=([1,3],[0,1]) ) ) 

		grad_E_orb_flat = concatenate( ( np.ndarray.flatten( grad_E_orb[:norb_core,norb_core:] ), np.ndarray.flatten( grad_E_orb[norb_core:norb_occ,norb_occ:] ) ) )

	else: 
		grad_E_orb_flat = np.ndarray.flatten( grad_E_orb[norb_core:norb_occ,norb_occ:] )

	return grad_E_orb_flat

# Orbital energy hessian, approximate as diagonal and with Fock operator
# 	Hess_orb = (1/2) * (1+Pijkl) * <psi| [Eij, [Ekl, F]] |psi> 
# ...approximate as diagonal, where ij is a compound index
# 	Hess_orb_{ij,ij} = (1/2) * (1 + P_{ij,ij}) * <psi| [Eij, [Eij, F]] |psi> 
# 			 = <psi| [Eij, [Eij, F]] |psi> 
def build_Hess_E_orb( Xlen, Xindices, coreList, actvList, virtList, rdm1, oints, tints, norb_cut_start):

	# Effective 1-eri
	#	g_pq = h_pq + sum_{r}^{Nspacial} ( 2*(pq|rr) - (pr|rq) )
	#	Nspacial (noCore,frozenCore) = active
	#	Nspacial (closedCore) = occupied
	eff_oints = oints \
	  + 2.0 * trace( tints[:,:,norb_cut_start:norb_occ,norb_cut_start:norb_occ], axis1=2, axis2=3 ) \
	  - 1.0 * trace( tints[:,norb_cut_start:norb_occ,norb_cut_start:norb_occ,:], axis1=1, axis2=2 )

	Hess_E_orb_mat = np.zeros((norb_tot,norb_tot))

	# i=active, j=virtual
	Hess_E_orb_mat[norb_core:norb_occ,norb_occ:] += 2.0 * transpose( tensordot( np.diagonal( eff_oints[norb_occ:,norb_occ:] )[None,:], np.diagonal( rdm1[norb_core-norb_cut_start:norb_occ-norb_cut_start,norb_core-norb_cut_start:norb_occ-norb_cut_start] )[:,None], axes=([0],[1]) ) ) \
	  - 2.0 * transpose( np.tile( np.diagonal( tensordot( eff_oints[norb_core:norb_occ,norb_core:norb_occ], rdm1[norb_core-norb_cut_start:norb_occ-norb_cut_start,norb_core-norb_cut_start:norb_occ-norb_cut_start], axes=([1],[1]) ) ), (norb_virt,1) ) )

	if user_inputs['core_type'] == 'closedCore':

		# i=core, j=active
		Hess_E_orb_mat[:norb_core,norb_core:norb_occ] += 2.0 * tensordot( np.diagonal( eff_oints[:norb_core,:norb_core] )[:,None], np.diagonal( rdm1[norb_core-norb_cut_start:norb_occ-norb_cut_start,norb_core-norb_cut_start:norb_occ-norb_cut_start] )[None,:], axes=([1],[0]) ) \
		  + 2.0 * transpose( tensordot( np.diagonal( eff_oints[norb_core:norb_occ,norb_core:norb_occ] )[None,:], np.diagonal( rdm1[:norb_core-norb_cut_start,:norb_core-norb_cut_start] )[:,None], axes=([0],[1]) ) ) \
		  - 2.0 * np.tile( reshape( np.diagonal( eff_oints[:norb_core-norb_cut_start,:norb_core-norb_cut_start] ) * np.diagonal( rdm1[:norb_core-norb_cut_start,:norb_core-norb_cut_start] ), ([norb_core,-1]) ), (1,norb_act) ) \
		  - 2.0 * np.tile( np.diagonal( tensordot( eff_oints[norb_core:norb_occ,norb_core:norb_occ], rdm1[norb_core-norb_cut_start:norb_occ-norb_cut_start,norb_core-norb_cut_start:norb_occ-norb_cut_start], axes=([1],[1]) ) ), (norb_core,1) )
	
		# i=core, j=virtual
		Hess_E_orb_mat[:norb_core,norb_occ:] += 2.0 * transpose( tensordot( np.diagonal( eff_oints[norb_occ:,norb_occ:] )[None,:], np.diagonal( rdm1[:norb_core-norb_cut_start,:norb_core-norb_cut_start] )[:,None], axes=([0],[1]) ) ) \
		  - 2.0 * np.tile( reshape( np.diagonal( eff_oints[:norb_core-norb_cut_start,:norb_core-norb_cut_start] ) * np.diagonal( rdm1[:norb_core-norb_cut_start,:norb_core-norb_cut_start] ), ([norb_core,-1]) ), (1,norb_virt) ) \

		Hess_E_orb_flat = concatenate( ( np.ndarray.flatten( Hess_E_orb_mat[:norb_core,norb_core:] ), np.ndarray.flatten( Hess_E_orb_mat[norb_core:norb_occ,norb_occ:] ) ) )

	else: 
		Hess_E_orb_flat = np.ndarray.flatten( Hess_E_orb_mat[norb_core:norb_occ,norb_occ:] )

	return Hess_E_orb_flat

# Diagonal of the Hamiltonian for a given product of strings
# TODO make faster by using list slicing
def prep_ham_diagonal( nstr, nelec_act, norb_act, oints, tints, Ia_occ, Ib_occ ):

	Hdiag = 0
	Ia_virt = [a for a in range(norb_act) if a not in Ia_occ]
	Ib_virt = [a for a in range(norb_act) if a not in Ib_occ]
	
	# contributions from alpha orbitals
	for ia in Ia_occ:
		Hdiag += oints[ia,ia] - 0.5 * trace( tints[ia,:,:,ia], axis1=0, axis2=1 )
		for ja in Ia_occ:
			Hdiag += 0.5 * tints[ia,ia,ja,ja]
		for aa in Ia_virt:
			Hdiag += 0.5 * tints[ia,aa,aa,ia]
		for jb in Ib_occ:
			Hdiag += 0.5 * tints[ia,ia,jb,jb]
	# contributions from beta orbitals
	for ib in Ib_occ:
		Hdiag += oints[ib,ib] - 0.5 * trace( tints[ib,:,:,ib], axis1=0, axis2=1 )
		for jb in Ib_occ:
			Hdiag += 0.5 * tints[ib,ib,jb,jb]
		for ab in Ib_virt:
			Hdiag += 0.5 * tints[ib,ab,ab,ib]
		for ja in Ia_occ:
			Hdiag += 0.5 * tints[ib,ib,ja,ja]

	return Hdiag

# calculate for a given X and C...
# 	Energy = C^+HC / C^+C = [Tr(oints * rdm1.T) + (1/2)*Tr(tints * rdm2)] / C^+C
# 	dE/dC  = 2*(HC - EC) / C^+C
# 	dE/dX  = <psi| [Eij, H] |psi>
def dE_X_C( varXflat, varC, give_d2E_info=False ):

	global Trot, Tci, Torb, Thess

	if user_inputs['core_type'] == 'closedCore':
		varC /= norm( varC )

	# build 1-electron and 2-electron rdm's using a pyscf function
	var_rdm1_act,var_rdm2_act = make_rdm12( varC, norb_act, (int(nelec_act/2.),int(nelec_act/2.)), reorder=True )

	if user_inputs['core_type'] == 'closedCore':
		# add occupied blocks to rdm's
		var_rdm1,var_rdm2 = rdm12_act2occ(var_rdm1_act, var_rdm2_act, norb_core, norb_occ)
	else:	# noCore, frozenCore
		var_rdm1 = var_rdm1_act
		var_rdm2 = var_rdm2_act

	# normalize: C.T C
	var_wfn_norm = np.sum(np.square( varC ),dtype=np.float64)

	orbtime = time.time()

	# rotate 1 and 2-electron integrals
	rottime = time.time()
	var_MOtot_oints, var_MOtot_tints = rotate_ints( varXflat )

	Trot += time.time() - rottime

	var_energy = (trace( matmul(var_MOtot_oints[norb_cut_start:norb_cut_stop,norb_cut_start:norb_cut_stop],transpose(var_rdm1)) ) + (1./2.) * tensordot( var_MOtot_tints[norb_cut_start:norb_cut_stop,norb_cut_start:norb_cut_stop,norb_cut_start:norb_cut_stop,norb_cut_start:norb_cut_stop],var_rdm2, [[0,1,2,3],[0,1,2,3]] )) / var_wfn_norm
	
	var_dE_dX = grad_E_orb( Xlen, coreList, actvList, virtList, var_rdm1, var_rdm2, var_MOtot_oints, var_MOtot_tints, norb_cut_start)

	#print ('dE_orb evaluation time in seconds = %12.6f' %(time.time() - orbtime))
	Torb += time.time() - orbtime
	citime = time.time()

	# prep integrals for dE/dC
	if user_inputs['core_type'] == 'closedCore':
		# account for mean-field effect of the core in the 1-electron integrals
		# cut down electron integrals to just the active space
		var_h2e = absorb_h1e( oints_2MFcore(var_MOtot_oints[norb_cut_start:norb_cut_stop,norb_cut_start:norb_cut_stop], var_MOtot_tints[norb_cut_start:norb_cut_stop,norb_cut_start:norb_cut_stop,norb_cut_start:norb_cut_stop,norb_cut_start:norb_cut_stop], norb_core, norb_occ)[norb_core:norb_occ,norb_core:norb_occ], var_MOtot_tints[norb_core:norb_occ,norb_core:norb_occ,norb_core:norb_occ,norb_core:norb_occ], norb_act, nelec_act, 0.5 )

	else:
		# cut down electron integrals to just the active space
		var_h2e = absorb_h1e( var_MOtot_oints[norb_cut_start:norb_cut_stop,norb_cut_start:norb_cut_stop], var_MOtot_tints[norb_cut_start:norb_cut_stop,norb_cut_start:norb_cut_stop,norb_cut_start:norb_cut_stop,norb_cut_start:norb_cut_stop], norb_act, nelec_act, 0.5 )

	# calculate H.C using pyscf functions
	hc = reshape( contract_2e( var_h2e, reshape(varC,(nstr,nstr)), norb_act, nelec_act ),(-1) )

	# active space energy
	E_act = np.dot(varC.T,hc) / var_wfn_norm

	# dE/dC
	var_dE_dC = ( 2. * ( hc - ( E_act * varC ) ) ) / var_wfn_norm
	#print (' dE_ci evaluation time in seconds = %12.6f' %(time.time() - citime))
	Tci += time.time() - citime

	# approx Hessian
	if give_d2E_info == True:

		hesstime = time.time()

		# Get diagonal element of Hamiltonian
		if user_inputs['core_type'] == 'closedCore':
			ham_diagonal = lambda Ia,Ib: prep_ham_diagonal( nstr, nelec_act, norb_act, oints_2MFcore(var_MOtot_oints[norb_cut_start:norb_cut_stop,norb_cut_start:norb_cut_stop], var_MOtot_tints[norb_cut_start:norb_cut_stop,norb_cut_start:norb_cut_stop,norb_cut_start:norb_cut_stop,norb_cut_start:norb_cut_stop], norb_core, norb_occ)[norb_core:norb_occ,norb_core:norb_occ], var_MOtot_tints[norb_core:norb_occ,norb_core:norb_occ,norb_core:norb_occ,norb_core:norb_occ], Ia, Ib )
		else:
			ham_diagonal = lambda Ia,Ib: prep_ham_diagonal( nstr, nelec_act, norb_act, var_MOtot_oints[norb_cut_start:norb_cut_stop,norb_cut_start:norb_cut_stop], var_MOtot_tints[norb_cut_start:norb_cut_stop,norb_cut_start:norb_cut_stop,norb_cut_start:norb_cut_stop,norb_cut_start:norb_cut_stop], Ia, Ib )

		# CI energy hessian, approximate using davidson CI
		#	Hess_ci_{ii} | _{k} = 2 * ( Hdiag_{ii} - E_{k} )
		# TODO can definitely make this more efficient...
		Hess_E_ci = np.zeros((nstr**2))
		for i in range(nstr**2):
			Ia_occ = occ_strings[i][0]
			Ib_occ = occ_strings[i][1]
			Hess_E_ci[i] = 2 * ( (ham_diagonal( Ia_occ, Ib_occ ) / np.sum(np.square( varC ))) - E_act )

		# Orbital energy hessian, approximate as diagonal and with Fock
		Hess_E_orb = build_Hess_E_orb( Xlen, Xindices, coreList, actvList, virtList, var_rdm1, var_MOtot_oints, var_MOtot_tints, norb_cut_start )

		Thess += time.time() - hesstime

		return var_energy, var_dE_dX, var_dE_dC, Hess_E_orb, Hess_E_ci
	else:
		return var_energy, var_dE_dX, var_dE_dC


# objective function to calculate L, dL, and d2L (optional)
# 	  L = chi[ mu*(omega - E)^2 + (1-mu)*|gradE|^2 ] + (1-chi)*gradE
def prep_obj_func( Vflat, mu, chi, giveVariables ):

	objfunc_start = time.time()

	global micro_counter

	Xflat,C = split_array(Vflat)

	# calculate energy gradients (and Hessian)
	if hessian == True:
		#Energy, dE_dX, dE_dC, hess_E_flat = dE_X_C( Xflat, C, True )
		Energy, dE_dX, dE_dC, Hess_E_orb, Hess_E_ci = dE_X_C( Xflat, C, True )
	else:
		Energy, dE_dX, dE_dC = dE_X_C( Xflat, C )

	# calculates and returns only specified variables
	if giveVariables == True:
		C_norm = C / norm( C )
		rdm1_norm, rdm2_norm = make_rdm12( C_norm, norb_act, (int(nelec_act/2.),int(nelec_act/2.)) )
		natorb_occ, eigvecs = eig( rdm1_norm )
		return Energy, C, Xflat, natorb_occ, eigvecs

	# reshape arrays for easier handling	
	dE_dX_flat = np.ndarray.flatten(dE_dX)
	dE_dC_flat = np.ndarray.flatten(dE_dC)
	gradE_flat = concatenate([dE_dX_flat,dE_dC_flat],0)

	# calculate norms
	gradE_norm = norm( gradE_flat )
	dE_dC_norm = norm( dE_dC )
	dE_dX_norm = norm( dE_dX )

	#objfunc_dE = time.time()
	#print ('energy derivatives evaluation time in seconds = %12.6f' %(time.time() - objfunc_start))

	# Derivative of |dE/dV|^2 wrt V using central finite difference
	# perturb V by maximum +/- (input lambda value)*dE/dV
	lamda = lam / max(gradE_norm,1.)
	Vp = np.copy( Vflat )
	Vm = np.copy( Vflat )
	Vp += lamda * gradE_flat
	Vm -= lamda * gradE_flat
	Xp,Cp = split_array( Vp )
	Xm,Cm = split_array( Vm )

	# calculate dE/d(V+/-lambda)
	Ep,dE_dXp,dE_dCp = dE_X_C( Xp, Cp )
	Em,dE_dXm,dE_dCm = dE_X_C( Xm, Cm )
	dEdXp_flat = np.ndarray.flatten( dE_dXp )
	dEdXm_flat = np.ndarray.flatten( dE_dXm )
	dEdCp_flat = np.ndarray.flatten( dE_dCp )
	dEdCm_flat = np.ndarray.flatten( dE_dCm )
	gradEp_flat = concatenate([dEdXp_flat,dEdCp_flat],0)
	gradEm_flat = concatenate([dEdXm_flat,dEdCm_flat],0)

	# calculate d|dE/dV|^2/dV with central finite difference
	dV_gradE_normsq = (1./lamda) * (gradEp_flat - gradEm_flat)
	dX_gradE_normsq,dC_gradE_normsq = split_array( dV_gradE_normsq )

	#objfunc_fd = time.time()
	#print ('finite difference evaluation time in seconds = %12.6f' %(time.time() - objfunc_dE))

	# Build Lagrangian and its derivatives
	# for relaxing X in zeroth macro-iteration...
	#	L     = (omega - E)^2
	#	dL/dX = -2*(omega - E)*dE/dX
	#	dL/dC = 0
	if user_inputs['Xrelax'] == True:

		# build Lagrangian
		Lag = np.square( omega - Energy )
		dLag_dC = np.zeros((nstr**2))
		dLag_dX = -2. * (omega - Energy) * dE_dX_flat
		
		if hessian == True:
			# Build approximate Lagrangian Hessian
			d2Lag_dC = np.ones((nstr**2))
			d2Lag_dX = -2. * ( ((omega - Energy) * Hess_E_orb) - dX_gradE_normsq )	#TODO remove mu, not necessary here

			# positive constant shift away from zero
			if min(abs(d2Lag_dX)) < user_inputs['hess_shift']:
				shift = user_inputs['hess_shift'] - min(abs(d2Lag_dX))
				d2Lag_dX += shift

			## signed constant shift away from zero
			#if min(abs(d2Lag_dX)) < user_inputs['hess_shift']:
			#	shift = user_inputs['hess_shift'] - min(abs(d2Lag_dX))
			#	for i in range(len(d2Lag_dX)):
			#		d2Lag_dX[i] += np.sign(d2Lag_dX[i]) * shift

			# Pack hess of ObjFunc into vector
			d2C = reshape(d2Lag_dC, [d2Lag_dC.size])
			d2X = reshape(d2Lag_dX, [d2Lag_dX.size])
			hess_lag_flat = concatenate([d2X,d2C], 0)

			# function that performs H^-1.vec for input vector
			hess_lag_inv = lambda vec: vec / hess_lag_flat

	# Build Lagrangian and its derivatives
	# for all coupled macro-iterations
	#	L     = chi*[ mu*(omega - E)^2 + (1-mu)*(|dE/dC|^2 + |dE/dX|^2) ] + (1-chi)*E
	#	dL/dX =  
	#	dL/dC =
	else:
		# build Lagrangian
		Lag = chi * ( (mu * (np.square( omega - Energy ))) + ((1.-mu) * (np.sum(np.square( dE_dC_flat )) + np.sum(np.square( dE_dX_flat )))) ) + ( (1. - chi) * Energy )
		
		# build Lagrangian derivative
		dLag_dX = (chi * ( (-2. * mu * ( omega - Energy ) * dE_dX_flat) + ((1.-mu) * (dX_gradE_normsq)) )) + ( (1.-chi) * dE_dX_flat )
		dLag_dC = (chi * ( (-2. * mu * ( omega - Energy ) * dE_dC_flat)      + ((1.-mu) * (dC_gradE_normsq)) )) + ( (1.-chi) * dE_dC )
	
		if hessian == True:
			# Build approximate Lagrangian Hessian
			d2Lag_dX = (chi * ( ( -2. * mu * ( ((omega - Energy) * Hess_E_orb) - dX_gradE_normsq ) ) + ( (1. - mu) * 2. * Hess_E_orb**2. ))) + ( (1. - chi) * Hess_E_orb )
			d2Lag_dC = (chi * ( ( -2. * mu * ( ((omega - Energy) * Hess_E_ci ) - dC_gradE_normsq ) ) + ( (1. - mu) * 2. * Hess_E_ci**2.  ))) + ( (1. - chi) * Hess_E_ci )	
	
			# Pack hess of ObjFunc into vector
			d2C = reshape(d2Lag_dC, [d2Lag_dC.size])
			d2X = reshape(d2Lag_dX, [d2Lag_dX.size])
			hess_lag_flat = concatenate([d2X,d2C], 0)

			# shift hessian values away from zero for numerical stability
			if min(abs(hess_lag_flat)) < user_inputs['hess_shift']:
				if mu > (user_inputs['hess_shift_signed_mu'] + 0.01):	# for large mu
					# positive constant shift away from zero
					#print('shifting hessian by a positive constant: ',user_inputs['hess_shift'])
					shift = user_inputs['hess_shift'] - min(hess_lag_flat)
					hess_lag_flat += shift
				else:	# for small mu	
					# signed constant shift away from zero
					#print('shifting hessian by a signed constant: +/-',user_inputs['hess_shift'])
					shift = user_inputs['hess_shift'] - min(abs(hess_lag_flat))
					for i in range(len(hess_lag_flat)):
						hess_lag_flat[i] += np.sign(hess_lag_flat[i]) * shift

			# function that performs H^-1.vec for input vector
			hess_lag_inv = lambda vec: vec / hess_lag_flat

	# Pack grad of ObjFunc into vector
	dC = reshape(dLag_dC, [dLag_dC.size])
	dX = reshape(dLag_dX, [dLag_dX.size])
	grad_flat = concatenate([dX,dC], 0)

	#print ('Lag Hess: \n',hess_lag_flat)
	#print ('\nHess condition number: ',np.divide(np.max(np.abs(hess_lag_flat)),np.min(np.abs(hess_lag_flat))))

	#print ('obj func evaluation time in seconds = %12.6f' %(time.time() - objfunc_start))

	# print iteration info and update counter
	micro_counter += 1
	if micro_counter % 2 == 0: print("chi,mu: %2d, %1.2f   func call = %4d    E = %10.12f    |dE_dC| = %10.12f    |dE_dX| = %10.12f    |dL| = %10.12f" %(chi, mu, micro_counter, Energy+energy_nuc, dE_dC_norm, dE_dX_norm, norm(grad_flat)))

	if hessian == True:
		return Lag, grad_flat, hess_lag_inv
	else:
		return Lag, grad_flat
	
# Set up computational graph
#print ("\nSetting up computational graph\n")
#sys.stdout.flush()

# import input guess X matrix and CI vector
Cguess = np.loadtxt(user_inputs['Cguess'])
Cguess_flat = reshape(Cguess,(nstr**2))
if user_inputs['Xguess'] == None: Xguess_flat = np.zeros((Xlen))
#if user_inputs['Xguess'] == None: Xguess_flat = np.ones((Xlen))
else: Xguess_flat = reshape(np.loadtxt(user_inputs['Xguess']),(Xlen))
#Cguess_flat += np.random.uniform(-5e-3,5e-3,(nstr**2))	# TAG: for debugging
#Xguess_flat += np.random.uniform(-5e-3,5e-3,(Xlen))	# TAG: for debugging
V0 = concatenate([Xguess_flat,Cguess_flat],0)	# initial input array of Xguess and Cguess

# get electron integrals
# frozenCore: account for MF effect of core in act and virt orbs
HF_oints, HF_tints = read_fcidump( user_inputs['eris'], coreList, active_list=orb_list )
if user_inputs['core_type'] == 'frozenCore': HF_oints = oints_2MFcore(HF_oints,HF_tints,norb_core,norb_tot)

# initialize objective function: L = chi[ mu*(omega - E)^2 + (1-mu)*|gradE|^2 ] + (1-chi)*gradE
thresh_initial = user_inputs['bfgs_thresh_initial']				# initial convergence threshold for BFGS optimization and Xrelax
thresh_final = user_inputs['bfgs_thresh_final']	# final convergence threshold for BFGS optimization
omega  = user_inputs['omega'] - energy_nuc	# targeted energy value
mus = np.arange(0.5,-0.1,-0.1)			# decreasing range of mu values 0.5 to 0	
#mus = np.insert(mus,0,1.)	# adds a macroiteration with L = (omega - E)^2
Emacros = []
chi = 1.0	#TODO
micro_counter = 0	# counts iterations within a BFGS call
macro_counter = 0	# counts number of BFGS calls
var_mu  = 0.5
var_chi = 1.0
var_giveVariables = False
objf = lambda Vflat: prep_obj_func( Vflat, var_mu, var_chi, var_giveVariables )
rotate_ints = lambda Xflat: prep_rotate_ints( Xflat, HF_oints, HF_tints )
split_array = lambda Vflat: prep_split_array( Vflat, Xlen, nstr )

print ("\nInitial Values:")
Trot = 0
Tci = 0
Torb = 0
Thess = 0
var_giveVariables = True
hessian = False
e,c,xflat,no_occ,ucas = objf(V0)
Emacros.append([macro_counter,e+energy_nuc])
print ("initial E = %4.12f " %(e+energy_nuc))
print ("initial natural orbital occupations = \n",no_occ.tolist())
#print ("initial C = \n",reshape(c,[nstr,nstr]))
print ("initial C = \n",c)
print ("initial X = \n",xflat)

#print ('\nCreating molden file for orbital visualization\n')
#U = expm( xtot )
#vis_orbs_molden( U, user_inputs['output_dir']+'/orbs_relaxed', molecule, user_inputs['basis'], norb_core, norb_occ, eigvecs_cas=ucas, natorbs=no_occ )

#exit()

# Optimization setup
print ('\n--------------------------------------')
print ('|                                    |')
print ('|       Starting optimization        |')
print ('|                                    |')
print ('--------------------------------------\n')

startTime2 = time.time()

# BFGS optimizer engine
def myEngine( V, mu, chi, bfgs_thresh, max_iter=10000, approxHess=False, step_control_version=None ):
	var_mu  = mu
	var_chi = chi
	var_giveVariables = False
	objf = lambda Vflat: prep_obj_func( Vflat, var_mu, var_chi, var_giveVariables )	#TODO is this necessary??
	if approxHess == False:		#use scipy's lbfgs optimizer
		optOptions = {'disp': None,
		              'maxcor':100,
		              'ftol': bfgs_thresh * np.finfo(float).eps,
		              'gtol': bfgs_thresh,
		              'eps':1e-08,
		              'maxfun':10000,
		              'maxiter':max_iter,
		              'iprint': user_inputs['bfgs_print'],
		              'maxls':50,
		             }
		optResult = minimize( objf, V, jac=True, method='L-BFGS-B', options=optOptions)
	else:	# use lbfgs.py
		optResult = lbfgs( objf, V, max_iter, max_hist=20, grad_thresh=bfgs_thresh, step_control_func=step_control_version )
	return optResult

# Relax Xguess to correspond to Cguess by freezing the CI vector during optimization
if user_inputs['Xrelax'] == True:
	mu_Xrelax  = 0.0
	chi_Xrelax = 1.0
	var_giveVariables = False
	hessian = user_inputs['Xrelax_bfgs_hess']
	print ("\nRelaxing X with CI coefficients frozen...")
	print ( "\n\nXrelax_ITER:%4d   Tolerance: %.4g  " %(macro_counter, thresh_initial))
	if hessian == False:
		print ("Using scipy's l-bfgs-b optimizer with no hessian information.")
		optResult = myEngine( V0, mu_Xrelax, chi_Xrelax, thresh_final, approxHess=False)
		tol = np.sum( np.square( optResult.jac ) )
		V_Xrelax = reshape(optResult.x,V0.shape)
	else:
		print ("Using local l-bfgs optimizer that uses the hessian information.")
		optResult,tol = myEngine( V0, mu_Xrelax, chi_Xrelax, thresh_initial, step_control_version=step_control_gridsearch, approxHess=True )
		#optResult,tol = myEngine( V0, mu_Xrelax, chi_Xrelax, thresh_initial, step_control_version=backtrack_linesearch, approxHess=True )
		tol = np.square( tol )
		V_Xrelax = reshape(optResult,V0.shape)
	print( "\nAfter Xrelax_iter:%4d   Error: %12.12f  \n" %(macro_counter, tol))
	var_giveVariables = True
	e,c,Xrelaxed_flat,no_occ,ucas = objf(V_Xrelax)
	Emacros.append([macro_counter,e+energy_nuc])
	print ("\nRelaxed X: \n",Xrelaxed_flat)
	np.savetxt(user_inputs['output_dir']+'/Xflat_relaxed.txt',Xrelaxed_flat)	# checkpoint file for restarting calc if necessary
	#Xrelaxed_flat = np.loadtxt('Xrelaxed.txt')
	Vcurrent = concatenate([Xrelaxed_flat,Cguess_flat],0)
	user_inputs['Xrelax'] = False
else:
	Vcurrent = V0

# Reset optimization inputs and parameters
var_giveVariables = False
hessian = user_inputs['bfgs_hess']
max_macro = user_inputs['macro_maxiters']
micro_counter = 0	# counts iterations within a BFGS call
macro_counter = 0	# counts number of BFGS calls
tol = 1000		# just to get macroiters going, not real tol value
thresh = thresh_initial

# Enter full optimization
# with chi = 1, mu = {.5,.4,.3,.2,.1,0}
chi = 1.0
while macro_counter < 6:	
	try:
		mu = mus[ macro_counter ]
	except:
		mu = 0.0
	macro_counter += 1
	print( "\n\nMACRO_ITER:%4d    Mu:%2.2f     Chi:%2.2f   Threshold: %.4g" %(macro_counter, mu, chi, thresh))
	var_giveVariables = False
	if hessian == False:
		print ("Using scipy's l-bfgs-b optimizer with no hessian information.")
		optResult = myEngine( Vcurrent, mu, chi, thresh )
		Vcurrent = reshape(optResult.x,V0.shape)
		tol = np.sum( np.square( optResult.jac ) )
	else:
		print ("Using local l-bfgs optimizer that uses the hessian information.")
		#if macro_counter < 2:
		#	thresh_tmp = 1e-4
		#else:
		#	thresh_tmp = thresh
		optResult,tol = myEngine( Vcurrent, mu, chi, thresh, step_control_version=backtrack_linesearch, approxHess=True )
		tol = np.square( tol )
		Vcurrent = reshape(optResult,V0.shape)
	print( "\nAfter macro_iter:%4d    Mu:%2.2f     Chi:%2.2f   Error: %12.12f " %(macro_counter, mu, chi, tol))
	micro_counter = 0
	var_giveVariables = True
	e_temp, c_temp, Xav_temp, no_occ_temp, ucas_temp = objf(Vcurrent)
	print ("natural orbital occupations = \n",no_occ_temp)
	Emacros.append([macro_counter,e_temp+energy_nuc])
	np.savetxt(user_inputs['output_dir']+'/C_macroiter_'+str(macro_counter)+'.txt',c_temp)				# formatted as checkpoint file for restarting calc if necessary
	np.savetxt(user_inputs['output_dir']+'/Xflat_macroiter_'+str(macro_counter)+'.txt',Xav_temp)	# checkpoint file for restarting calc if necessary
	if np.around(thresh,8) > thresh_final: thresh *= 1e-1
	else: thresh = thresh_final
	if tol <= 1e-11: 
		print ("\nterminated mu>0 due to tolerance: ",tol)
		break
	

# with chi = 1, mu = 0
thresh = thresh_final
while macro_counter < max_macro:
	#if tol <= thresh_final: 
	mu = 0.0
	macro_counter += 1
	print( "\n\nMACRO_ITER:%4d    Mu:%2.2f     Chi:%2.2f   Threshold: %.4g  " %(macro_counter, mu, chi, thresh))
	var_giveVariables = False
	if hessian == False:
		print ("Using scipy's l-bfgs-b optimizer with no hessian information.")
		optResult = myEngine( Vcurrent, mu, chi, thresh )
		Vcurrent = reshape(optResult.x,V0.shape)
		tol = np.sum( np.square( optResult.jac ) )
	else:
		print ("Using local l-bfgs optimizer that uses the hessian information.")
		optResult,tol = myEngine( Vcurrent, mu, chi, thresh, step_control_version=backtrack_linesearch, approxHess=True )
		tol = np.square( tol )
		Vcurrent = reshape(optResult,V0.shape)
	print( "\nAfter macro_iter:%4d    Mu:%2.2f     Chi:%2.2f   Error: %12.12f " %(macro_counter, mu, chi, tol))
	micro_counter = 0
	var_giveVariables = True
	e_temp, c_temp, Xav_temp, no_occ_temp, ucas_temp = objf(Vcurrent)
	print ("natural orbital occupations = \n",no_occ_temp)
	Emacros.append([macro_counter,e_temp+energy_nuc])
	np.savetxt(user_inputs['output_dir']+'/C_macroiter_'+str(macro_counter)+'.txt',c_temp)				# formatted as checkpoint file for restarting calc if necessary
	np.savetxt(user_inputs['output_dir']+'/Xflat_macroiter_'+str(macro_counter)+'.txt',Xav_temp)	# checkpoint file for restarting calc if necessary
	if tol <= 1e-11: 
		print ("\nterminated mu=0 due to tolerance: ",tol)
		break

# chi = 0, mu = 0
if user_inputs['chi0'] == True:
	mu = 0.0
	chi = 0.0
	while macro_counter < (max_macro+5):	
		macro_counter += 1
		print( "\n\nMACRO_ITER:%4d    Mu:%2.2f     Chi:%2.2f   Threshold: %.4g  " %(macro_counter, mu, chi, thresh))
		var_giveVariables = False
		if hessian == False:
			print ("Using scipy's l-bfgs-b optimizer with no hessian information.")
			optResult = myEngine( Vcurrent, mu, chi, thresh )
			Vcurrent = reshape(optResult.x,V0.shape)
			tol = np.sum( np.square( optResult.jac ) )
		else:
			print ("Using local l-bfgs optimizer that uses the hessian information.")
			optResult,tol = myEngine( Vcurrent, mu, chi, thresh, step_control_noCore, approxHess=True )
			tol = np.square( tol )
			Vcurrent = reshape(optResult,V0.shape)
		print( "\nAfter macro_iter:%4d    Mu:%2.2f     Chi:%2.2f   Error: %12.12f " %(macro_counter, mu, chi, tol))
		micro_counter = 0
		var_giveVariables = True
		e_temp, c_temp, Xav_temp, no_occ_temp, ucas_temp = objf(Vcurrent)
		print ("natural orbital occupations = \n",no_occ_temp)
		Emacros.append([macro_counter,e_temp+energy_nuc])
		np.savetxt(user_inputs['output_dir']+'/C_macroiter_'+str(macro_counter)+'.txt',c_temp)				# formatted as checkpoint file for restarting calc if necessary
		np.savetxt(user_inputs['output_dir']+'/Xflat_macroiter_'+str(macro_counter)+'.txt',Xav_temp)	# checkpoint file for restarting calc if necessary
		#if tol <= thresh_final: 
		if tol <= 1e-12: 
			print ("\nterminated chi=0 due to tolerance: ",tol)
			break

# print and save final energy, CI vector, X matrix (act-virt block)
print ("\n\nfinal values:")
var_giveVariables = True
e,c,xflat,no_occ,ucas = objf(Vcurrent)
print ("final E = %4.12f"%(e+energy_nuc))
print ("final natural orbital occupations = \n",no_occ)
print ("final C = \n",reshape(c,[nstr,nstr]))
print ("final Xflat = \n",xflat)

# Create file to analyze ci vector
print ('\nCi vector MO orbital configurations:')
for i in range(nstr**2):
	if c[i] > 5e-2:
		print ('occupied orbitals (alpha,beta): ',ci2strings(nstr,nelec_act,norb_act,i),' coeff = ', c[i])

# write optimized data to files with output directory
np.savetxt(user_inputs['output_dir']+'/C.txt',c)				# formatted as checkpoint file for restarting calc if necessary
#np.savetxt(user_inputs['output_dir']+'/Xtot.txt',xtot)
np.savetxt(user_inputs['output_dir']+'/Xflat.txt',xflat)	# checkpoint file for restarting calc if necessary
#np.savetxt(user_inputs['output_dir']+'/U.txt',expm(xtot))
np.savetxt(user_inputs['output_dir']+'/Cguess.txt',Cguess)
np.savetxt(user_inputs['output_dir']+'/Eplot.txt',Emacros)
#os.rename('count.inp',user_inputs['output_dir']+'/count.inp')
#os.rename('fcidump.txt',user_inputs['output_dir']+'/fcidump.txt')

## Create molden files for orbital visualization
#if user_inputs['molden'] == True:
#	print ('\nCreating molden file for orbital visualization\n')
#	U = expm( xtot )
#	vis_orbs_molden( U, user_inputs['output_dir']+'/orbs_relaxed', molecule, user_inputs['basis'], norb_core, norb_occ, eigvecs_cas=ucas, natorbs=no_occ )

print ('\nJob finished, elapsed time in seconds = %12.6f' %(time.time() - startTime))
print (  '                           in minutes = %12.6f' %((time.time() - startTime)/60.))
print (  '                           in hours   = %12.6f' %((time.time() - startTime)/3600.))
print ('\nTotal integral rotation time in seconds = %12.6f' %(Trot))
print (  '            Total dE/dC time in seconds = %12.6f' %(Tci))
print (  '            Total dE/dX time in seconds = %12.6f' %(Torb))
print (  '          Total hessian time in seconds = %12.6f' %(Thess))

