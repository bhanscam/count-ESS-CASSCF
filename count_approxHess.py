#!/usr/bin/env python3.8

# Last edit: Becky Hanscam, 7/19/20
# python3.8 gvp_cas_frozencore.py input_file > output_file.log

# get starting time
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
#import matplotlib.pyplot as plt
from distutils.util import strtobool
from scipy.optimize import minimize
from scipy.special import comb
from scipy.linalg import expm,logm,eigh
from functools import reduce
from pyscf import gto, scf, ao2mo 
from pyscf.fci.direct_spin0 import make_rdm12
from pyscf.fci.direct_spin1 import absorb_h1e,contract_2e

# TensorFlow as v.1
import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore", category=FutureWarning)
	warnings.simplefilter(action='ignore', category=DeprecationWarning)
	import tensorflow.compat.v1 as tf
	#import tensorflow as tf
	tf.disable_v2_behavior()

# custom tools 
from my_tools import *
from approxHess import *
from lbfgs import *

sys.stdout.flush()

# process input file into input dictionary
user_inputs = get_user_input(sys.argv[1])
print ('\nFinished imports, elapsed time in seconds = ', time.time() - startTime)

# print out user inputs
print("\nUser input dictionary: ")
for key in user_inputs:
	print("%30s : %30s" % (key, user_inputs[key]))
print()
sys.stdout.flush()

# prepare output directory
if os.path.isfile(user_inputs['output_dir']):
	raise RuntimeError ("%s is a file and cannot be used as the output data directory" % user_inputs['outdir'] )
if not os.path.isdir(user_inputs['output_dir']):
	os.mkdir(user_inputs['output_dir'])
print ("Data files will be placed in the following directory: %s" % user_inputs['output_dir'] )
print ()
sys.stdout.flush()

# import molecule
molecule = []
with open( user_inputs['geometry'],'r') as f:
	for line in f:
		molecule.append( line )
molecule = molecule[1:]
natoms = len(molecule)
nelec_act = user_inputs['nelec_active']
norb_act  = user_inputs['norb_active']
if user_inputs['active_list'] == None:
	orb_list = None
else:
	orb_list = [int(s) for s in user_inputs['active_list'].split(',')]

print ('orb_list = ',orb_list)	#TAG1

# print molecule geometry
tmp = "MOLECULE"
print ( "%s" %tmp + " "*(abs(8-len(tmp)))  + " %6s" %("X") + " %16s" %("Y") + " %16s" %("Z") )
for i in range(0,natoms):
	atom = molecule[i]
	atom = atom.split(',')
	atom[1] = float(atom[1])
	atom[2] = float(atom[2])
	atom[3] = float(atom[3])
	molecule[i] = atom
	print ( "%s" %atom[0] + " "*(abs(8-len(atom[0]))) + " %16s" %("%.10f	%.10f	%.10f" %(atom[1],atom[2],atom[3]) ))
print ('\n\n')

# generate integrals and core energy from RHF calc
nelec_tot, norb_tot, energy_nuc_core, energy_nuc = my_pyscf( molecule, user_inputs['basis'], nelec_act, norb_act, active_list=orb_list)
if user_inputs['core_type'] != 'closedCore': energy_nuc = energy_nuc_core

# set variables and parameters
nelec_core   = int(nelec_tot - nelec_act)			# number of core electrons
norb_core    = int(nelec_core/2.)				# number of core orbitals
norb_virt    = int(norb_tot - norb_act - norb_core)		# number of virtual orbitals
norb_occ     = int(norb_act + norb_core)			# number of occupied orbitals
nstr         = int(comb(norb_act,int(nelec_act/2.)))		# number of alpha (or beta) strings with the active space; norb choose nelec/2
lam          = user_inputs['lambda']				# magnitude of perturbation in finite difference
if user_inputs['core_type'] == 'closedCore':
	Xlen = int( (norb_act*norb_virt) + (norb_core*(norb_act+norb_virt)) )	# number of rotation coefficients in X (where U=e^X) included in the optimization
	norb_cut_start = 0			# initial slicing index for rotated integrals
	norb_cut_stop  = norb_occ		# final slicing index for rotated integrals
	norb_cut_dim   = norb_occ		# dimension of cut down integrals
else:	# noCore, frozenCore
	Xlen = int(norb_act * norb_virt)			
	norb_cut_start = norb_core
	norb_cut_stop  = norb_occ
	norb_cut_dim   = norb_act

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

# import input guess X matrix and CI vector
Cguess = np.loadtxt(user_inputs['Cguess'])
Cguess_flat = np.reshape(Cguess,(nstr**2))
if user_inputs['Xguess'] == None: Xguess_flat = np.zeros((Xlen))
else: Xguess_flat = np.reshape(np.loadtxt(user_inputs['Xguess']),(Xlen))
#Cguess_flat += np.random.uniform(-5e-3,5e-3,(nstr**2))	# TAG: for debugging
#Xguess_flat += np.random.uniform(-5e-3,5e-3,(Xlen))	# TAG: for debugging
V0 = np.concatenate([Xguess_flat,Cguess_flat],0)	# initial input array of Xguess and Cguess

# get electron integrals
# frozenCore: account for MF effect of core in act and virt orbs
HF_oints, HF_tints = read_fcidump( user_inputs['eris'], coreList, active_list=orb_list )
HF_tints_tf = np.copy( np.array(HF_tints) )
if user_inputs['core_type'] == 'frozenCore': HF_oints_tf = oints_2MFcore(HF_oints,HF_tints,norb_core,norb_tot)
else: HF_oints_tf = np.copy( np.array(HF_oints) )
del HF_oints, HF_tints

print ()
print ("Total Electrons and Orbitals:")
print ("------------------------------")
print ("nelec_tot  = %3d "  %nelec_tot   )
print ("norb_tot = %3d "  %norb_tot  )
print ()
    
print ("Active Electrons and Orbitals:")
print ("------------------------------")
print ("nelec_act  = %3d "  %nelec_act   )
print ("norb_act = %3d "  %norb_act  )
print ("norb_virt  = %3d "  %norb_virt   )
print ()

print ("Core Electrons and Orbitals:")
print ("------------------------------")
print ("nelec_core  = %3d "  %nelec_core   )
print ("norb_core = %3d "  %norb_core  )
print ()

if norb_core == 0 and user_inputs['core_type'] != 'noCore': raise RuntimeError ("Core orbitals required for Frozen Core or Closed Core algorithms")
print ("core_type = ", user_inputs['core_type'])
print ("number of ci strings = ",nstr)
print ("number of orb params = ",Xlen)
print ()

if (user_inputs['Xrelax_bfgs_hess'] == True) or (user_inputs['bfgs_hess'] == True):
	print ("Building array of string information")
	occ_strings = []
	for i in range(nstr**2):
		Ia_occ, Ib_occ = ci2strings( nstr, nelec_act, norb_act, i )
		occ_strings.append([Ia_occ,Ib_occ])

# numpy settings
np.set_printoptions( linewidth=250, precision=6, suppress=True )
#np.set_printoptions( linewidth=250, suppress=False)

sys.stdout.flush()

# splits flat array into X (vector) and C (matrix) variables
def split_array(flat):
	global Xlen
	flat = np.reshape(flat,[-1])
	X  = flat[:Xlen]
	C  = np.reshape(flat[Xlen:], [nstr**2])
	return X, C

# TensorFlow graph to calculate dE/dX
def prep_tf_dX( oints_inp, tints_inp ):
	f64 = tf.float64
	Xcaav_tf = tf.placeholder(f64, [norb_core+norb_act,norb_act+norb_virt])
	C_tf     = tf.placeholder(f64, [nstr**2])
	rdm1_tf  = tf.placeholder(f64, [norb_cut_dim,norb_cut_dim])
	rdm2_tf  = tf.placeholder(f64, [norb_cut_dim,norb_cut_dim,norb_cut_dim,norb_cut_dim])
	oints_tf = tf.constant(oints_inp)
	tints_tf = tf.constant(tints_inp)

	## Build the X rotation matrix. X should be anti-Hermitian
	Xupper = tf.concat((tf.zeros([norb_core+norb_act+norb_virt,norb_core],dtype=f64),tf.concat((Xcaav_tf,tf.zeros([norb_virt,norb_act+norb_virt],dtype=f64)), axis=0)), axis=1)
	Xmat_tf = Xupper - tf.transpose(Xupper)

	# Build unitary matrix U = exp(X) using taylor expansion to 13th order
	U_tf = tf.linalg.expm( Xmat_tf )

	# Rotate one electron integrals into updated MO basis and cut down to core and active orbitals
	MOtot_oints_tf = tf.matmul( tf.transpose(U_tf), tf.matmul(oints_tf, U_tf) )
	
	# Rotate two electron integrals into updated MO basis
	## Full process with rank-4 tensor G: G_pqrs -> G_PQRS sum_{pqrs} U_pP.T U_qQ.T U_rR U_sS G_pqrs
	## G_pqrs -> G_pqr,s -> sum_s G_pqr,s U_sS -> G_pqr,S -> G_pqrS
	tmp_tf = tf.matmul( tf.reshape( tints_tf, [norb_tot*norb_tot*norb_tot,norb_tot]) , U_tf )
	tmp_tf = tf.reshape( tmp_tf, [norb_tot,norb_tot,norb_tot,norb_tot] )
	## G_pqrS -> G_pqSr -> G_pqS,r -> sum_r G_pqS,r U_rR -> G_pqS,R -> G_pqSR
	tmp_tf = tf.transpose( tmp_tf, [0,1,3,2] )
	tmp_tf = tf.matmul( tf.reshape( tmp_tf, [norb_tot*norb_tot*norb_tot,norb_tot]) , U_tf )
	tmp_tf = tf.reshape( tmp_tf, [norb_tot,norb_tot,norb_tot,norb_tot] )
	## G_pqSR -> G_qpRS -> G_q,pRS -> sum_q U_qQ.T G_q,pRS -> G_Q,pRS -> G_QpRS
	tmp_tf = tf.transpose( tmp_tf, [1,0,3,2] )
	tmp_tf = tf.matmul( tf.transpose(U_tf), tf.reshape( tmp_tf, [norb_tot, norb_tot*norb_tot*norb_tot]) )
	tmp_tf = tf.reshape( tmp_tf, [norb_tot,norb_tot,norb_tot,norb_tot] )
	## G_QpRS -> G_pQRS -> G_p,QRS _> sum_p U_pP.T G_p,QRS -> G_PQRS
	tmp_tf = tf.transpose( tmp_tf, [1,0,2,3] )
	tmp_tf = tf.matmul( tf.transpose(U_tf), tf.reshape( tmp_tf, [norb_tot, norb_tot*norb_tot*norb_tot]) )
	MOtot_tints_tf = tf.reshape( tmp_tf, [norb_tot,norb_tot,norb_tot,norb_tot] )

	# Cut down integrals to occupied or active space
	MOcut_oints_tf = MOtot_oints_tf[norb_cut_start:norb_cut_stop,norb_cut_start:norb_cut_stop]
	MOcut_tints_tf = MOtot_tints_tf[norb_cut_start:norb_cut_stop,norb_cut_start:norb_cut_stop,norb_cut_start:norb_cut_stop,norb_cut_start:norb_cut_stop]

	# Electronic energy
	energy_tf = (tf.trace( tf.matmul(MOcut_oints_tf,tf.transpose(rdm1_tf)) ) + (1./2.) * tf.tensordot( MOcut_tints_tf,rdm2_tf, [[0,1,2,3],[0,1,2,3]] )) / tf.reduce_sum(tf.square(C_tf))

	# Energy derivative wrt to the active-virtual block of X
	dEdX = tf.gradients(energy_tf,Xcaav_tf)[0]

	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)

	# Feed inputs through tf graph
	def myf(Xflat,C_np,rdm1_np,rdm2_np):
		if user_inputs['core_type'] == 'closedCore':
			Xcaav_np = np.concatenate((np.reshape(Xflat[:(norb_core*(norb_act+norb_virt))],[norb_core,norb_act+norb_virt]),np.concatenate((np.zeros((norb_act,norb_act)), np.reshape(Xflat[(norb_core*(norb_act+norb_virt)):], [norb_act,norb_virt]) ), axis=1) ), axis=0)
		else:
			Xcaav_np = np.concatenate((np.zeros((norb_core,norb_act+norb_virt)),np.concatenate((np.zeros((norb_act,norb_act)),np.reshape(Xflat,(norb_act,norb_virt))), axis=1)), axis=0)

		# map between inputs and tf variables
		feed_dict = {Xcaav_tf:Xcaav_np, C_tf:C_np, rdm1_tf:rdm1_np, rdm2_tf:rdm2_np}

		# retrieve values from tf graph
		e     = sess.run(energy_tf,   feed_dict)
		oints_tot = sess.run(MOtot_oints_tf, feed_dict)
		oints_cut = sess.run(MOcut_oints_tf, feed_dict)
		tints_tot = sess.run(MOtot_tints_tf, feed_dict)
		tints_cut = sess.run(MOcut_tints_tf, feed_dict)
		dedx_caav  = sess.run(dEdX,        feed_dict)
		if user_inputs['core_type'] == 'closedCore':
			dedx = np.concatenate((np.ndarray.flatten(dedx_caav[:norb_core,:]), np.ndarray.flatten(dedx_caav[norb_core:,norb_act:])), axis=0)
		else:
			dedx = np.ndarray.flatten(dedx_caav[norb_core:,norb_act:])

		return e, dedx, oints_tot, tints_tot, oints_cut, tints_cut 

	return myf

# calculate for a given X and C...
# 	Energy (tf) = C^+HC / C^+C = [Tr(oints * rdm1.T) + (1/2)*Tr(tints * rdm2)] / C^+C
# 	dE/dC       = 2*(HC - EC) / C^+C
# 	dE/dX       = dE/dU * dU/dX (calculated by tensorflow graph)
def dE_X_C( varXflat, varC, give_d2E_info=False ):

	if user_inputs['core_type'] == 'closedCore':
		varC /= np.linalg.norm( varC )

	# build 1-electron and 2-electron rdm's using a pyscf function
	var_rdm1_act,var_rdm2_act = make_rdm12( varC, norb_act, (int(nelec_act/2.),int(nelec_act/2.)) )

	if user_inputs['core_type'] == 'closedCore':
		# add occupied blocks to rdm's
		var_rdm1,var_rdm2 = rdm12_act2occ(var_rdm1_act, var_rdm2_act, norb_core, norb_occ)
	else:	# noCore, frozenCore
		var_rdm1 = var_rdm1_act
		var_rdm2 = var_rdm2_act

	# normalize: C.T C
	var_wfn_norm = np.sum(np.square( varC ),dtype=np.float64)

	# tf function: E, dE/dX, rotated 1 and 2-electron integrals cut down to occupied space
	var_energy,var_dE_dX,var_MOtot_oints,var_MOtot_tints,var_MOcut_oints,var_MOcut_tints = tf_dX(varXflat,varC,var_rdm1,var_rdm2)

	# prep integrals for dE/dC
	if user_inputs['core_type'] == 'closedCore':
		# account for mean-field effect of the core in the 1-electron integrals
		var_MFcore_oints = oints_2MFcore(var_MOcut_oints, var_MOcut_tints, norb_core, norb_occ)

		# cut down 2-electron integrals to just the active space
		var_MO_oints_act  = var_MFcore_oints[norb_core:norb_occ,norb_core:norb_occ]
		var_MO_tints_act  = var_MOcut_tints[norb_core:norb_occ,norb_core:norb_occ,norb_core:norb_occ,norb_core:norb_occ]

	else:	# noCore, frozenCore
		var_MO_oints_act = var_MOcut_oints
		var_MO_tints_act = var_MOcut_tints

	# calculate HC using pyscf functions
	var_h2e = absorb_h1e( var_MO_oints_act, var_MO_tints_act, norb_act, nelec_act, 0.5 )
	hc = contract_2e( var_h2e, np.reshape(varC,(nstr,nstr)), norb_act, nelec_act )
	hc = np.reshape(hc,(-1))

	# E (active space): debugging
	E_act = np.dot(varC.T,hc) / var_wfn_norm

	# dE/dC
	var_dE_dC = ( 2. * ( hc - ( (np.dot(varC.T,hc) / var_wfn_norm) * varC ) ) ) / var_wfn_norm

	if give_d2E_info == True:
		return var_energy, var_dE_dX, var_dE_dC, var_h2e, var_rdm1, var_MOtot_oints, var_MOtot_tints, var_MO_oints_act, var_MO_tints_act, E_act
	else:
		return var_dE_dX, var_dE_dC


# objective function to calculate L, dL, and d2L (optional)
# 	  L = chi[ mu*(omega - E)^2 + (1-mu)*|gradE|^2 ] + (1-chi)*gradE
#	 dL = 
#	d2L =
def prep_obj_func( Vflat, mu, chi, giveVariables ):

	global micro_counter

	Xflat,C = split_array(Vflat)

	Xones = np.ones((Xflat.size))	# for identifying the indices of Xflat in matrix X

	# build full anti-hermitian matrix X of rotation coefficients from only the active-virtual block
	if user_inputs['core_type'] == 'closedCore':
		Xcaav = np.concatenate((np.reshape(Xflat[:(norb_core*(norb_act+norb_virt))],[norb_core,norb_act+norb_virt]),np.concatenate((np.zeros((norb_act,norb_act)), np.reshape(Xflat[(norb_core*(norb_act+norb_virt)):], [norb_act,norb_virt]) ), axis=1) ), axis=0)
		Xcaav_ones = np.concatenate((np.reshape(Xones[:(norb_core*(norb_act+norb_virt))],[norb_core,norb_act+norb_virt]),np.concatenate((np.zeros((norb_act,norb_act)), np.reshape(Xones[(norb_core*(norb_act+norb_virt)):], [norb_act,norb_virt]) ), axis=1) ), axis=0)
	else:
		Xcaav = np.concatenate((np.zeros((norb_core,norb_act+norb_virt)),np.concatenate((np.zeros((norb_act,norb_act)),np.reshape(Xflat,(norb_act,norb_virt))), axis=1)), axis=0)
		Xcaav_ones = np.concatenate((np.zeros((norb_core,norb_act+norb_virt)),np.concatenate((np.zeros((norb_act,norb_act)),np.reshape(Xones,(norb_act,norb_virt))), axis=1)), axis=0)

	Xupper = np.concatenate((np.zeros((norb_core+norb_act+norb_virt,norb_core)),np.concatenate((Xcaav,np.zeros((norb_virt,norb_act+norb_virt))), axis=0)), axis=1)
	Xupper_ones = np.concatenate((np.zeros((norb_core+norb_act+norb_virt,norb_core)),np.concatenate((Xcaav_ones,np.zeros((norb_virt,norb_act+norb_virt))), axis=0)), axis=1)
	X = Xupper - Xupper.T

	# build unitary rotation matrix U = e^X
	U = expm( X )

	# store indices of Xflat parameters in X matrix
	Xindices = []
	for i in range(norb_tot):
		for j in range(norb_tot):
			if Xupper_ones[i,j] != 0: Xindices.append((i,j))

	Energy, dE_dX, dE_dC, h2e, rdm1, MO_oints_tot, MO_tints_tot, MO_oints_act, MO_tints_act, Eact = dE_X_C( Xflat, C, True )	# TAG1
	
	# calculates and returns only specified variables
	if giveVariables == True:
		natorb_occ, eigvecs = eigh( rdm1 )
		return Energy, C, Xflat, X, natorb_occ, eigvecs

	# reshape arrays for easier handling	
	dE_dX_flat = np.ndarray.flatten(dE_dX)
	dE_dC_flat = np.ndarray.flatten(dE_dC)
	gradE_flat = np.concatenate([dE_dX_flat,dE_dC_flat],0)

	# calculate norms
	gradE_norm = np.linalg.norm( gradE_flat )
	dE_dC_norm = np.linalg.norm( dE_dC )
	dE_dX_norm = np.linalg.norm( dE_dX )

	if hessian == True:
		# Get diagonal element of Hamiltonian
		ham_diagonal = lambda Ia,Ib: prep_ham_diagonal_fast( nstr, nelec_act, norb_act, MO_oints_act, MO_tints_act, Ia, Ib )

		# Effective 1-eri
		#	g_pq = h_pq + sum_{r}^{Nspacial} ( 2*(pq|rr) - (pr|rq) )
		#	Nspacial (noCore,frozenCore) = active
		#	Nspacial (closedCore) = occupied
		eff_oint = lambda i,j: prep_eff_oint( i, j, MO_oints_tot, MO_tints_tot, norb_cut_start, norb_occ )
	
		# CI energy hessian, approximate using davidson CI
		#	Hess_ci_{ii} | _{k} = 2 * ( Hdiag_{ii} - E_{k} )
		Hess_E_ci = np.zeros((nstr**2))
		for i in range(nstr**2):
			Ia_occ = occ_strings[i][0]
			Ib_occ = occ_strings[i][1]
			Hess_E_ci[i] = 2 * ( (ham_diagonal( Ia_occ, Ib_occ ) / np.sum(np.square( C ))) - Eact )
	
		# Orbital energy hessian, approximate as diagonal and with Fock
		Hess_E_orb = build_Hess_E_orb( Xlen, Xindices, coreList, actvList, virtList, eff_oint, rdm1, norb_cut_start )
	
		hess_E_flat = np.concatenate([Hess_E_orb,Hess_E_ci], 0)


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
	dE_dXp,dE_dCp = dE_X_C( Xp, Cp )
	dE_dXm,dE_dCm = dE_X_C( Xm, Cm )
	dEdXp_flat = np.ndarray.flatten( dE_dXp )
	dEdXm_flat = np.ndarray.flatten( dE_dXm )
	dEdCp_flat = np.ndarray.flatten( dE_dCp )
	dEdCm_flat = np.ndarray.flatten( dE_dCm )
	gradEp_flat = np.concatenate([dEdXp_flat,dEdCp_flat],0)
	gradEm_flat = np.concatenate([dEdXm_flat,dEdCm_flat],0)

	# calculate d|dE/dV|^2/dV with central finite difference
	dV_gradE_normsq = (1./lamda) * (gradEp_flat - gradEm_flat)
	dX_gradE_normsq,dC_gradE_normsq = split_array( dV_gradE_normsq )

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
			d2C = np.reshape(d2Lag_dC, [d2Lag_dC.size])
			d2X = np.reshape(d2Lag_dX, [d2Lag_dX.size])
			hess_lag_flat = np.concatenate([d2X,d2C], 0)

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
			d2C = np.reshape(d2Lag_dC, [d2Lag_dC.size])
			d2X = np.reshape(d2Lag_dX, [d2Lag_dX.size])
			hess_lag_flat = np.concatenate([d2X,d2C], 0)

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
	dC = np.reshape(dLag_dC, [dLag_dC.size])
	dX = np.reshape(dLag_dX, [dLag_dX.size])
	grad_flat = np.concatenate([dX,dC], 0)

	#print ('Lag Hess: \n',hess_lag_flat)
	#print ('\nHess condition number: ',np.divide(np.max(np.abs(hess_lag_flat)),np.min(np.abs(hess_lag_flat))))

	## Calculate norm of dE/dV
	#dE_dC = np.reshape(dE_dC, [dE_dC.size])
	#dE_norm = np.linalg.norm( np.concatenate( [dE_dX_flat, dE_dC], 0 ) )

	# print iteration info and update counter
	micro_counter += 1
	print("chi,mu: %2d, %1.2f   func call = %4d    E = %10.12f    |dE_dC| = %10.12f    |dE_dX| = %10.12f    L = %10.12f    |dL| = %10.12f" %(chi, mu, micro_counter, Energy+energy_nuc, dE_dC_norm, dE_dX_norm, Lag, np.linalg.norm(grad_flat)))

	if hessian == True:
		#return Lag, grad_flat, hess_lag_flat, gradE_flat, hess_E_flat, Energy, Eact#, dE_dV_normsq, dV_gradE_normsq	#TODO debugging only
		return Lag, grad_flat, hess_lag_inv
	else:
		return Lag, grad_flat
	
# Set up computational graph
print ("\nSetting up computational graph\n")
sys.stdout.flush()

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
tf_dX = prep_tf_dX(HF_oints_tf,HF_tints_tf)
sys.stdout.flush()
print ('Initialized TensorFlow computational graph, elapsed time in seconds =', ( time.time() - startTime ))
sys.stdout.flush()
print ("\ninitial values:")
var_giveVariables = True
e,c,xflat,xtot,no_occ,ucas = objf(V0)
Emacros.append([macro_counter,e+energy_nuc])
print ("initial E = %4.12f " %(e+energy_nuc))
print ("initial natural orbital occupations = \n",no_occ.tolist())
print ("initial C = \n",np.reshape(c,[nstr,nstr]))
print ("initial X = \n",xflat)


# Optimization setup
print ('\n--------------------------------------')
print ('|                                    |')
print ('|       Starting optimization        |')
print ('|                                    |')
print ('--------------------------------------\n')

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
		V_Xrelax = np.reshape(optResult.x,V0.shape)
	else:
		print ("Using local l-bfgs optimizer that uses the hessian information.")
		optResult,tol = myEngine( V0, mu_Xrelax, chi_Xrelax, thresh_initial, step_control_version=step_control_gridsearch, approxHess=True )
		#optResult,tol = myEngine( V0, mu_Xrelax, chi_Xrelax, thresh_initial, step_control_version=backtrack_linesearch, approxHess=True )
		tol = np.square( tol )
		V_Xrelax = np.reshape(optResult,V0.shape)
	print( "\nAfter Xrelax_iter:%4d   Error: %12.12f  \n" %(macro_counter, tol))
	var_giveVariables = True
	e,c,Xrelaxed_flat,xtot,no_occ,ucas = objf(V_Xrelax)
	Emacros.append([macro_counter,e+energy_nuc])
	print ("\nRelaxed X: \n",Xrelaxed_flat)
	np.savetxt(user_inputs['output_dir']+'/Xflat_relaxed.txt',Xrelaxed_flat)	# checkpoint file for restarting calc if necessary
	#Xrelaxed_flat = np.loadtxt('Xrelaxed.txt')
	Vcurrent = np.concatenate([Xrelaxed_flat,Cguess_flat],0)
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
		Vcurrent = np.reshape(optResult.x,V0.shape)
		tol = np.sum( np.square( optResult.jac ) )
	else:
		print ("Using local l-bfgs optimizer that uses the hessian information.")
		#if macro_counter < 2:
		#	thresh_tmp = 1e-4
		#else:
		#	thresh_tmp = thresh
		optResult,tol = myEngine( Vcurrent, mu, chi, thresh, step_control_version=backtrack_linesearch, approxHess=True )
		tol = np.square( tol )
		Vcurrent = np.reshape(optResult,V0.shape)
	print( "\nAfter macro_iter:%4d    Mu:%2.2f     Chi:%2.2f   Error: %12.12f " %(macro_counter, mu, chi, tol))
	micro_counter = 0
	var_giveVariables = True
	e_temp, c_temp, Xav_temp, Xtot_temp, no_occ_temp, ucas_temp = objf(Vcurrent)
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
		Vcurrent = np.reshape(optResult.x,V0.shape)
		tol = np.sum( np.square( optResult.jac ) )
	else:
		print ("Using local l-bfgs optimizer that uses the hessian information.")
		optResult,tol = myEngine( Vcurrent, mu, chi, thresh, step_control_version=backtrack_linesearch, approxHess=True )
		tol = np.square( tol )
		Vcurrent = np.reshape(optResult,V0.shape)
	print( "\nAfter macro_iter:%4d    Mu:%2.2f     Chi:%2.2f   Error: %12.12f " %(macro_counter, mu, chi, tol))
	micro_counter = 0
	var_giveVariables = True
	e_temp, c_temp, Xav_temp, Xtot_temp, no_occ_temp, ucas_temp = objf(Vcurrent)
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
			Vcurrent = np.reshape(optResult.x,V0.shape)
			tol = np.sum( np.square( optResult.jac ) )
		else:
			print ("Using local l-bfgs optimizer that uses the hessian information.")
			optResult,tol = myEngine( Vcurrent, mu, chi, thresh, step_control_noCore, approxHess=True )
			tol = np.square( tol )
			Vcurrent = np.reshape(optResult,V0.shape)
		print( "\nAfter macro_iter:%4d    Mu:%2.2f     Chi:%2.2f   Error: %12.12f " %(macro_counter, mu, chi, tol))
		micro_counter = 0
		var_giveVariables = True
		e_temp, c_temp, Xav_temp, Xtot_temp, no_occ_temp, ucas_temp = objf(Vcurrent)
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
e,c,xflat,xtot,no_occ,ucas = objf(Vcurrent)
print ("final E = %4.12f"%(e+energy_nuc))
print ("final natural orbital occupations = \n",no_occ)
print ("final C = \n",np.reshape(c,[nstr,nstr]))
print ("final Xflat = \n",xflat)

# Create file to analyze ci vector
print ('\nCi vector MO orbital configurations:')
for i in range(nstr**2):
	if c[i] > 5e-2:
		print ('occupied orbitals (alpha,beta): ',ci2strings(nstr,nelec_act,norb_act,i),' coeff = ', c[i])

# write optimized data to files with output directory
np.savetxt(user_inputs['output_dir']+'/C.txt',c)				# formatted as checkpoint file for restarting calc if necessary
np.savetxt(user_inputs['output_dir']+'/Xtot.txt',xtot)
np.savetxt(user_inputs['output_dir']+'/Xflat.txt',xflat)	# checkpoint file for restarting calc if necessary
np.savetxt(user_inputs['output_dir']+'/U.txt',expm(xtot))
np.savetxt(user_inputs['output_dir']+'/Cguess.txt',Cguess)
np.savetxt(user_inputs['output_dir']+'/Eplot.txt',Emacros)
#os.rename('count.inp',user_inputs['output_dir']+'/count.inp')
#os.rename('fcidump.txt',user_inputs['output_dir']+'/fcidump.txt')

# Create molden files for orbital visualization
if user_inputs['molden'] == True:
	print ('\nCreating molden file for orbital visualization\n')
	U = expm( xtot )
	vis_orbs_molden( U, user_inputs['output_dir']+'/orbs_relaxed', molecule, user_inputs['basis'], norb_core, norb_occ, eigvecs_cas=ucas, natorbs=no_occ )

print ('\nJob finished, elapsed time in seconds = %12.6f' %(time.time() - startTime))
print (  '                           in minutes = %12.6f' %((time.time() - startTime)/60.))
print (  '                           in hours   = %12.6f' %((time.time() - startTime)/3600.))

