#!/usr/bin/env python

import numpy as np
from my_tools import *

#_______________________________
# Approximate Hessian functions
#_______________________________

def prep_ham_diagonal_fast( nstr, nelec_act, norb_act, oints, tints, Ia_occ, Ib_occ ):

	Hdiag = 0
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

