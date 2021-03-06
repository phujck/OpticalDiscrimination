import numpy as np
from pyscf import fci

import des_cre as dc
import hub_lats as hub


class hhg(hub.Lattice):
    def __init__(self, cycles, delta, libN, field, nup, ndown, nx, ny, U, t=0.52, F0=10., a=4., lat_type='square',
                 bc=None, nnn=False, last_psi=0, last_J=0, phi=0, iter=0, neighbourexpec=0):
        hub.Lattice.__init__(self, nx, ny, lat_type, nnn, bc)
        self.cycles = cycles
        self.delta = delta
        self.libN = libN
        self.nup = nup
        self.ndown = ndown
        self.ne = nup + ndown
        self.last_psi = last_psi
        self.last_J = last_J
        self.phi = phi
        self.iter = iter
        self.neighbourexpec = neighbourexpec
        #input units: THz (field), eV (t, U), MV/cm (peak amplitude), Angstroms (lattice cst) 
        #converts to a'.u, which are atomic units but with energy normalised to t, so
        #Note, hbar=e=m_e=1/4pi*ep_0=1, and c=1/alpha=137
        factor=1./(t*0.036749323)
        # factor=1
        # factor=1./(0.036749323)
        # factor=1
        self.factor=factor
        # self.factor=1
        self.U=U/t
        # self.U=U
        self.t=1.
        # self.t=t/factor
        #field is the angular frequency, and freq the frequency = field/2pi
        self.field=field*factor*0.0001519828442
        self.freq = self.field/(2.*3.14159265359)
        self.a=(a*1.889726125)/factor
        self.F0=F0*1.944689151e-4*(factor**2)
        assert self.nup<=self.nsites,'Too many ups!'
        assert self.ndown<=self.nsites,'Too many downs!'

def setup(U_input):
    system=hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U_input, t=t, F0=F0, a=a, bc='pbc')
    return system

    
def compute_inner_product(civec, norbs, nelecs, ops, cres, alphas):
  neleca, nelecb = nelecs
  ciket = civec.copy()
  assert(len(ops)==len(cres))
  assert(len(ops)==len(alphas))
  for i in reversed(range(len(ops))):
    if alphas[i]:
      if cres[i]:
        ciket = dc.cre_a(ciket, norbs, (neleca, nelecb), ops[i])
        neleca += 1
      else:
        if neleca==0:
          return 0
        ciket = dc.des_a(ciket, norbs, (neleca, nelecb), ops[i])
        neleca -= 1
    else:
      if cres[i]:
        ciket = dc.cre_b(ciket, norbs, (neleca, nelecb), ops[i])
        nelecb += 1
      else:
        if nelecb==0:
          return 0
        ciket = dc.des_b(ciket, norbs, (neleca, nelecb), ops[i])
        nelecb -= 1
  return np.dot(civec.conj().reshape(-1), ciket.reshape(-1))

# def compute_inner_product(civec, norbs, nelecs, ops, cres, alphas):
#   neleca, nelecb = nelecs
#   ciket = civec.copy()
#   assert(len(ops)==len(cres))
#   assert(len(ops)==len(alphas))
#   for i, (alphas,cres,ops) in enumerate(zip(reversed(alphas),reversed(cres),reversed(ops))):
#     if alphas:
#       if cres:
#         ciket = dc.cre_a(ciket, norbs, (neleca, nelecb), ops)
#         neleca += 1
#       else:
#         if neleca==0:
#           return 0
#         ciket = dc.des_a(ciket, norbs, (neleca, nelecb), ops)
#         neleca -= 1
#     else:
#       if cres:
#         ciket = dc.cre_b(ciket, norbs, (neleca, nelecb), ops)
#         nelecb += 1
#       else:
#         if nelecb==0:
#           return 0
#         ciket = dc.des_b(ciket, norbs, (neleca, nelecb), ops)
#         nelecb -= 1
#   return np.dot(civec.conj().reshape(-1), ciket.reshape(-1))
#
#


#two-body part of hamiltonian
def ham2(lat):
    h = np.zeros((lat.nsites,lat.nsites,lat.nsites,lat.nsites))
    for i in range(lat.nsites):
        h[i,i,i,i] = lat.U
    return h

#calculates the ground state
def hubbard(lat):
    h1 = hub.create_1e_ham(lat,True)
    h2 = ham2(lat)
    cisolver = fci.direct_spin1.FCI()
    e, fcivec = cisolver.kernel(h1, h2, lat.nsites, (lat.nup,lat.ndown))
    return (e,fcivec.reshape(-1))

def progress(total, current):
    if total<10:
        print("Simulation Progress: " + str(round(100*current/total)) + "%")
    elif current%int(total/10)==0:
        print("Simulation Progress: " + str(round(100*current/total)) + "%")
    return


# calculates inner products _after_ adding two electrons

def compute_inner_product_doublon(civec, norbs, nelecs, ops, cres, alphas):
    f=0
    for k in range(norbs):
        neleca, nelecb = nelecs
        civec2=dc.cre_a(civec,norbs,(neleca,nelecb),k)
        neleca+=1
        civec2=dc.cre_b(civec2,norbs,(neleca,nelecb),k)
        nelecb+=1
        ciket = civec2.copy()
        assert (len(ops) == len(cres))
        assert (len(ops) == len(alphas))
        for i in reversed(range(len(ops))):
            if alphas[i]:
                if cres[i]:
                    ciket = dc.cre_a(ciket, norbs, (neleca, nelecb), ops[i])
                    neleca += 1
                else:
                    if neleca == 0:
                        return 0
                    ciket = dc.des_a(ciket, norbs, (neleca, nelecb), ops[i])
                    neleca -= 1
            else:
                if cres[i]:
                    ciket = dc.cre_b(ciket, norbs, (neleca, nelecb), ops[i])
                    nelecb += 1
                else:
                    if nelecb == 0:
                        return 0
                    ciket = dc.des_b(ciket, norbs, (neleca, nelecb), ops[i])
                    nelecb -= 1
        f+=np.dot(civec2.conj().reshape(-1), ciket.reshape(-1))
    return f/norbs

def compute_inner_product_singlon(civec, norbs, nelecs, ops, cres, alphas):
    f=0
    for z in [0,1]:
        for k in range(norbs):
            neleca, nelecb = nelecs
            if z==0:
                civec2=dc.cre_a(civec,norbs,(neleca,nelecb),k)
                neleca+=1
            else:
                civec2=dc.cre_b(civec2,norbs,(neleca,nelecb),k)
                nelecb+=1
            ciket = civec2.copy()
            assert (len(ops) == len(cres))
            assert (len(ops) == len(alphas))
            for i in reversed(range(len(ops))):
                if alphas[i]:
                    if cres[i]:
                        ciket = dc.cre_a(ciket, norbs, (neleca, nelecb), ops[i])
                        neleca += 1
                    else:
                        if neleca == 0:
                            return 0
                        ciket = dc.des_a(ciket, norbs, (neleca, nelecb), ops[i])
                        neleca -= 1
                else:
                    if cres[i]:
                        ciket = dc.cre_b(ciket, norbs, (neleca, nelecb), ops[i])
                        nelecb += 1
                    else:
                        if nelecb == 0:
                            return 0
                        ciket = dc.des_b(ciket, norbs, (neleca, nelecb), ops[i])
                        nelecb -= 1
            f+=np.dot(civec2.conj().reshape(-1), ciket.reshape(-1))
        return f/(2*norbs)



def compute_inner_product_doublon_mix(civec, norbs, nelecs, ops, cres, alphas):
    f=0
    for j in range(norbs):
        for k in range(norbs):
            neleca, nelecb = nelecs
            civec2=dc.cre_a(civec,norbs,(neleca,nelecb),j)
            neleca+=1
            civec2=dc.cre_b(civec2,norbs,(neleca,nelecb),k)
            nelecb+=1
            ciket = civec2.copy()
            assert (len(ops) == len(cres))
            assert (len(ops) == len(alphas))
            for i in reversed(range(len(ops))):
                if alphas[i]:
                    if cres[i]:
                        ciket = dc.cre_a(ciket, norbs, (neleca, nelecb), ops[i])
                        neleca += 1
                    else:
                        if neleca == 0:
                            return 0
                        ciket = dc.des_a(ciket, norbs, (neleca, nelecb), ops[i])
                        neleca -= 1
                else:
                    if cres[i]:
                        ciket = dc.cre_b(ciket, norbs, (neleca, nelecb), ops[i])
                        nelecb += 1
                    else:
                        if nelecb == 0:
                            return 0
                        ciket = dc.des_b(ciket, norbs, (neleca, nelecb), ops[i])
                        nelecb -= 1
            f+=np.dot(civec2.conj().reshape(-1), ciket.reshape(-1))
    return f/(norbs**2)
