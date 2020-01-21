import numpy as np
import evolve as evolve
import definition as harmonic
import observable as observable
import hub_lats as hub
import harmonic as har_spec
from scipy.integrate import ode
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# NOTE: time is inputted and plotted in terms of cycles, but the actual propagation happens in 'normal' time


# input units: THz (field), eV (t, U), MV/cm (peak amplitude, F0), Angstroms (lattice cst, a)
# they're then converted to t-normalised atomic units. bc='pbc' for periodic and 'abc' for antiperiodic


neighbour = []
phi_original = []
phi_reconstruct = [0., 0.]
boundary_1 = []
boundary_2 = []
two_body = []
two_body_old = []
error = []
J_field = []
D_cutfreq = []
Jalt=[]
systems=[]

number = 3
nelec = (number, number)
nx = 6
ny = 0
t = 0.52
# t=1.91
# t=1
U = 0.99 * t
cutoff = 40
delta = 0.01
cycles = 2
field = 32.9
# field=25
F0 = 10
a = 4
ascale = 1


parameternames='-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude' % (nx,cycles,U,t,number,delta,field,F0)
phi_cut = np.load('./data/original/discriminatorfield.npy')
for k in [0,1]:
    systems.append(harmonic.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc'))

print(systems[0].field)
print('\n')
print(vars(systems[0]))
time=cycles
psi_temp = harmonic.hubbard(systems[0])[1].astype(complex)
init=psi_temp
h= hub.create_1e_ham(systems[0],True)
N = int(time/(systems[0].freq*delta))+1

times = np.linspace(0.0, cycles/systems[0].freq, len(phi_cut))
phi_func = interp1d(times, phi_cut, fill_value=0, bounds_error=False, kind='cubic')

r = ode(evolve.integrate_f_discrim).set_integrator('zvode', method='bdf')
r.set_initial_value(psi_temp, 0).set_f_params(systems[0],h,phi_func)
branch = 0
delta=delta
while r.successful() and r.t < time/systems[0].freq:
    oldpsi=psi_temp
    r.integrate(r.t + delta)
    psi_temp = r.y
    newtime = r.t
    neighbour.append(har_spec.nearest_neighbour_new(systems[0], h, psi_temp))
    phi_original.append(phi_func(newtime))
    J_field.append(har_spec.J_expectation_track(systems[0], h, psi_temp, phi_original[-1]))
    harmonic.progress(N, int(newtime / delta))




np.save('./data/discriminate/Jfield'+parameternames,J_field)
np.save('./data/discriminate/phi'+parameternames,phi_original)
np.save('./data/discriminate/discriminatorfield', phi_original)