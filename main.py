import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import evolve as evolve
import observable as observable
import definition as harmonic
import hub_lats as hub
import harmonic as har_spec
import des_cre as dc
from matplotlib import cm as cm
from scipy.integrate import ode
import des_cre as dc

#input units: THz (field), eV (t, U), MV/cm (peak amplitude, F0), Angstroms (systems[0]tice cst, a)
#they're then converted to t-normalised atomic units. bc='pbc' for periodic and 'abc' for antiperiodic


neighbour = []
neighbour_check=[]
energy=[]
doublon_energy=[]
phi_original = []
J_field = []
phi_reconstruct = [0., 0.]
boundary_1 = []
boundary_2 = []
two_body = []
two_body_old=[]
error=[]
D=[]
X=[]
singlon=[]
systems=[]
number=3
nelec = (number, number)
nx = 6
ny = 0
t = 0.52
# t=1.91
# t=1
U = 1*t
delta = 0.01
cycles = 5
field= 32.9
# field=32.9*0.5
F0=10
a=4

libN=5
time = cycles

def phi_func(current_time):
    if current_time < time / (libN * systems[0].freq):
        phi = (systems[k].a * systems[k].F0 / systems[k].field) * np.sin(current_time) ** 2.
    else:
        phi = 0
    return phi

# phi_original=[phi_func(j) for j in np.linspace(0,cycles/(libN * systems[0].freq), )]

plt.plot(phi_original)
plt.show()
for k in range(1,libN):
    parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude' % (
    nx, cycles, U, t, number, delta, field, F0)
    U=k*t
    systems.append(harmonic.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc'))
    N = int(time / (systems[0].freq * delta)) + 1
    print("$t_0$:")
    print(systems[0].t)
    print("a:")
    print(systems[0].a)
    print("field:")
    print(systems[0].field)
    print("$E_0:")
    print(systems[0].F0)
    print('\n')
    print(vars(systems[0]))
    psi_temp = harmonic.hubbard(systems[k])[1].astype(complex)
    init=psi_temp
    h= hub.create_1e_ham(systems[0],True)
    print(N)


    prop=systems[k]
    r = ode(evolve.integrate_f_discrim).set_integrator('zvode', method='bdf')
    r.set_initial_value(psi_temp, 0).set_f_params(systems[0],h,phi_func)
    branch = 0
    delta=delta
    while r.successful() and r.t < k*time/(libN*systems[0].freq):
        oldpsi=psi_temp
        r.integrate(r.t + delta)
        psi_temp = r.y
        newtime = r.t
        # add to expectations

        # double occupancy fails for anything other than half filling.
        # D.append(evolve.DHP(prop,psi))
        harmonic.progress(N, int(newtime / delta))
        # psierror=evolve.f(systems[0],evolve.ham1(systems[0],h,newtime,time),oldpsi)
        # diff = (psi_temp - oldpsi) / delta
        # newerror = np.linalg.norm(diff + 1j * psierror)
        # error.append(newerror)
        neighbour.append(har_spec.nearest_neighbour_new(systems[0], h, psi_temp))
        # neighbour_check.append(har_spec.nearest_neighbour(systems[0], psi_temp))
        # X.append(observable.overlap(systems[0], psi_temp)[1])
        phi_original.append(phi_func(newtime))
        J_field.append(har_spec.J_expectation_track(systems[0], h, psi_temp, phi_original[-1]))
        two_body.append(har_spec.two_body_old(systems[0], psi_temp))
        D.append(observable.DHP(systems[0], psi_temp))


last=J_field[-1]
lastime=newtime
def J_func(current_time):
    J=last*np.exp(-2*(current_time-lastime))
    return J


r = ode(evolve.integrate_f_track_J).set_integrator('zvode', method='bdf')
r.set_initial_value(psi_temp, newtime).set_f_params(systems[0],h,J_func)
branch = 0
delta=delta
while r.successful() and r.t < time/(systems[0].freq):
    oldpsi=psi_temp
    r.integrate(r.t + delta)
    psi_temp = r.y
    newtime = r.t
    # add to expectations
    neighbour.append(har_spec.nearest_neighbour_new(systems[0], h, psi_temp))
    two_body.append(har_spec.two_body_old(systems[0], psi_temp))

    # tracking current
    phi_original.append(evolve.phi_J_track(systems[0], newtime, J_func, neighbour[-1], psi_temp))

    # tracking D
    # phi_original.append(evolve.phi_D_track(lat,newtime,D_func,two_body[-1],psi_temp))

    J_field.append(har_spec.J_expectation_track(systems[0], h, psi_temp, phi_original[-1]))
    # double occupancy fails for anything other than half filling.
    # D.append(evolve.DHP(prop,psi))
    harmonic.progress(N, int(newtime / delta))

# np.save('./data/original/Jfield'+parameternames,J_field)
# np.save('./data/original/phi'+parameternames,phi_original)
np.save('./data/original/discriminatorfield', phi_original)
# np.save('./data/original/phirecon'+parameternames,phi_reconstruct)
# np.save('./data/original/neighbour'+parameternames,neighbour)
# np.save('./data/original/energy'+parameternames,energy)
# np.save('./data/original/doublonenergy'+parameternames,doublon_energy)
# np.save('./data/original/singlonenergy'+parameternames,singlon)



# np.save('./data/original/position'+parameternames,X)



# plot_observables(systems[0], delta=0.02, time=5., K=.1)
# spectra(systems[0], initial=None, delta=delta, time=cycles, method='welch', min_spec=7, max_harm=50, gabor='fL')
