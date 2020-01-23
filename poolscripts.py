import numpy as np
# from tqdm import tqdm
import definition as harmonic
import poolscripts
import observable as observable
import hub_lats as hub
import harmonic as har_spec
from scipy.integrate import ode
from scipy.interpolate import interp1d
# This contains the stuff needed to calculate some expectations. Generally contains stuff
# that applies operators to the wave function
import evolve as evolve

# Contains lots of important functions.
# import definition as definition
# Sets up the lattice for the system
import hub_lats as hub

# These also contain various important observable calculators
import harmonic as har_spec

def get_U(system):
    return system.U






def pump(system):
    def phi_pump(current_time):
        frac=1
        if current_time < frac/(system.freq):
            phi = (system.a * system.F0 / system.field) * np.sin(current_time*system.field/frac)
        else:
            phi=0.
        return phi
    phi_original=[]
    psi_temp = harmonic.hubbard(system)[1].astype(complex)
    init=psi_temp
    h= hub.create_1e_ham(system,True)
    r = ode(evolve.integrate_f_discrim).set_integrator('zvode', method='bdf')
    r.set_initial_value(psi_temp, 0).set_f_params(system,h,phi_pump)
    while r.successful() and r.t < 1/system.freq:
        oldpsi=psi_temp
        r.integrate(r.t + system.delta)
        psi_temp = r.y
        newtime = r.t
        # add to expectations

        # double occupancy fails for anything other than half filling.
        # D.append(evolve.DHP(prop,psi))
        # harmonic.progress(N, int(newtime / delta))
        phi_original.append(phi_pump(newtime))
    J_field=har_spec.J_expectation_track(system, h, psi_temp, phi_original[-1])
    system.last_psi=psi_temp
    system.last_J=J_field
    system.phi=phi_original
    return system

#
def evolve_others(system):
    phi_original = system.phi
    inittime = 1/system.freq+system.iter*(system.cycles-1)/(system.libN*system.freq)
    times=np.linspace(0,(system.iter+1)*system.cycles/(system.libN*system.freq),len(system.phi))
    phi_func = interp1d(times, phi_original, fill_value=0, bounds_error=False, kind='cubic')

    psi_temp = harmonic.hubbard(system)[1].astype(complex)
    init=psi_temp
    h= hub.create_1e_ham(system,True)
    r = ode(evolve.integrate_f_discrim).set_integrator('zvode', method='bdf')
    r.set_initial_value(psi_temp, 0).set_f_params(system,h,phi_func)
    f=system.freq
    while r.successful() and r.t <inittime+(system.iter+1)*(system.cycles-1)/(system.libN*system.freq):
        oldpsi=psi_temp
        r.integrate(r.t + system.delta)
        psi_temp = r.y
        newtime = r.t
        # add to expectations

        # double occupancy fails for anything other than half filling.
        # D.append(evolve.DHP(prop,psi))
        # harmonic.progress(N, int(newtime / delta))
    phi_original=(phi_func(newtime))
    J_field=har_spec.J_expectation_track(system, h, psi_temp, phi_original)
    system.last_psi=psi_temp
    system.last_J=J_field
    return system