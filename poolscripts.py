# from tqdm import tqdm
# This contains the stuff needed to calculate some expectations. Generally contains stuff
# that applies operators to the wave function
# These also contain various important observable calculators
# Contains lots of important functions.
# import definition as definition
# Sets up the lattice for the system


def get_U(system):
    return system.U






def pump(system):
    def phi_pump(current_time):
        frac = 0.5
        if current_time < 1 / (system.freq):
            phi = (system.a * system.F0 / system.field) * np.sin(current_time * system.field / (frac)) * np.sin(
                0.33 * current_time * system.field) ** 2
        else:
            phi = 0.
        return phi

    phi_original = []
    J_field = []
    neighbours = []
    psi_temp = harmonic.hubbard(system)[1].astype(complex)
    init = psi_temp
    h = hub.create_1e_ham(system, True)
    r = ode(evolve.integrate_f_discrim).set_integrator('zvode', method='bdf')
    r.set_initial_value(psi_temp, 0).set_f_params(system, h, phi_pump)
    while r.successful() and r.t < 1 / system.freq:
        oldpsi = psi_temp
        r.integrate(r.t + system.delta)
        psi_temp = r.y
        newtime = r.t
        # add to expectations

        # double occupancy fails for anything other than half filling.
        # D.append(evolve.DHP(prop,psi))
        # harmonic.progress(N, int(newtime / delta))
        phi_original.append(phi_pump(newtime))
        neighbour = har_spec.nearest_neighbour_new(system, h, psi_temp)
        J_field.append(har_spec.J_expectation_track(system, h, psi_temp, phi_original[-1]))
        neighbours.append(neighbour)
    system.last_psi = psi_temp
    system.last_J = J_field
    system.phi = phi_original
    system.neighbourexpec = neighbours
    return system

import numpy as np
from scipy.integrate import ode
from scipy.interpolate import interp1d

# from tqdm import tqdm
import definition as harmonic
# This contains the stuff needed to calculate some expectations. Generally contains stuff
# that applies operators to the wave function
import evolve as evolve
# These also contain various important observable calculators
import harmonic as har_spec
# Contains lots of important functions.
# import definition as definition
# Sets up the lattice for the system
import hub_lats as hub


def get_U(system):
    return system.U






def long_pump(system):
    def phi_pump(current_time):
        frac = 1
        phi = (system.a * system.F0 / system.field) * np.sin(current_time * system.field / (frac))*np.sin(0.5*current_time * system.field/system.cycles)**2
        return phi

    phi_original = []
    J_field = []
    neighbours = []
    psi_temp = harmonic.hubbard(system)[1].astype(complex)
    init = psi_temp
    h = hub.create_1e_ham(system, True)
    r = ode(evolve.integrate_f_discrim).set_integrator('zvode', method='bdf')
    r.set_initial_value(psi_temp, 0).set_f_params(system, h, phi_pump)
    while r.successful() and r.t < system.cycles / system.freq:
        oldpsi = psi_temp
        r.integrate(r.t + system.delta)
        psi_temp = r.y
        newtime = r.t
        # add to expectations

        # double occupancy fails for anything other than half filling.
        # D.append(evolve.DHP(prop,psi))
        # harmonic.progress(N, int(newtime / delta))
        phi_original.append(phi_pump(newtime))
        neighbour = har_spec.nearest_neighbour_new(system, h, psi_temp)
        J_field.append(har_spec.J_expectation_track(system, h, psi_temp, phi_original[-1]))
        neighbours.append(neighbour)
    system.last_psi = psi_temp
    system.last_J = J_field
    system.phi = phi_original
    system.neighbourexpec = neighbours
    return system


def pump_RK4(system):
    delta = system.delta

    def phi_pump(current_time):
        frac = 1
        if current_time < frac / (system.freq):
            phi = (system.a * system.F0 / system.field) * np.sin(current_time * system.field / (2 * frac))
        else:
            phi = 0.
        return phi

    N = int(1 / (system.freq * delta))
    phi_original = []
    J_field = []
    psi_temp = harmonic.hubbard(system)[1].astype(complex)
    init = psi_temp
    h = hub.create_1e_ham(system, True)
    for k in range(N):
        phi_original.append(phi_pump(k * delta))
        J_field.append(har_spec.J_expectation_track(system, h, psi_temp, phi_original[-1]))
        psi_temp = evolve.RK4_phi_track(system, h, delta, k * delta, phi_pump, psi_temp)
        # add to expectations
    phi_original.append(phi_pump(N))
    J_field.append(har_spec.J_expectation_track(system, h, psi_temp, phi_original[-1]))
    # double occupancy fails for anything other than half filling.
    # D.append(evolve.DHP(prop,psi))
    # harmonic.progress(N, int(newtime / delta))

    system.last_psi = psi_temp
    system.last_J = J_field
    system.phi = phi_original
    return system


#
def evolve_others(system):
    J_field = system.last_J.copy()
    phi_original = system.phi.copy()
    psi_temp = system.last_psi.copy()
    neighbours = system.neighbourexpec.copy()
    newphis = []
    newtimes = []
    inittime = 1 / system.freq + system.iter * (system.cycles - 1) / (system.libN * system.freq)
    # times = np.linspace(inittime, (system.iter + 1) * (system.cycles-1) / (system.libN * system.freq), len(system.phi))
    times = np.linspace(0, 1 / system.freq + (system.iter + 1) * (system.cycles - 1) / (system.libN * system.freq),
                        len(system.phi))
    phi_func = interp1d(times, phi_original, fill_value='extrapolate', bounds_error=False, kind='cubic')

    h = hub.create_1e_ham(system, True)
    r = ode(evolve.integrate_f_discrim).set_integrator('zvode', method='bdf')
    r.set_initial_value(psi_temp, inittime).set_f_params(system, h, phi_func)
    f = system.freq
    while r.successful() and r.t < 1 / system.freq + (system.iter + 1) * (system.cycles - 1) / (
            system.libN * system.freq):
        oldpsi = psi_temp
        r.integrate(r.t + system.delta)
        psi_temp = r.y
        newtime = r.t
        # add to expectations
        newphis.append(phi_func(newtime))
        newtimes.append(newtime)
        # double occupancy fails for anything other than half filling.
        # D.append(evolve.DHP(prop,psi))
        # harmonic.progress(N, int(newtime / delta))
        neighbour = har_spec.nearest_neighbour_new(system, h, psi_temp)
        J_field.append(har_spec.J_expectation_track(system, h, psi_temp, phi_func(newtime)))
        neighbours.append(neighbour)
    # plt.plot(times, phi_func(times))
    # plt.plot(newtimes,newphis,'--')
    # plt.show()
    system.last_psi = psi_temp
    system.last_J = J_field
    system.neighbourexpec = neighbours
    return system


def evolve_others_RK4(system):
    delta = system.delta
    k = system.iter
    phi_original = system.phi.copy()
    init_step = int((k + 1) / (system.freq * system.delta))
    end_step = init_step + int(1 / (system.freq * system.delta))
    J_field = system.last_J.copy()[:-1]
    psi_temp = system.last_psi
    h = hub.create_1e_ham(system, True)
    times = np.linspace(0, end_step * system.delta, len(system.phi))
    # phi_func = interp1d(times, phi_original, fill_value=0, bounds_error=False, kind='cubic')
    phi_func = interp1d(times, phi_original, fill_value='extrapolate', bounds_error=False, kind='cubic')
    phi_original = phi_original[:-1]
    for j in range(init_step, end_step):
        newtime = j * system.delta
        phi_original.append(phi_func(newtime))
        J_field.append(har_spec.J_expectation_track(system, h, psi_temp, phi_original[-1]))
        psi_temp = evolve.RK4_phi_track(system, h, delta, k * delta, phi_func, psi_temp)
    newtime = end_step * system.delta
    phi_original.append(phi_func(newtime))
    J_field.append(har_spec.J_expectation_track(system, h, psi_temp, phi_original[-1]))
    system.last_psi = psi_temp
    system.last_J = J_field
    return system


def evolve_simultaneous(system):
    J_field = system.last_J.copy()
    phi_original = system.phi.copy()
    psi_temp = system.last_psi.copy()
    inittime = 1 / system.freq + system.iter * (system.cycles - 1) / (system.libN * system.freq)
    # times = np.linspace(inittime, (system.iter + 1) * (system.cycles-1) / (system.libN * system.freq), len(system.phi))
    times = np.linspace(0, (system.iter + 1) * (system.cycles - 1) / (system.libN * system.freq), len(system.phi))
    phi_func = interp1d(times, phi_original, fill_value=0, bounds_error=False, kind='cubic')

    h = hub.create_1e_ham(system, True)
    r = ode(evolve.integrate_f_discrim).set_integrator('zvode', method='bdf')
    r.set_initial_value(psi_temp, inittime).set_f_params(system, h, phi_func)
    f = system.freq
    while r.successful() and r.t < inittime + (system.cycles - 1) / (system.libN * system.freq):
        oldpsi = psi_temp
        r.integrate(r.t + system.delta)
        psi_temp = r.y
        newtime = r.t
        # add to expectations

        # double occupancy fails for anything other than half filling.
        # D.append(evolve.DHP(prop,psi))
        # harmonic.progress(N, int(newtime / delta))
        J_field.append(har_spec.J_expectation_track(system, h, psi_temp, phi_func(newtime)))
    system.last_psi = psi_temp
    system.last_J = J_field
    system.iter += 1
    return system

def evolve_discrimate(system):
    J_field = []
    newphi = []
    phi_original = system.phi.copy()
    times = np.linspace(0, system.cycles / system.freq, len(system.phi))
    phi_func = interp1d(times, phi_original, fill_value=0, bounds_error=False, kind='cubic')

    psi_temp = harmonic.hubbard(system)[1].astype(complex)
    init = psi_temp
    h = hub.create_1e_ham(system, True)
    r = ode(evolve.integrate_f_discrim).set_integrator('zvode', method='bdf')
    r.set_initial_value(psi_temp, 0).set_f_params(system, h, phi_func)
    f = system.freq
    while r.successful() and r.t < system.cycles / system.freq:
        oldpsi = psi_temp
        r.integrate(r.t + system.delta)
        psi_temp = r.y
        newtime = r.t
        # add to expectations
        newphi.append(phi_func(newtime))
        # double occupancy fails for anything other than half filling.
        # D.append(evolve.DHP(prop,psi))
        # harmonic.progress(N, int(newtime / delta))
        J_field.append(har_spec.J_expectation_track(system, h, psi_temp, newphi[-1]))
    system.phi = newphi
    system.last_J = J_field
    return system
