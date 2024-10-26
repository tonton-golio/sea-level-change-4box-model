# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import corner
import sys

# basic box model
def box_model(F, T_0, a, b, tau):
    T = np.zeros(len(F))
    T[0] = T_0
    T_eq = a * F + b
    for t in range(1, len(T)):
        dT_dt = (T_eq[t-1] - T[t-1]) / tau
        T[t] = T[t-1] + dT_dt
    return T

# Greenland
def precipitation(area, T, z):
    """
    this z should be shared between Greenland and Antarctica
    """

    # water carrying capacity of air
    C = lambda T: 6.6106 * np.exp(0.0499*T)

    precipitation_rate = area * C(T) * z

    return precipitation_rate

def melt(length_boundary, height_avg, T, T_c, z):
    """
    This z can maybe be shared
    """

    # area of boundary
    area = length_boundary * height_avg


    melt_rate = area * np.exp((T-T_c)/z)
    return melt_rate

def discharge(volume_overhang, T_ocean, T_c, z):
    """
    z shared?
    """
    return volume_overhang * np.exp(T_ocean-T_c) * z

def sublimation(area, radioactive_forcing, T, z):
    return area * radioactive_forcing * T * z

def greenland(F, T_air_greenland, T_ocean_greenland, z1, z2, z3, z4, constants, **kwargs):
    # extract constants
    area_greenland = constants['area_greenland']
    l_boundary_greenland = constants['l_boundary_greenland']
    h_avg_greenland = constants['h_avg_greenland']
    volume_overhang_greenland = constants['volume_overhang_greenland']
    T_c = constants['T_c']
    area_ocean = constants['area_ocean']

    # calculate terms
    P = precipitation(area_greenland, T_air_greenland, z1)
    L = melt(l_boundary_greenland, h_avg_greenland, T_ocean_greenland, T_c, z2)
    D = discharge(volume_overhang_greenland, T_ocean_greenland, T_c, z3)
    B = sublimation(area_greenland, F, T_air_greenland, z4)

    # calculate mass change
    mass_change = P - (L + D + B)

    # calculate sea level change
    dS = mass_change/area_ocean

    if kwargs.get('verbose', False):
        print(f'P = {P}')
        print(f'L = {L}')
        print(f'D = {D}')
        print(f'B = {B}')
        print(f"dM = {mass_change}")
        print(f"dS = {dS}")

    return dS

# data loading
def load_forcing(path = '../data/schmidt_11_paleoforcings_cleaned.csv'):
    df = pd.read_csv(path, index_col=0)

    F = df['Total forcing [V.1,S.1,L,G]'].values
    time_forcing = df.index.values

    # print(f'F = {F}')
    # print(f'time_forcing = {time_forcing}')

    return F, time_forcing

def load_dangendorf():
    df = pd.read_csv('../data/dangendorf_data.csv', index_col=0)
    time_sea_level = df.index.values
    sea_level = df['total'].values
    sea_level_error = df['std'].values

    # print(f'time_sea_level = {time_sea_level}')
    # print(f'sea_level = {sea_level}')
    # print(f'sea_level_error = {sea_level_error}')

    return sea_level, sea_level_error, time_sea_level

# error calculation
def calc_error(S, S_pred, time, time_pred):
    # Find the indices of common times
    common_times = np.intersect1d(time, time_pred, assume_unique=True)
    
    # Create masks for common indices in time and time_pred
    time_mask = np.isin(time, common_times)
    time_pred_mask = np.isin(time_pred, common_times)

    # Align S and S_pred with the common times
    S_aligned = S[time_mask]
    S_pred_aligned = S_pred[time_pred_mask]

    # Mean Squared Error
    mse = np.sum((S_aligned - S_pred_aligned) ** 2)
    
    # Negative Log Likelihood
    neg_log_likelihood = -0.5 * np.sum((S_aligned - S_pred_aligned) ** 2) + np.log(2 * np.pi * mse)

    return neg_log_likelihood

def get_S(params):
    z1 = params['z1']
    z2 = params['z2']
    z3 = params['z3']
    z4 = params['z4']
    S_0 = params['S_0']
    a = params['a']
    b = params['b']
    tau = params['tau']
    T_0 = params['T_0']
    
    # get temperature evolution with forward Euler
    T_air_greenland = box_model(F, T_0, a, b, tau)
    T_ocean_greenland = T_air_greenland

    # evolve sea level
    S = np.zeros(len(F))
    S[0] = S_0

    for i in range(len(F)-1):
        T_air_greenland_i = T_air_greenland[i]
        T_ocean_greenland_i = T_ocean_greenland[i]
        F_i = F[i]

        dS = greenland(F_i, T_air_greenland_i, T_ocean_greenland_i, z1, z2, z3, z4, constants, verbose=False)

        S[i+1] = S[i] + dS

    return S

# define prior and likelihood
def log_prior(theta):
    z1, z2, z3, z4, S_0, a, b, tau, T_0 = theta
    cond1 = 0 < z1 < 100
    cond2 = 0 < z2 < 1e12
    cond3 = 0 < z3 < 1
    cond4 = 0 < z4 < 1
    cond5 = -3000 < S_0 < 3500
    cond6 = 0 < a < 10
    cond7 = 0 < b < 20
    cond8 = 0 < tau < 200
    cond9 = -50 < T_0 < 50
    if cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7 and cond8 and cond9:
        return 0.0  # Uniform prior (log probability is zero within bounds)
    return -np.inf  # Log probability is -infinity outside bounds

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    S = get_S({k: v for k, v in zip(['z1', 'z2', 'z3', 'z4', 'S_0', 'a', 'b', 'tau', 'T_0'], theta)})
    neq_log_likelihood = calc_error(sea_level, S, time_sea_level, time_forcing)
    return lp + neq_log_likelihood

# constants
constants = {
    'area_greenland': 2.2 * 10**6 *10**3,  # m^2
    'l_boundary_greenland': 4.4 *10**4 * 10**3,  # m
    'h_avg_greenland': 100,  # m
    'volume_overhang_greenland': 1200, # ???
    'T_c': -3,  # deg C
    'area_ocean': 20000000  # m^2
}

# load data
F, time_forcing = load_forcing()  # open forcing data
sea_level, sea_level_error, time_sea_level = load_dangendorf()  # open sea level data

fig, ax = plt.subplots(1, 1, figsize=(8, 5), sharex=True)
ax = [ax, ax.twinx()]
ax[0].plot(time_forcing[1000:],  F[1000:], color='red')
ax[0].set_ylabel('Radiative forcing [Wm$^{-2}$]', color='red')
ax[1].plot(time_sea_level, sea_level, color='blue')
ax[1].set_ylabel('Sea level anomaly [mm]', color='blue')
ax[1].set_xlabel('Time')
fig.suptitle('Radiative forcing and sea level anomaly')
plt.tight_layout()
plt.show()


sys.exit()

# fitting params
params = {
    'z1': 20,
    'z2': 10000000000,
    'z3': .0000001,
    'z4': .0000001,
    'S_0': -200,
    'a': 1,
    'b': 10,
    'tau': 120,
    'T_0': 10
}

# try objective function
S = get_S(params)
error = calc_error(sea_level, S, time_sea_level, time_forcing)
print(f'Error = {error}')

# optimize with MCMC
initial = [params['z1'], params['z2'], params['z3'], params['z4'], params['S_0'], params['a'], params['b'], params['tau'], params['T_0']]


ndim, nwalkers = len(initial), 100
pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)


sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)

print("Running burn-in...")
pos, _, _ = sampler.run_mcmc(pos, 50, progress=True)

print("Running production...")
sampler.reset()
sampler.run_mcmc(pos, 100, progress=True)

flat_samples = sampler.get_chain(discard=50, thin=15, flat=True)
print(flat_samples.shape)


labels = ['z1', 'z2', 'z3', 'z4', 'S_0', 'a', 'b', 'tau', 'T_0']
fig = corner.corner(flat_samples, labels=labels)
plt.show()

# get best params
best_params = np.mean(flat_samples, axis=0)
best_S = get_S({k: v for k, v in zip(labels, best_params)})
best_error = calc_error(sea_level, best_S, time_sea_level, time_forcing)

print(f'Best params = {best_params}')
print(f'Best error = {best_error}')
print(f'Initial error = {error}')

# plot
plt.plot(time_sea_level, sea_level, label='Data')
plt.plot(time_forcing, best_S, label='Model')
plt.legend()
plt.show()