import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import emcee
import corner
from tqdm import tqdm

class SeaLevelModel:
    def __init__(self, df_F_path, df_S_path):
        """
        Initialize the Sea Level Model with forcing and sea level data.

        Parameters:
        - df_F_path: str, path to the forcing data CSV file.
        - df_S_path: str, path to the sea level data CSV file.
        """
        # Load data
        self.df_F = pd.read_csv(df_F_path, index_col=0)
        self.df_S = pd.read_csv(df_S_path, index_col=0)
        
        # Preprocess data
        self._preprocess_data()

    def _preprocess_data(self):
        """
        Preprocess the data: prepare forcing and sea level data separately.
        """
        # Convert sea level data to meters if necessary
        # if self.df_S['GMSL_noGIA'].max() > 1:
        #     self.df_S['GMSL_noGIA'] /= 1000  # Convert mm to meters
        #     self.df_S['uncertainty'] /= 1000  # Convert mm to meters

        # Ensure indices are integers (years)
        # self.df_S.set_index as time
        self.df_F.index = self.df_F.index.astype(int)
        self.df_S.index = self.df_S.index.astype(int)

        # Forcing data (full time series)
        self.F_full = self.df_F['total'].values
        self.years_full = self.df_F.index.values


        # Observed sea level data
        self.SL_obs = self.df_S['GMSL (mm)'].values
        self.SL_unc = self.df_S['GMSL uncertainty (mm)'].values
        self.years_obs = self.df_S.index.values.astype(int)

        # Align forcing data with observed data for likelihood computation
        common_years = np.intersect1d(self.years_full, self.years_obs)
        F_obs_indices = np.isin(self.years_full, common_years)
        SL_obs_indices = np.isin(self.years_obs, common_years)

        self.F_obs = self.F_full[F_obs_indices]
        self.years_common = self.years_full[F_obs_indices]

        # Re-order sea level observations to match the years in common
        SL_obs_order = np.argsort(self.years_obs[SL_obs_indices])
        self.SL_obs = self.SL_obs[SL_obs_indices][SL_obs_order]
        self.SL_unc = self.SL_unc[SL_obs_indices][SL_obs_order]
        self.years_obs = self.years_obs[SL_obs_indices][SL_obs_order]

    def get_S(self, F, S_0, tau2, a2, b2, S_eq_func=None):
        """
        Compute the sea level S based on the forcing F and model parameters.

        Parameters:
        - F: array-like, radiative forcing data.
        - S_0: float, initial sea level.
        - tau2: float, time constant.
        - a2, b2: floats, model coefficients.
        - S_eq_func: callable, optional equilibrium sea level function.

        Returns:
        - S: array-like, modeled sea level.
        """
        if S_eq_func is None:
            S_eq = a2*100 * F + b2*1000
        else:
            S_eq = S_eq_func(F, a2, b2)

        S = np.zeros(len(F))
        S[0] = S_0
        for i in range(1, len(F)):
            ds = (S_eq[i-1] - S[i-1]) / tau2
            S[i] = S[i-1] + ds
        
        return S

    def plot_data(self):
        """
        Plot the radiative forcing and sea level data.
        """
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        
        # Plot radiative forcing (full time series)
        ax1.plot(self.years_full, self.F_full, color='blue', label='Radiative Forcing')
        ax1.set_ylabel('Radiative Forcing (W/m^2)', color='blue')
        ax1.set_xlabel('Year')
        
        # Plot sea level observations with error bars
        # ax2.errorbar(self.years_obs, self.SL_obs, yerr=self.SL_unc,
        #              fmt='.', color='black', label='Sea Level Observations')
        ax2.fill_between(self.years_obs, self.SL_obs - self.SL_unc, self.SL_obs + self.SL_unc, color='black', alpha=0.3)
        ax2.set_ylabel('Sea Level Anomaly (mm)', color='black')
        
        fig.suptitle('Radiative Forcing and Sea Level Rise')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.show()

    def curve_fit_model(self, S_eq_func=None):
        """
        Fit the sea level model to the data using scipy's curve_fit.

        Returns:
        - popt: array, optimized parameters.
        - pcov: 2D array, covariance of the optimized parameters.
        """
        # Wrapper function for curve_fit
        def model_func(F_obs, S_0, tau2, a2, b2):
            # Compute modeled sea level over full time series
            S_model_full = self.get_S(self.F_full, S_0, tau2, a2, b2, S_eq_func)
            # Extract modeled sea level at observation years
            S_model_obs = np.interp(self.years_obs, self.years_full, S_model_full)
            return S_model_obs
        
        # Initial guess for the parameters
        initial_guess = [-0.2, 40, 5, 2]
        
        # Perform curve fitting
        popt, pcov = curve_fit(
            model_func, self.F_obs, self.SL_obs, sigma=self.SL_unc,
            p0=initial_guess, absolute_sigma=True, maxfev=10000
        )
        return popt, pcov

    def log_prior(self, theta):
        """
        Compute the log prior probability of the parameters.

        Parameters:
        - theta: array-like, model parameters.

        Returns:
        - lp: float, log prior probability.
        """
        S_0, tau2, a2, b2 = theta
        # Example of normal priors
        lp = 0
        lp += -0.5 * ((S_0 + 800)/100)**2  # Mean at -0.1, sigma=0.5
        lp += -0.5 * ((tau2 - 350)/100)**2  # Mean at 350, sigma=200
        lp += -0.5 * ((a2 - 5)/3)**2      # Mean at 5, sigma=3
        lp += -0.5 * ((b2 - 5)/3)**2       # Mean at 5, sigma=3

        # Check if parameters are within bounds
        # if -2 < S_0 < 0 and 1 < tau2 < 700 and -2 < a2 < 90 and -1 < b2 < 20:
        if -2000 < S_0 < 2000 and 0 < tau2 < 1000 and -100 < a2 < 100 and -100 < b2 < 100:
            return lp
        return -np.inf

    def log_likelihood(self, theta, S_eq_func=None):
        """
        Compute the log likelihood of the data given the parameters.

        Parameters:
        - theta: array-like, model parameters.
        - S_eq_func: callable, optional equilibrium sea level function.

        Returns:
        - ll: float, log likelihood.
        """
        S_0, tau2, a2, b2 = theta
        # Compute modeled sea level over full time series
        S_model_full = self.get_S(self.F_full, S_0, tau2, a2, b2, S_eq_func)
        # Interpolate modeled sea level at observation years
        S_model_obs = np.interp(self.years_obs, self.years_full, S_model_full)
        residuals = self.SL_obs - S_model_obs
        ll = -0.5 * np.sum((residuals / self.SL_unc)**2 + np.log(2 * np.pi * self.SL_unc**2))
        return ll

    def log_probability(self, theta, S_eq_func=None):
        """
        Compute the total log probability (log prior + log likelihood).

        Parameters:
        - theta: array-like, model parameters.
        - S_eq_func: callable, optional equilibrium sea level function.

        Returns:
        - lp + ll: float, total log probability.
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, S_eq_func)

    def run_mcmc(self, initial_guess, nwalkers=32, nsteps=5000, S_eq_func=None):
        """
        Run MCMC sampling to estimate the posterior distribution of the parameters.

        Parameters:
        - initial_guess: array-like, initial guess of the parameters.
        - nwalkers: int, number of MCMC walkers.
        - nsteps: int, number of steps in MCMC chain.
        - S_eq_func: callable, optional equilibrium sea level function.

        Returns:
        - sampler: emcee.EnsembleSampler object.
        - samples: 2D array, sampled parameter values.
        """
        ndim = len(initial_guess)
        # Initialize walkers around the initial guess
        pos = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)
        
        # Set up the sampler with partial function
        def log_prob_fn(theta):
            return self.log_probability(theta, S_eq_func)
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn)
        
        # Run MCMC
        try:
            sampler.run_mcmc(pos, nsteps, progress=True)
        except:
            print('MCMC run failed. continuing')
            pass
        
        # Flatten the chain and discard burn-in samples
        samples = sampler.get_chain(discard=int(nsteps/2), thin=15, flat=True)
        return sampler, samples

    def plot_model(self, theta, S_eq_func=None):
        """
        Plot the observed sea level data and the model prediction over the full time series.

        Parameters:
        - theta: array-like, model parameters.
        - S_eq_func: callable, optional equilibrium sea level function.
        """
        S_model_full = self.get_S(self.F_full, *theta, S_eq_func=S_eq_func)
        plt.figure(figsize=(12, 6))
        # Plot modeled sea level over full time range
        plt.plot(self.years_full, S_model_full, label='Modeled Sea Level', color='red')
        # Plot observed sea level data
        # plt.errorbar(self.years_obs, self.SL_obs, yerr=self.SL_unc,
        #              fmt='o', label='Observed Sea Level', color='black')
        plt.fill_between(self.years_obs, self.SL_obs - self.SL_unc, self.SL_obs + self.SL_unc, color='black', alpha=0.3)
        plt.xlabel('Year')
        plt.ylabel('Sea Level Anomaly (mm)')
        plt.title('Sea Level Model vs Observations')
        plt.legend()
        plt.show()

    def plot_model_with_ci(self, samples, confidence=95, S_eq_func=None):
        """
        Plot the observed sea level data and the model prediction with confidence intervals over the full time series.

        Parameters:
        - samples: array-like, MCMC samples of parameters.
        - confidence: float, confidence level for intervals.
        - S_eq_func: callable, optional equilibrium sea level function.
        """
        percentiles = [(100 - confidence) / 2, 50, 100 - (100 - confidence) / 2]
        S_models = np.array([self.get_S(self.F_full, *theta, S_eq_func=S_eq_func) for theta in samples])
        perc = np.percentile(S_models, [2.5, 50, 97.5], axis=0)

        plt.figure(figsize=(12, 6))
        # Plot observed sea level data
        # plt.errorbar(self.years_obs, self.SL_obs , yerr=self.SL_unc,
        #              fmt='o', label='Observed Sea Level', color='black')
        plt.fill_between(self.years_obs, self.SL_obs - self.SL_unc, self.SL_obs + self.SL_unc, color='black', alpha=0.3)
        # Plot median modeled sea level
        plt.plot(self.years_full, perc[1], label='Median Modeled Sea Level', color='red')
        # Plot confidence interval
        plt.fill_between(self.years_full, perc[0], perc[2],
                         color='red', alpha=0.3, label=f'{confidence}% Confidence Interval')
        plt.xlabel('Year')
        plt.ylabel('Sea Level Anomaly (mm)')
        plt.title('Sea Level Model with Confidence Intervals')
        plt.legend()
        plt.show()

    def compute_aic_bic(self, theta, S_eq_func=None):
        """
        Compute AIC and BIC for the model.

        Parameters:
        - theta: array-like, model parameters.
        - S_eq_func: callable, optional equilibrium sea level function.

        Returns:
        - aic: float, Akaike Information Criterion.
        - bic: float, Bayesian Information Criterion.
        """
        ll = self.log_likelihood(theta, S_eq_func)
        k = len(theta)  # Number of parameters
        n = len(self.SL_obs)  # Number of observations

        aic = 2 * k - 2 * ll
        bic = np.log(n) * k - 2 * ll

        return aic, bic


if __name__ == '__main__':

    # Create an instance of the model
    model = SeaLevelModel('../data/forcing_all.csv', '../data/CSIRO_Recons_gmsl_yr_2011.csv')

    # Plot the data
    model.plot_data()

    # Fit the model using curve_fit
    popt, pcov = model.curve_fit_model()
    print('Optimized parameters from curve_fit:', popt)

    # Plot the model prediction with optimized parameters over full time series
    model.plot_model(popt)

    # Compute AIC and BIC
    aic, bic = model.compute_aic_bic(popt)
    print(f"AIC: {aic}, BIC: {bic}")

    # Run MCMC sampling
    initial_guess = [-800, 600, 5, .2]
    sampler, samples = model.run_mcmc(initial_guess, nwalkers=50, nsteps=1000)

    # Plot the corner plot of the posterior distributions
    corner.corner(samples, labels=["S_0", "tau2", "a2", "b2"], truths=popt)
    plt.show()

    # Compute the median of the samples to get the best-fit parameters
    theta_mcmc = np.median(samples, axis=0)
    print('Optimized parameters from MCMC:', theta_mcmc)

    # Plot the model prediction with MCMC parameters over full time series
    model.plot_model(theta_mcmc)

    # Plot model with confidence intervals over full time series
    model.plot_model_with_ci(samples)

    # pretty print the results
    print("MCMC Results:")
    for i, param in enumerate(["S_0", "tau2", "a2", "b2"]):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = f"{param} = {mcmc[1]:.2f} + {q[1]:.2f} - {q[0]:.2f}"
        print(txt)
