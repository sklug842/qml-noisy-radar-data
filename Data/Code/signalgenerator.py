# This code is based on https://github.com/AishSweety/hybrid-quantum-classical-Neural-Network-for-radar-data
# and adjusted for the experiments in the paper:
# "Hybrid Quantum Neural Networks with Amplitude Encoding: Advancing Recovery Rate Predictions".

import math
import cmath
import numpy as     np

# Imaginary unit j = sqrt(-1), conventional notation in signal processing
j = complex(0, 1)


def sinc(x):
    """
    Scalar sinc function: sin(x)/x with the removable discontinuity at x=0
    filled by continuity (value 1).

    Note
    ----
    This is different from np.sinc, which is normalized as sin(pi x)/(pi x).
    """
    if x == 0:
        return 1  # continuous extension at x = 0
    return math.sin(x) / x


def psigenerator(
    f_c,
    lamb,
    N,
    L_1,
    L_2,
    f_rot,
    phi_p=0,
    SNR=None,
    deterministic=False,
    **kwargs,
):
    """
    Generate a (possibly noisy) complex-valued psi(t) function representing
    the radar return from a rotating-blade drone.

    Parameters
    ----------
    f_c : float
        Carrier frequency [Hz].
    lamb : float
        Wavelength [m].
    N : int
        Number of blades per rotor.
    L_1 : float
        Inner blade radius [m].
    L_2 : float
        Outer blade radius [m].
    f_rot : float
        Rotor rotation frequency [Hz] (revolutions per second).
    phi_p : float, optional
        Phase offset parameter, used in alpha/beta terms.
    SNR : float or None, optional
        Signal-to-noise ratio in dB. If not None, additive
        complex Gaussian noise (per sample) is added based on SNR.
    deterministic : bool, optional
        If True, use fixed values for amplitude and geometry
        (A_r=1, R=0, theta=0, V_rad=0).
        If False, draw them randomly.
    **kwargs :
        Ignored here; included to allow dictionary expansion with extra keys.

    Returns
    -------
    callable
        A function psi(t) (or fuzzypsi(t) if noise is included) that maps
        time t (float) to a complex radar return value.
    """

    # ------------------------------------------------------------------
    # Draw random amplitude and geometric parameters unless deterministic
    # ------------------------------------------------------------------
    if deterministic:
        A_r   = 1  # deterministic amplitude
        R     = 0  # deterministic range [m]
        theta = 0  # aspect angle [rad]
        V_rad = 0  # radial velocity [m/s]
    else:
        # Amplitude A_r from a chi-square distribution with 4 dof
        A_r = np.random.chisquare(4)

        # Random range R in [1000, 5000] meters
        R = np.random.uniform(low=1000, high=5000)

        # Random angle theta in [0, pi/2] radians
        theta = np.random.uniform(low=0, high=np.pi / 2)

        # Radial velocity (currently fixed to 0)
        V_rad = 0

    # ------------------------------------------------------------------
    # Define the noiseless psi(t) function
    # ------------------------------------------------------------------
    def psi(t):
        """
        Compute the complex radar return psi(t) at time t.
        """

        def get_Omega_n(n):
            """
            Blade phase and Doppler term for blade index n.

            Omega_n : mechanical phase of blade n at time t
            gamma_n : Doppler/spatial frequency term for blade n
            """
            # Blade phase angle (includes time evolution and blade index offset)
            Omega_n = 2 * np.pi * (f_rot * t + n / N)

            # Gamma_n captures Doppler-like modulation from rotating blade geometry
            gamma_n = (4 * np.pi / lamb) * np.cos(theta) * np.sin(Omega_n)

            return Omega_n, gamma_n

        # ------------------------------------------------------------------
        # Compute alpha, beta terms (dependent on theta and phi_p)
        # ------------------------------------------------------------------
        sin_plus, sin_minus = np.abs(theta) + np.array([phi_p, -1 * phi_p])

        sin_plus  = np.sin(sin_plus)
        sin_minus = np.sin(sin_minus)

        alpha = sin_plus + sin_minus
        beta  = np.sign(theta) * (sin_plus - sin_minus)

        # ------------------------------------------------------------------
        # Blade geometry terms: mid-radius and half-span
        # ------------------------------------------------------------------
        L1_L2 = (L_1 + L_2) / 2    # mean radius
        L2_L1 = (L_2 - L_1) / 2    # half of blade span

        # Carrier term: propagation phase and range/velocity contribution
        prefactor = A_r * cmath.exp(
            j * (2 * math.pi * f_c * t - (4 * math.pi / lamb) * (R + V_rad * t))
        )
        # Alternative (commented) option to set prefactor=1 (pure micro-Doppler):
        # prefactor = 1

        # Accumulator for sum over blades
        accum = complex(0, 0)

        # ------------------------------------------------------------------
        # Sum contribution from each blade n = 0..N-1
        # ------------------------------------------------------------------
        for n in range(N):
            Omega_n, gamma_n = get_Omega_n(n)

            # Amplitude modulation term due to angle and phase
            prefactor_2 = alpha + beta * np.cos(Omega_n)

            # Spatial phase term for the blade along its radius
            exponential = np.exp(-1j * (L1_L2 * gamma_n))

            # sinc-like weighting term over the finite blade length
            # Note: np.sinc(x) = sin(pi x) / (pi x) (normalized sinc)
            sincterm = np.sinc(L2_L1 * gamma_n)

            # Accumulate blade contribution
            accum += prefactor_2 * exponential * sincterm

        # Final complex return
        return prefactor * accum

    # ----------------------------------------------------------------------
    # Optionally wrap psi(t) with additive complex Gaussian noise
    # ----------------------------------------------------------------------
    if SNR is not None:
        # Re-arrangement of:
        #    SNR_dB = 10 * log10(A_r^2 / variance)
        # -> variance = A_r^2 / 10^(SNR_dB/10)
        # which is equivalent to:
        #    variance = 10**(2*log10(A_r) - SNR/10)
        variance = 10 ** (2 * math.log10(A_r) - (SNR / 10))

        def fuzzypsi(t):
            """
            Noisy version of psi(t): additive complex Gaussian noise
            with variance derived from the requested SNR.
            """
            real_noise = np.random.normal(0, math.sqrt(variance))
            imag_noise = np.random.normal(0, math.sqrt(variance))
            return psi(t) + real_noise + imag_noise * j

        return fuzzypsi

    else:
        # No noise requested: return the clean psi(t)
        return psi


def generateData(psi, f_s, sample_length, offset=False):
    """
    Sample a continuous-time function psi(t) at rate f_s over a given duration.

    Parameters
    ----------
    psi : callable
        Function of time psi(t) returning complex values.
    f_s : float
        Sampling frequency in Hz.
    sample_length : float
        Duration of the sample in seconds.
    offset : bool, optional
        If True, shift the time axis by a random offset in [0, 1] seconds
        before sampling (simulates random start time within a window).

    Returns
    -------
    xs : list of float
        Sample times.
    real_ys : list of float
        Real parts of psi evaluated at the sample times.
    imaginary_ys : list of float
        Imaginary parts of psi evaluated at the sample times.
    """

    xs           = []
    real_ys      = []
    imaginary_ys = []
    offset_val   = 0.0

    # Optional random time offset in [0, 1) seconds
    if offset:
        offset_val = np.random.uniform(0, 1)

    # Number of discrete samples: floor(f_s * sample_length)
    num_samples = int(f_s * sample_length)

    for i in range(num_samples):
        # Base sampling time
        x = i / f_s
        xs.append(x)

        # Evaluate psi at shifted time (offset_val + x)
        val = psi(offset_val + x)

        # Store real and imaginary parts separately
        real_ys.append(val.real)
        imaginary_ys.append(val.imag)

    return xs, real_ys, imaginary_ys