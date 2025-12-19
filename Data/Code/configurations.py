# This code is based on https://github.com/AishSweety/hybrid-quantum-classical-Neural-Network-for-radar-data
# and adjusted for the experiments in the paper:
# "Hybrid Quantum Neural Networks with Amplitude Encoding: Advancing Recovery Rate Predictions".

# This file holds all the configuration dictionaries that are used for the sim simulation


# Speed of light in meters per second
c = 2.998e8  # [m/s]

# ---------------------------------------------------------------------
# Scenario: X band radar
# ---------------------------------------------------------------------
scenarioXband = {
    # Wavelength for X band (≈ 3 cm)
    "lamb": 0.02998,                 # [m]

    # Carrier frequency corresponding to the given wavelength
    # f_c = c / λ
    "f_c": c / 0.02998,              # [Hz]

    # Default signal-to-noise ratio in dB (can be overwritten elsewhere)
    "SNR": 0,
}

# ---------------------------------------------------------------------
# Platform: DJI Mavic Air 2
# ---------------------------------------------------------------------
djimavicair2 = {
    "name":  "DJI_Mavic_Air_2",

    # Number of blades per rotor
    "N":     2,

    # Blade root radius (inner radius), approximate
    "L_1":   0.005,                  # [m]

    # Blade tip radius (outer radius), approximate
    "L_2":   0.07,                   # [m]

    # Rotor rotation frequency in Hz (revolutions per second)
    # Source: https://mavicpilots.com/threads/mavic-average-rpm.4982/
    "f_rot": 91.66,                  # [Hz]
}

# ---------------------------------------------------------------------
# Platform: DJI Matrice 300 RTK
# ---------------------------------------------------------------------
djimatrice300rtk = {
    # Product reference link for propeller specs:
    # https://www.bhphotovideo.com/c/product/1565975-REG/dji_cp_en_00000270_01_matrice_300_series_propeller.html
    "name":  "DJI_Matrice_300_RTK",

    # Number of blades per rotor
    "N":     2,

    # Blade root radius (inner radius), approximate
    "L_1":   0.05,                   # [m]

    # Blade tip radius (outer radius), approximate
    "L_2":   0.2665,                 # [m]

    # Rotor rotation frequency in Hz (revolutions per second)
    # This is currently an estimate/guess.
    "f_rot": 70,                     # [Hz]
}
