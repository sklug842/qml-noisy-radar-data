"""
Optimize_Models.py

Driver script for training and evaluating hybrid quantum–classical models and
matched classical baselines on radar data. The script can be run from the
command line and supports:

- Quantum models with different QVC architectures and encodings
- Classical neural networks matched in depth/capacity
- Two optimizers:
    * Hashemi: gradient-free evolutionary optimizer
    * ADAM:    gradient-based optimizer

For each configuration, multiple independent runs are executed, and accuracy
statistics across epochs and runs are reported.
"""

import os
import sys
from   datetime                              import datetime

import numpy                                 as     np
import tensorflow                            as     tf

from   help_classes.wrapper_hybrid_qml_model import Wrapper_Hybrid_QML_Model
from   optimizers.hashemi                    import Hashemi
from   optimizers.adam                       import ADAM


# ---------------------------------------------------------------------------
# TensorFlow / GPU configuration
# ---------------------------------------------------------------------------

# Silence verbose TensorFlow logging (errors only).
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Select GPU via command-line argument (14th argument, index 13).
# -1 is used to indicate "no GPU" (no CUDA device visible to this process).
os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[13])

# Ensure memory growth is enabled on all visible GPUs, to avoid pre-allocating
# all GPU memory at once and improve co-existence with other processes.
physical_gpus = tf.config.list_physical_devices("GPU")
if physical_gpus:
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Now it is safe to list logical devices; useful for debugging/logging.
logical_gpus = tf.config.list_logical_devices("GPU")
print("Logical GPUs:", logical_gpus)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
# Expected arguments (all required):
#
#   1  shape       : int
#                    Input dimension after pooling (e.g. 6 or 10).
#
#   2  type_QVC    : str
#                    Quantum variational circuit architecture:
#                      - "V1"
#                      - "V2"
#
#   3  n_qubits    : int
#                    Number of qubits in the QVC (e.g. 3, 4, 5).
#
#   4  State_Prep  : str
#                    State-preparation / encoding scheme:
#                      - "Angle"
#                      - "Amplitude"
#
#   5  flag_FT     : str
#                    Preprocessing / transform applied to classical input:
#                      - "DFT" (discrete Fourier transform)
#                      - "QFT" (quantum Fourier transform inside the circuit)
#
#   6  flag_NN     : int
#                    Model type:
#                      - 0 : quantum model (QVC)
#                      - 1 : purely classical neural network baseline
#
#   7  optimizer   : str
#                    Optimization scheme:
#                      - "Hashemi" (gradient-free genetic algorithm)
#                      - "Adam"    (gradient-based Adam optimizer)
#
#   8  sigma_init  : float
#                    Initial standard deviation for the Hashemi optimizer.
#                    Only used if optimizer == "Hashemi".
#
#   9  sigma_gs    : float
#                    Mutation noise standard deviation for the Hashemi optimizer.
#                    Only used if optimizer == "Hashemi".
#
#  10  eta         : float
#                    Learning rate for Adam / SGD.
#                    Only used if optimizer in {"Adam", "SGD"}.
#
#  11  noise       : str
#                    Dataset / noise type:
#                      - ""        : noiseless dataset
#                      - "awgn"    : additive white Gaussian noise (AWGN)
#                      - "car1gn"  : complex AR(1) colored Gaussian noise
#
#  12  db          : int
#                    SNR level in dB (e.g. -1, 2, ...), used in dataset naming.
#
#  13  GPU         : int
#                    GPU ID for CUDA_VISIBLE_DEVICES; use -1 for CPU-only.
#
# The 14th argument (index 13) is already consumed above for GPU selection.
# ---------------------------------------------------------------------------

shape      = np.int64(  sys.argv[ 1])
type_QVC   = str(       sys.argv[ 2])
n_qubits   = np.int64(  sys.argv[ 3])
State_Prep = str(       sys.argv[ 4])
flag_FT    = str(       sys.argv[ 5])
flag_NN    = np.int64(  sys.argv[ 6])
optimizer  = str(       sys.argv[ 7])
sigma_init = np.float64(sys.argv[ 8])
sigma_gs   = np.float64(sys.argv[ 9])
eta        = np.float64(sys.argv[10])
noise      = str(       sys.argv[11])
db         = np.int64(  sys.argv[12])


# ---------------------------------------------------------------------------
# Example calls (for documentation / reproducibility)
# ---------------------------------------------------------------------------
#
# Quantum machine-learning models
# -------------------------------
# Optimizer choices:
#   - "Hashemi" (gradient-free)
#       * sigma_init ∈ {0.25, 0.50, 1.00}
#       * sigma_gs   ∈ {0.25, 0.50, 1.00}
#   - "Adam" (gradient-based)
#       * eta        ∈ {0.001, 0.01, 0.1, 0.25, 0.50}
#
# Datasets:
#   - noise == ""       : noiseless
#   - noise == "awgn"   : AWGN
#   - noise == "car1gn" : complex AR(1) colored Gaussian noise
#
# GPU:
#   - GPU == -1 : CPU-only (no CUDA device visible)
#   - GPU >= 0  : ID of CUDA device to use
#
# 3-qubit models (6-bin scenario, shape = 6)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# V1, Angle encoding, DFT:
#   python3 Optimize_Models.py 6 "V1" 3 "Angle"     "DFT" 0 optimizer sigma_init sigma_gs eta dataset noise gpu
#
# V2, Angle encoding, DFT:
#   python3 Optimize_Models.py 6 "V2" 3 "Angle"     "DFT" 0 optimizer sigma_init sigma_gs eta dataset noise gpu
#
# V1, Amplitude encoding, DFT:
#   python3 Optimize_Models.py 6 "V1" 3 "Amplitude" "DFT" 0 optimizer sigma_init sigma_gs eta dataset noise gpu
#
# V2, Amplitude encoding, DFT:
#   python3 Optimize_Models.py 6 "V2" 3 "Amplitude" "DFT" 0 optimizer sigma_init sigma_gs eta dataset noise gpu
#
# V1, Amplitude encoding, QFT:
#   python3 Optimize_Models.py 6 "V1" 3 "Amplitude" "QFT" 0 optimizer sigma_init sigma_gs eta dataset noise gpu
#
# V2, Amplitude encoding, QFT:
#   python3 Optimize_Models.py 6 "V2" 3 "Amplitude" "QFT" 0 optimizer sigma_init sigma_gs eta dataset noise gpu
#
# Example (noiseless, no GPU, Adam):
#   python3 Optimize_Models.py 6 "V1" 3 "Angle" "DFT" 0 "Adam" 0 0 0.001 "" -1 -1
#
#
# 4-qubit models (10-bin scenario, shape = 10)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# V1, Amplitude encoding, DFT:
#   python3 Optimize_Models.py 10 "V1" 4 "Amplitude" "DFT" 0 optimizer sigma_init sigma_gs eta dataset noise gpu
# ...
#
# 5-qubit models (10-bin scenario, shape = 10)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# V1, Angle encoding, DFT:
#   python3 Optimize_Models.py 10 "V1" 5 "Angle"     "DFT" 0 optimizer sigma_init sigma_gs eta dataset noise gpu
# ...
#
#
# Classical baselines matched to QML models
# -----------------------------------------
# NN matched to 3-qubit QML (6-bin):
#   python3 Optimize_Models.py 6  "" 3 "" "DFT" 1 "Adam" 0 0 eta dataset noise gpu
#
# NN matched to 5-qubit QML (10-bin):
#   python3 Optimize_Models.py 10 "" 5 "" "DFT" 1 "Adam" 0 0 eta dataset noise gpu
#
# Example (AWGN, GPU 0):
#   python3 Optimize_Models.py 6 "" 3 "" "DFT" 1 "Adam" 0 0 0.01 "awgn" 2 0
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Hyperparameters that are fixed for all runs in this script
# ---------------------------------------------------------------------------

n_runs   = 10    # number of independent restarts per configuration
n_layers = 2     # number of QVC layers (fixed in the experiments)


# ---------------------------------------------------------------------------
# Estimate number of QVC evaluations per epoch (for reporting)
# ---------------------------------------------------------------------------

if flag_NN == 1:
    # Purely classical model: QVC is not used, gradient cost is not relevant.
    n_evals_per_epoch = -1

else:
    # Quantum model: estimate the number of circuit evaluations per epoch,
    # assuming parameter-shift rule and explicit encoding derivatives.
    if State_Prep == "Angle":
        # Angle encoding: each qubit gets two angles (RZ, RY) per layer.
        n_inputs          = 2 * n_qubits
        # 2 evaluations per input parameter per layer.
        n_evals_encoding  = 2 * n_inputs * n_layers

    elif State_Prep == "Amplitude":
        # Amplitude encoding: dimension is 2^n_qubits.
        n_inputs          = 2 ** n_qubits
        # Encoding only in first layer, hence no factor n_layers here.
        n_evals_encoding  = 2 * n_inputs

    # Parameter count per layer depends on the QVC template.
    if type_QVC == "V1":
        # V1: simple layer with 2 parameters per qubit (e.g., RZ + RY).
        n_params_per_layer = 2 * n_qubits

    elif type_QVC == "V2":
        # V2: more expressive template; the formula encodes
        # (single-qubit rotations + entangling operations + final layer)
        # as used in the implemented ansatz.
        n_params_per_layer  = (
            (n_qubits * 2 + (n_qubits ** 2 - n_qubits) + n_qubits * 2) * n_layers
            - n_qubits
        )

    # 1 forward evaluation + 2 per parameter (parameter-shift) + encoding cost.
    n_evals_per_epoch = 1 + 2 * n_params_per_layer * n_layers + n_evals_encoding


# ---------------------------------------------------------------------------
# Log configuration
# ---------------------------------------------------------------------------

print(" ")
print("------------------------------------------")
print(f"shape:                  {shape            }")
print(f"noise:                  {noise            }")
print(f"db:                     {db               }")
print(f"Classical (flag_NN):    {flag_NN          }")
print(f"Type of QVC:            {type_QVC         }")
print(f"flag_FT:                {flag_FT          }")
print(f"n_layers:               {n_layers         }")
print(f"n_qubits:               {n_qubits         }")
print(f"State_Prep:             {State_Prep       }")
print(f"n_evals_per_epoch:      {n_evals_per_epoch}")

if optimizer == "Hashemi":
    print(f"sigma_init:             {sigma_init   }")
    print(f"sigma_gs:               {sigma_gs     }")

elif optimizer in ["Adam", "SGD"]:
    print(f"eta:                    {eta          }")

print(f"# Runs per combination: {n_runs           }")
print(" ")


# ---------------------------------------------------------------------------
# Main training loop: multiple independent runs
# ---------------------------------------------------------------------------

start      = datetime.now()
n_epochs   = 20

# accuracies has shape (n_epochs, n_runs)
accuracies = np.zeros((n_epochs, n_runs))

for run in range(n_runs):
    # Instantiate a fresh wrapper (data + model builder) for each run to ensure
    # independent initialization.
    myWrapper = Wrapper_Hybrid_QML_Model(
        shape       = shape     ,
        noise       = noise     ,
        db          = db        ,
        type_QVC    = type_QVC  ,
        n_qubits    = n_qubits  ,
        n_layers    = n_layers  ,
        n_shots     = 10000     ,
        max_threads = 50        ,
        flag_NN     = flag_NN   ,
        flag_FT     = flag_FT   ,
        State_Prep  = State_Prep,
    )

    # Select optimizer and run training.
    if optimizer == "Hashemi":
        accuracies[:, run] = Hashemi(
            wrapper           = myWrapper        ,
            n_evals_per_epoch = n_evals_per_epoch,
            n_epochs          = n_epochs         ,
            sigma_init        = sigma_init       ,
            sigma_gs          = sigma_gs         ,
        ).run()

    elif optimizer == "Adam":
        accuracies[:, run] = ADAM(
            wrapper  = myWrapper,
            n_epochs = n_epochs ,
            eta      = eta      ,
        ).run()

    print(f"accuracies run {run}: ", np.round(accuracies[:, run], 2))


# ---------------------------------------------------------------------------
# Aggregate statistics across runs
# ---------------------------------------------------------------------------

accuracies_mean     = np.mean(accuracies, axis=1)
accuracies_std      = np.std(accuracies, axis=1)

# Adjusted accuracy: mean minus 3 standard deviations, used to select a
# conservative "best" epoch across restarts.
accuracies_adjusted = accuracies_mean - 3 * accuracies_std
best_n_epoch        = np.argmax(accuracies_adjusted)

end                 = datetime.now()


# Per-epoch reporting
for i in range(n_epochs):
    print("\n\nEpoch:             ", i + 1             )
    print("Accuracy:          ", accuracies[         i])
    print("Accuracy Mean:     ", accuracies_mean[    i])
    print("Accuracy Std:      ", accuracies_std[     i])
    print("Accuracy Adjusted: ", accuracies_adjusted[i])

# Summary for the best epoch under the adjusted metric
print("\nBest Epoch:        ",                   best_n_epoch + 1)
print("Accuracy:          ", accuracies[         best_n_epoch]   )
print("Accuracy Mean:     ", accuracies_mean[    best_n_epoch]   )
print("Accuracy Std:      ", accuracies_std[     best_n_epoch]   )
print("Accuracy Adjusted: ", accuracies_adjusted[best_n_epoch]   )


# ---------------------------------------------------------------------------
# Final configuration and timing summary
# ---------------------------------------------------------------------------

print(" ")
print("-----------------------------------------------")
print(f"shape:                  {shape            }")
print(f"noise:                  {noise            }")
print(f"db:                     {db               }")
print(f"Classical (flag_NN):    {flag_NN          }")
print(f"Type of QVC:            {type_QVC         }")
print(f"flag_FT:                {flag_FT          }")
print(f"n_layers:               {n_layers         }")
print(f"n_qubits:               {n_qubits         }")
print(f"State_Prep:             {State_Prep       }")
print(f"n_evals_per_epoch:      {n_evals_per_epoch}")

if optimizer == "Hashemi":
    print(f"sigma_init:             {sigma_init   }")
    print(f"sigma_gs:               {sigma_gs     }")

elif optimizer in ["Adam", "SGD"]:
    print(f"eta:                    {eta          }")
    print(f"n_epochs:               {n_epochs     }")

print(f"# Runs per combination: {n_runs           }")
print(f"Start:                  {start            }")
print(f"End:                    {end              }")
print(f"Duration:               {end - start      }")
print("-----------------------------------------------")
print(" ")