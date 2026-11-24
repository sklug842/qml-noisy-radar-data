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
# Number of QVC evaluations per optimization step
# ---------------------------------------------------------------------------

if flag_NN == 1:
    # Purely classical model: no QVC evaluation budget is required.
    n_evals_per_step = -1

else:
    # Number of trainable QVC parameters P.
    if type_QVC == "V1":
        n_params = 2 * n_qubits * n_layers

    elif type_QVC == "V2":
        n_params = (
            (n_qubits * 2 + (n_qubits ** 2 - n_qubits) + n_qubits * 2)
            * n_layers
            - n_qubits
        )

    # One forward evaluation plus two parameter-shift evaluations
    # for each trainable QVC parameter.
    n_evals_per_step = 1 + 2 * n_params


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
print(f"n_evals_per_step:       {n_evals_per_step }")

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
accuracies_val  = np.zeros((n_epochs, n_runs))
accuracies_test = np.zeros((n_epochs, n_runs))

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
        accuracies_val[:, run], accuracies_test[:, run] = Hashemi(
            wrapper           = myWrapper        ,
            n_evals_per_epoch = n_evals_per_step ,
            n_epochs          = n_epochs         ,
            sigma_init        = sigma_init       ,
            sigma_gs          = sigma_gs         ,
        ).run()

    elif optimizer == "Adam":
        accuracies_val[:, run], accuracies_test[:, run] = ADAM(
            wrapper  = myWrapper,
            n_epochs = n_epochs ,
            eta      = eta      ,
        ).run()

    print(f"accuracies run {run}: ", np.round(accuracies_val[:, run], 2),  np.round(accuracies_test[:, run], 2))


# ---------------------------------------------------------------------------
# Aggregate statistics across runs
# ---------------------------------------------------------------------------

accuracies_val_mean  = np.mean(accuracies_val, axis=1)
accuracies_val_std   = np.std(accuracies_val, axis=1)

accuracies_test_mean = np.mean(accuracies_test, axis=1)
accuracies_test_std  = np.std(accuracies_test, axis=1)

# Adjusted validation accuracy: mean minus 3 standard deviations.
# The best epoch is selected exclusively from validation performance.
accuracies_val_adjusted = accuracies_val_mean - 3 * accuracies_val_std
best_n_epoch            = np.argmax(accuracies_val_adjusted)

end = datetime.now()


# ---------------------------------------------------------------------------
# Per-epoch reporting
# ---------------------------------------------------------------------------

for i in range(n_epochs):
    print("\n\nEpoch:                    ", i + 1)

    print("\nValidation")
    print("Accuracy:                  ", accuracies_val[i])
    print("Accuracy Mean:             ", accuracies_val_mean[i])
    print("Accuracy Std:              ", accuracies_val_std[i])
    print("Accuracy Adjusted:         ", accuracies_val_adjusted[i])

    print("\nTest")
    print("Accuracy:                  ", accuracies_test[i])
    print("Accuracy Mean:             ", accuracies_test_mean[i])
    print("Accuracy Std:              ", accuracies_test_std[i])


# ---------------------------------------------------------------------------
# Summary at validation-selected best epoch
# ---------------------------------------------------------------------------

print("\n\nBest Epoch (Validation):   ", best_n_epoch + 1)

print("\nValidation")
print("Accuracy:                  ", accuracies_val[best_n_epoch])
print("Accuracy Mean:             ", accuracies_val_mean[best_n_epoch])
print("Accuracy Std:              ", accuracies_val_std[best_n_epoch])
print("Accuracy Adjusted:         ", accuracies_val_adjusted[best_n_epoch])

print("\nTest")
print("Accuracy:                  ", accuracies_test[best_n_epoch])
print("Accuracy Mean:             ", accuracies_test_mean[best_n_epoch])
print("Accuracy Std:              ", accuracies_test_std[best_n_epoch])
