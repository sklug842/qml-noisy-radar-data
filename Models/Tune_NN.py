"""
Tune_NN.py

Hyperparameter tuning of classical neural network baselines using Optuna.

This script optimizes:
- Learning rate (eta) for the Adam optimizer
- Depth (number of hidden layers)
- Width (number of units per hidden layer)

for a purely classical neural network that is implemented via the
Wrapper_Hybrid_QML_Model with `flag_NN=True` and `flag_NN_optuna=True`.

For each candidate architecture, multiple independent runs are executed; the
objective value is the best "adjusted" accuracy across epochs:

    adjusted_accuracy(epoch) = mean(epoch) - 3 * std(epoch)  (across runs)

which yields a conservative epoch selection with respect to variance.
"""

import os
import sys

import numpy as np
import optuna
import tensorflow as tf

from help_classes.wrapper_hybrid_qml_model import Wrapper_Hybrid_QML_Model
from optimizers.adam import ADAM


# ---------------------------------------------------------------------------
# TensorFlow / GPU configuration
# ---------------------------------------------------------------------------

# Silence verbose TensorFlow logging (errors only).
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Select GPU via command-line argument (4th argument, index 3).
# -1 is used to indicate "no GPU" (no CUDA device visible to this process).
os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[4])

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
# Training helper: build wrapper and run ADAM for a single architecture
# ---------------------------------------------------------------------------

def agent_train(eta, NN_units, n_epochs, shape, noise, db, q=None):
    """
    Train a purely classical neural network with a given architecture and
    learning rate, using the ADAM optimizer.

    Parameters
    ----------
    eta : float
        Learning rate for the Adam optimizer.
    NN_units : list[int]
        List of hidden-layer widths; length = number of hidden layers.
    n_epochs : int
        Number of epochs for ADAM.run().
    shape : int
        Input dimension after pooling (e.g. 6 or 10).
    noise : str
        Noise type in the dataset:
          - ""       : noiseless
          - "awgn"   : additive white Gaussian noise
          - "car1gn" : complex AR(1) colored Gaussian noise
    db : int
        SNR level in dB.

    Returns
    -------
    np.ndarray
        Accuracy per epoch (shape: (n_epochs,)), as returned by ADAM.run().
    """

    # Fixed data and architecture settings for this tuning script.
    flag_FT    = "DFT"   # use DFT-based preprocessing (FFT + magnitude)
    flag_NN    = True    # purely classical model (no QVC)

    # QVC-related arguments are still required by the wrapper, but are not
    # used for the classical baseline (type_QVC == "", flag_NN == True).
    type_QVC   = ""
    n_qubits   = 3
    n_layers   = 2
    State_Prep = ""

    # Instantiate the wrapper for this architecture.
    myWrapper = Wrapper_Hybrid_QML_Model(
        shape          = shape,
        noise          = noise,
        db             = db,
        type_QVC       = type_QVC,
        n_qubits       = n_qubits,
        n_layers       = n_layers,
        max_threads    = 50,
        flag_NN        = flag_NN,
        flag_NN_optuna = True,
        NN_units       = NN_units,
        flag_FT        = flag_FT,
        State_Prep     = State_Prep,
    )

    # Run gradient-based optimization with Adam; returns accuracy per epoch.
    result = ADAM(
        wrapper        = myWrapper,
        n_epochs       = n_epochs , 
        eta            = eta      ,
    ).run()

    return result


# ---------------------------------------------------------------------------
# Optuna objective: define search space and evaluation protocol
# ---------------------------------------------------------------------------

def objective(trial, NUM_RUNS, n_epochs, shape, noise, db):
    """
    Optuna objective function.

    For each trial, this function:
    - Samples a learning rate (eta) and a hidden-layer architecture (NN_units)
    - Trains the model NUM_RUNS times with independent initializations
    - Aggregates per-epoch accuracies across runs
    - Selects the epoch with maximum adjusted accuracy:
         adjusted = mean - 3 * std
    - Stores diagnostic information in the trial attributes
    - Returns the best adjusted accuracy as the objective value.
    """

    # Learning rate search space (Adam).
    eta = trial.suggest_categorical(
        "learning_rate", [5e-1, 2.5e-1, 1e-1, 1e-2, 1e-3, 5e-4]
    )

    # Number of hidden layers: integer in [2, 7].
    num_layers = trial.suggest_discrete_uniform(
        "Number layers policy", 2, 7, 1
    )

    # Width (units) per hidden layer: each in {5, 10, ..., 50}.
    NN_units = []
    for n in range(int(num_layers)):
        units_n = trial.suggest_discrete_uniform(
            f"Number units layer {n} policy", 5, 50, 5
        )
        NN_units.append(int(units_n))

    # accuracies has shape (n_epochs, NUM_RUNS)
    accuracies = np.zeros((n_epochs, NUM_RUNS))

    # Multiple independent runs per architecture to average out randomness.
    for run in range(NUM_RUNS):
        accuracies[:, run] = agent_train(
            eta      = eta     ,
            NN_units = NN_units,
            n_epochs = n_epochs,
            shape    = shape   ,
            noise    = noise   ,
            db       = db      ,
        )

    # Aggregate statistics across runs for each epoch.
    accuracies_mean     = np.mean(accuracies, axis=1)
    accuracies_std      = np.std( accuracies, axis=1)

    # Conservative accuracy metric for epoch selection.
    accuracies_adjusted = accuracies_mean - 3 * accuracies_std
    best_n_epoch        = np.argmax(accuracies_adjusted)

    # Store diagnostic information for later analysis.
    trial.set_user_attr("Best_epoch_values", str(accuracies[best_n_epoch]))
    trial.set_user_attr("Best Epoch"       , int(best_n_epoch) + 1        )

    trial.set_user_attr("accuracies_mean"  , accuracies_mean[best_n_epoch])
    trial.set_user_attr("accuracies_std"   , accuracies_std[ best_n_epoch])
    trial.set_user_attr(
        "accuracies_adjusted", accuracies_adjusted[best_n_epoch]
    )

    trial.set_user_attr("nn_units", str(NN_units))

    # Print summary for this trial.
    print("\nBestEpoch:         ",                   best_n_epoch + 1)
    print("Accuracy:          ", accuracies[         best_n_epoch])
    print("Accuracy Mean:     ", accuracies_mean[    best_n_epoch])
    print("Accuracy Std:      ", accuracies_std[     best_n_epoch])
    print("Accuracy Adjusted: ", accuracies_adjusted[best_n_epoch])

    # Optuna maximizes this value.
    return np.max(accuracies_adjusted)


# ---------------------------------------------------------------------------
# Main: create study, run optimization, and log summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Number of independent runs per architecture (robustness to initialization).
    NUM_RUNS = 10

    # Number of Optuna trials (candidate architectures).
    NUM_TRIALS = 150

    # Command-line arguments:
    #
    #   1  shape : int
    #             Input dimension after pooling (e.g. 6 or 10).
    #
    #   2  noise : str
    #             Dataset / noise type:
    #               - ""       : noiseless
    #               - "awgn"   : AWGN
    #               - "car1gn" : complex AR(1) colored Gaussian noise
    #
    #   3  db    : int
    #             SNR level in dB (e.g. -1, 2, ...).
    #
    #   4  GPU   : int
    #             GPU ID for CUDA_VISIBLE_DEVICES; use -1 for CPU-only.
    #
    shape    = np.int64(sys.argv[1])
    n_epochs = 20
    noise    = str(sys.argv[2])
    db       = np.int64(sys.argv[3])

    if noise == "":
        noise_study = "noiseless"
    else:
        noise_study = noise

    # Each study is stored in its own SQLite database file for reproducibility.
    study_name = f"tune_NN_{shape}_{noise_study}_{db}"

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=f"sqlite:///Optuna/{study_name}.db",
        load_if_exists=True,
    )

    # Launch the optimization.
    study.optimize(
        lambda trial: objective(
            trial, NUM_RUNS, n_epochs, shape, noise, db
        ),
        n_trials=NUM_TRIALS,
    )

    print("Number of finished trials: {}".format(len(study.trials)))


# ---------------------------------------------------------------------------
# Example calls (for documentation / reproducibility)
# ---------------------------------------------------------------------------
#
# GPU selection:
#   - GPU == -1 : CPU-only (no CUDA device visible)
#   - GPU >= 0  : ID of CUDA device to use
#
# 6-bin scenario (shape = 6)
# --------------------------
# python3 Tune_NN.py 6 ""       -1  gpu
# python3 Tune_NN.py 6 "awgn"    2  gpu
# python3 Tune_NN.py 6 "car1gn"  2  gpu
#
# Example:
#   python3 Tune_NN.py 6 "" -1 0
#
# 10-bin scenario (shape = 10)
# ----------------------------
# python3 Tune_NN.py 10 ""       -1  gpu
# python3 Tune_NN.py 10 "awgn"    2  gpu
# python3 Tune_NN.py 10 "car1gn"  2  gpu
#
# Example:
#   python3 Tune_NN.py 10 "car1gn" 2 0
# ---------------------------------------------------------------------------