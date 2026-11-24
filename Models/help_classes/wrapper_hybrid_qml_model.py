"""Hybrid quantum–classical model wrapper.

This module provides a high-level wrapper around:
  * PennyLane quantum variational circuits (QVCs) implemented as QNodes,
  * TensorFlow / Keras models for classical baselines,
  * Data loading and preprocessing pipelines,
  * Utilities for mapping between flat parameter vectors and model weights.

It supports multiple QVC architectures ("V1", "V2"), different state-preparation
schemes ("Mottonen", "Angle"), and optional neural-network baselines.
"""

import itertools
from multiprocessing import Pool

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input

import pennylane as qml
from pennylane import numpy as np

from help_classes.abstract_classes import AbstractWrapper
from help_classes.myquantumlayer import QuantumLayer

# Use float64 throughout to match PennyLane's default dtype
tf.keras.backend.set_floatx("float64")


class Wrapper_Hybrid_QML_Model(AbstractWrapper):
    """Wrapper for hybrid QML models and classical baselines.

    Responsibilities:
      * Load and preprocess radar datasets (optionally with noise and FFT).
      * Construct quantum models (QVC variants) or purely classical baselines.
      * Provide helper utilities for evolutionary / population-based optimizers:
          - flattening model weights into individuals,
          - mapping individuals back onto model weights,
          - evaluating BCE and accuracy.

    The main entry points used in experiments are:
      * `get_model()` - builds and compiles the Keras model.
      * `evaluate_single_final()` - evaluates a given individual on a dataset.
    """

    def __init__(
        self,
        max_threads,
        shape,
        type_QVC,
        n_qubits,
        State_Prep,
        flag_FT,
        flag_NN,
        flag_NN_optuna=False,
        NN_units=None,
        n_layers=2,
        noise="",
        db=-1,
        n_shots=10000,
        flag_summary=False,
        flag_draw=False,
    ) -> None:
        super().__init__()

        # Hardware / parallelism
        self.max_threads = max_threads
        self.n_shots     = n_shots

        # Data configuration
        self.noise          = noise
        self.db             = str(db)
        self.shape          = shape

        # Quantum variational circuit (QVC) configuration
        self.type_QVC       = type_QVC
        self.n_qubits       = n_qubits
        self.n_layers       = n_layers
        self.State_Prep     = State_Prep
        self.flag_FT        = flag_FT
        if State_Prep == "Angle":
            # QFT is only meaningful for amplitude encoding
            self.flag_FT   = "DFT"

        # Classical NN baseline configuration
        self.flag_NN        = flag_NN
        self.flag_NN_optuna = flag_NN_optuna
        self.NN_units       = [] if NN_units is None else NN_units

        # Misc. flags
        self.flag_summary   = flag_summary
        self.flag_draw      = flag_draw

        # Load data immediately
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.get_data()

        # Check if a GPU is available for PennyLane's lightning.gpu backend
        self.flag_GPU        = len(tf.config.list_physical_devices("GPU")) == 1

    def get_data(self):
        """Load and preprocess training, validation, and test data.

        Returns:
            (train_data, train_labels, val_data, val_labels,
             test_data, test_labels) where:
              * data arrays have shape (n_samples, shape),
              * labels are flattened binary labels.

        Processing steps:
          * Load pooled range profiles with/without injected noise.
          * Optionally apply an FFT for Angle encoding.
          * Take magnitude of complex-valued inputs.
          * For Mottonen state prep, normalize each sample to unit ℓ2 norm.
        """
        self.batchsize = 25

        # Determine file names depending on noise configuration
        if self.noise == "":
            train_data = np.load(
                f"../Data/Data/X_Train_900_pooled_mean_{self.shape}_OffsetFalse.npy"
            )
            val_data = np.load(
                f"../Data/Data/X_Validation_900_pooled_mean_{self.shape}_OffsetFalse.npy"
            )
            test_data = np.load(
                f"../Data/Data/X_Test_900_pooled_mean_{self.shape}_OffsetFalse.npy"
            )
        else:
            suffix = f"_{self.noise}_db{self.db}"
            train_data = np.load(
                f"../Data/Data/X_Train_900_pooled_mean_{self.shape}_OffsetFalse{suffix}.npy"
            )
            val_data = np.load(
                f"../Data/Data/X_Validation_900_pooled_mean_{self.shape}_OffsetFalse{suffix}.npy"
            )
            test_data = np.load(
                f"../Data/Data/X_Test_900_pooled_mean_{self.shape}_OffsetFalse{suffix}.npy"
            )

        train_labels = np.load(
            f"../Data/Data/y_Train_900_OffsetFalse.npy"
        ).flatten()
        val_labels = np.load(
            f"../Data/Data/y_Validation_900_OffsetFalse.npy"
        ).flatten()
        test_labels = np.load(
            f"../Data/Data/y_Test_900_OffsetFalse.npy"
        ).flatten()

        # Optional FFT on complex data for Angle encoding
        if self.flag_FT == "DFT" and self.State_Prep == "Angle":
            train_data = np.fft.fft(train_data)
            val_data   = np.fft.fft(val_data)
            test_data  = np.fft.fft(test_data)

        # Use magnitude of complex radar returns
        train_data = np.abs(train_data)
        val_data   = np.abs(val_data)
        test_data  = np.abs(test_data)

        # For Mottonen amplitude encoding, normalize each sample to unit norm
        if self.State_Prep == "Amplitude":
            train_data = train_data / np.linalg.norm(
                x=train_data, axis=1, keepdims=True
            )
            val_data = val_data / np.linalg.norm(
                x=val_data, axis=1, keepdims=True
            )
            test_data = test_data / np.linalg.norm(
                x=test_data, axis=1, keepdims=True
            )

        return train_data, train_labels, val_data, val_labels, test_data, test_labels



    def get_model(self):
        """Build and compile the hybrid model (quantum or classical baseline).

        Depending on configuration, this constructs either:
          * a quantum variational classifier (QVC) with:
              - amplitude encoding via Mottonen-style state preparation, or
              - angle encoding with per-qubit RZ/RY rotations,
            and one of two ansatz variants ("V1", "V2"); or
          * a purely classical fully connected network baseline (flag_NN=True).

        Returns:
            tf.keras.Model: compiled model with sigmoid output for binary
            classification.
        """

        # ------------------------------------------------------------------
        # 1. State preparation: Mottonen-style amplitude encoding (batched)
        # ------------------------------------------------------------------
        def State_Preparation(b):
            """Mottonen-style state preparation for a batch of real vectors.

            Args:
                b: array of shape (batch_size, dim). If dim < 2**n_qubits,
                   the vector is zero-padded to length 2**n_qubits.
            """

            # Ensure dimension is a power of 2 by zero-padding if needed
            if 2**self.n_qubits != self.shape:
                b = np.concatenate(
                    [b, np.zeros((len(b), 2**self.n_qubits - self.shape))],
                    axis=1,
                )

            _, total_dim = b.shape
            nb = int(np.log2(total_dim))

            # Split into sign and magnitude
            signs = np.sign(b)          # (batch_size, total_dim)
            b_abs = np.abs(b)           # (batch_size, total_dim)

            # Precompute binary patterns for indices 0..(2**nb - 1)
            all_bits = np.array(
                [list(map(int, np.binary_repr(j, width=nb)))
                 for j in range(total_dim)]
            )  # (total_dim, nb)

            # Top-level RY on wire 0: split full vector into two halves
            denom_full = np.sqrt(np.sum(b_abs**2, axis=1))  # (batch_size,)
            half = 2**(nb - 1)
            numerator = np.sqrt(np.sum(b_abs[:, half:total_dim]**2, axis=1))

            ratio0 = np.zeros_like(numerator)
            mask = denom_full != 0
            ratio0[mask] = numerator[mask] / denom_full[mask]
            beta0 = 2 * np.arcsin(ratio0)

            qml.RY(phi=beta0, wires=0)
            qml.Barrier()

            # Precompute angles β_s for each layer s and block j
            beta_all = {}
            for s in reversed(range(1, nb)):
                chunk_size = 2**s
                half_chunk = 2**(s - 1)
                num_blocks = total_dim // chunk_size

                B = b_abs.reshape(len(b), num_blocks, chunk_size)
                denom_s = np.sqrt(np.sum(B**2, axis=2))  # (batch_size, num_blocks)
                numer_s = np.sqrt(
                    np.sum(B[:, :, half_chunk:half_chunk + half_chunk]**2, axis=2)
                )

                ratio_s = np.zeros_like(numer_s)
                mask = denom_s != 0
                ratio_s[mask] = numer_s[mask] / denom_s[mask]
                beta_s = 2 * np.arcsin(ratio_s)

                beta_all[s] = beta_s

            # Controlled RY rotations for each layer s (wires = nb - s)
            for s in reversed(range(1, nb)):
                wire_index = nb - s
                ctrl_wires = list(range(wire_index))

                betas = beta_all[s]                  # (batch_size, num_blocks)
                num_blocks = betas.shape[1]
                ctrl_patterns_for_s = all_bits[:num_blocks, :wire_index]

                for j in range(num_blocks):
                    angle_vector = betas[:, j]
                    ctrl_vals = ctrl_patterns_for_s[j]

                    qml.ctrl(
                        op=qml.RY,
                        control_values=list(ctrl_vals),
                        control=ctrl_wires,
                    )(angle_vector, wires=wire_index)

            qml.Barrier()

            # Sign corrections on the last wire (wire = nb - 1)
            rot_Z = np.where(signs < 0, np.pi, 0.0)
            last_bits = all_bits[:, -1]
            mask_X = (signs < 0) & (last_bits[None, :] == 0)
            rot_X = np.where(mask_X, np.pi, 0.0)

            wire_target = nb - 1
            ctrl_patterns_final = all_bits[:, :wire_target]

            for i in range(total_dim):
                phiX = rot_X[:, i]
                phiZ = rot_Z[:, i]
                ctrl_vals_i = ctrl_patterns_final[i].tolist()

                qml.RX(phi=phiX, wires=wire_target)
                qml.ctrl(
                    op=qml.RZ,
                    control_values=ctrl_vals_i,
                    control=list(range(wire_target)),
                )(phiZ, wires=wire_target)
                qml.RX(phi=phiX, wires=wire_target)

        # ------------------------------------------------------------------
        # 2. Helper routines: encoding, single-qubit rotations, entanglers
        # ------------------------------------------------------------------
        def _hadamard_helper(n_qubits):
            """Apply Hadamards to all qubits."""
            for x in range(n_qubits):
                qml.Hadamard(wires=x)
            qml.Barrier()

        def _encoding_helper(n_qubits, inputs, i, State_Prep, flag_FT):
            """Encode classical data into the quantum state.

            For "Amplitude": amplitude encoding in the first layer, optionally
            followed by a QFT. For "Angle": per-qubit RZ/RY rotations.
            """
            if State_Prep == "Amplitude" and i == 0:
                State_Preparation(b=inputs)
                qml.Barrier()

                if flag_FT == "QFT":
                    qml.QFT(wires=range(self.n_qubits))
                    qml.Barrier()

            elif self.State_Prep == "Angle":
                for x in range(n_qubits):
                    qml.RZ(
                        phi=inputs[:, i * n_qubits * 2 + 2 * x + 0], wires=x
                    )
                    qml.RY(
                        phi=inputs[:, i * n_qubits * 2 + 2 * x + 1], wires=x
                    )
                qml.Barrier()

        def _rotations_helper(n_qubits, weights, i):
            """Single-qubit RZ/RY rotations for ansatz V1."""
            for x in range(self.n_qubits):
                qml.RZ(phi=weights[i * n_qubits * 2 + 2 * x + 0], wires=x)
                qml.RY(phi=weights[i * n_qubits * 2 + 2 * x + 1], wires=x)
            qml.Barrier()

        def _cnot_helper_V1(n_qubits):
            """Ring of CNOTs (x → x+1, with wraparound)."""
            for x in range(n_qubits):
                if x < n_qubits - 1:
                    qml.CNOT(wires=[x, x + 1])
                else:
                    qml.CNOT(wires=[x, 0])
            qml.Barrier()

        # ------------------------------------------------------------------
        # 3. QVC ansatz variants
        # ------------------------------------------------------------------
        def qnode_V1(inputs, weights):
            """Ansatz V1: encoding + local RZ/RY + ring CNOT entanglers."""
            weights = weights.flatten()

            for i in range(self.n_layers):
                if self.State_Prep == "Angle":
                    _hadamard_helper(self.n_qubits)

                _encoding_helper(self.n_qubits, inputs, i, self.State_Prep, self.flag_FT)
                _rotations_helper(self.n_qubits, weights, i)

                if i < self.n_layers - 1:
                    _cnot_helper_V1(self.n_qubits)

        def qnode_V2(inputs, weights):
            """Ansatz V2: encoding + RX/RZ + all-to-all CRX + RX/RZ."""
            weights = weights.flatten()

            for i in range(self.n_layers):
                if self.State_Prep == "Angle":
                    _hadamard_helper(self.n_qubits)

                _encoding_helper(self.n_qubits, inputs, i, self.State_Prep, self.flag_FT)

                # First RX/RZ block
                for x in range(self.n_qubits):
                    qml.RX(phi=weights[i * self.n_qubits * 2 + 2 * x + 0], wires=x)
                    qml.RZ(phi=weights[i * self.n_qubits * 2 + 2 * x + 1], wires=x)
                qml.Barrier()

                # All-to-all CRX entanglers
                for x in reversed(range(self.n_qubits)):
                    for y in reversed(range(self.n_qubits)):
                        if x != y:
                            qml.CRX(
                                phi=weights[i * self.n_qubits * 2 + 2 * x + 0],
                                wires=[x, y],
                            )
                qml.Barrier()

                # Second RX/RZ block
                for x in range(self.n_qubits):
                    qml.RX(phi=weights[i * self.n_qubits * 2 + 2 * x + 0], wires=x)
                    if i < self.n_layers - 1:
                        qml.RZ(
                            phi=weights[i * self.n_qubits * 2 + 2 * x + 1],
                            wires=x,
                        )
                qml.Barrier()

        # ------------------------------------------------------------------
        # 4. PennyLane device and QNode
        # ------------------------------------------------------------------
        if self.flag_GPU:
            dev = qml.device(
                "lightning.gpu", wires=self.n_qubits, shots=self.n_shots
            )
        else:
            dev = qml.device(
                "default.qubit", wires=self.n_qubits, shots=self.n_shots
            )

        @qml.qnode(device=dev, interface="tf")
        def qnode(inputs, weights):
            """PennyLane QNode wrapping the chosen QVC ansatz."""
            inputs_ = np.asarray(inputs)
            weights_ = np.asarray(weights[0])

            if self.type_QVC == "V1":
                qnode_V1(inputs_, weights_)
            elif self.type_QVC == "V2":
                qnode_V2(inputs_, weights_)

            return [
                qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)
            ]

        # Optionally draw the circuit for a random input/weight sample
        if self.flag_draw:
            inputs_draw = np.random.rand(self.batchsize, self.shape)

            if self.type_QVC == "V1":
                weights_draw = np.random.rand(
                    1, self.n_qubits * 2 * self.n_layers
                )
            else:
                n_params = (
                    2 * self.n_qubits * 2
                    + (self.n_qubits - 1) * self.n_qubits
                ) * self.n_layers - self.n_qubits
                weights_draw = np.random.rand(1, n_params)

            if self.State_Prep == "Amplitude":
                for a in range(len(inputs_draw)):
                    inputs_draw[a] = inputs_draw[a] / np.linalg.norm(
                        inputs_draw[a]
                    )
            elif self.State_Prep == "Angle":
                inputs_draw = np.random.rand(
                    self.batchsize, self.n_qubits * 2 * self.n_layers
                )

            fig, ax = qml.draw_mpl(qnode)(
                inputs_draw, weights_draw
            )
            fig.show()

        # ------------------------------------------------------------------
        # 5. Wrap QNode in QuantumLayer or build pure NN baseline
        # ------------------------------------------------------------------
        if not self.flag_NN:
            # Broadcast-expand QNode to handle batched inputs
            qnode_expanded = qml.transforms.broadcast_expand(qnode)

            # Number of trainable parameters in the circuit
            if self.type_QVC == "V1":
                n_params = self.n_qubits * 2 * self.n_layers
            elif self.type_QVC == "V2":
                n_params = (
                    2 * self.n_qubits * 2
                    + (self.n_qubits - 1) * self.n_qubits
                ) * self.n_layers - self.n_qubits

            qlayer = QuantumLayer(
                n_qubits=self.n_qubits,
                circuit=qnode_expanded,
                weight_shape=(1, n_params),
            )

        def get_classical_model(n_qubits, n_layers, shape):
            """Construct classical fully connected baseline.

            For flag_NN_optuna=False, use a fixed-width sin-activated MLP
            with width 2**n_qubits and depth 3*n_layers - 1. For
            flag_NN_optuna=True, use a tanh-activated MLP with layer sizes
            specified in self.NN_units.
            """
            model_input = Input(shape=(shape,))

            if not self.flag_NN_optuna:
                x = Dense(2**n_qubits, activation=tf.math.sin)(model_input)
                for _ in range(3 * n_layers - 2):
                    x = Dense(2**n_qubits, activation=tf.math.sin)(x)
                model_output = Dense(
                    2**n_qubits, activation=tf.math.sin
                )(x)
            else:
                x = Dense(
                    self.NN_units[0], activation=tf.math.tanh
                )(model_input)
                for units in self.NN_units[1:]:
                    x = Dense(units, activation=tf.math.tanh)(x)
                model_output = x

            return Model(inputs=model_input, outputs=model_output)

        # ------------------------------------------------------------------
        # 6. Assemble final model: encoder/QVC or pure NN + sigmoid head
        # ------------------------------------------------------------------
        if self.flag_NN:
            # Classical baseline only
            model = get_classical_model(
                self.n_qubits, self.n_layers, self.shape
            )
            model_input = model.input
            output = model.output
        else:
            # Quantum model
            model_input = Input(shape=(self.shape,))

            # For angle encoding, tile features across layers; for Mottonen
            # the same (already normalized) input is used once.
            tile_shape = self.n_layers if self.State_Prep == "Angle" else 1
            qlayer_input = tf.tile(model_input, (1, tile_shape))

            output = qlayer(qlayer_input)

        # Final sigmoid output for binary classification
        mlp4 = Dense(units=1, activation="sigmoid")(output)

        model = Model(inputs=model_input, outputs=mlp4)
        model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
            run_eagerly=False,
        )

        if self.flag_summary:
            model.summary()

        return model


    def get_ind_size(self):
        """Return the total number of trainable scalar parameters in the model.

        This is used to determine the dimensionality of the "individual"
        vector in evolutionary / population-based optimization.
        """
        model = self.get_model()
        return int(
            np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        )

    def map_individual(self, individual, model):
        """Map a flat parameter vector ("individual") onto a compiled Keras model.

        The ordering in `individual` is assumed to match the current model layout:

          * Dense layers: [W (flattened, row-major), b]
          * BatchNormalization layers:
              - If self.flag_GF is True: only gamma, beta (mean=0, std=1 fixed)
              - Otherwise: gamma, beta, moving_mean, moving_variance
          * QuantumLayer: single weight tensor of shape (1, n_params)

        Args:
            individual: 1D array containing all trainable parameters.
            model:      Keras model instance with the correct architecture
                        but arbitrary weights.

        Returns:
            model: same model instance with weights set from `individual`.
        """
        individual = np.asarray(individual)

        input_dim = model.input_shape[1]

        # First "real" layer may either be an InputLayer (no weights) or Dense
        if isinstance(model.layers[0], layers.InputLayer):
            units = input_dim
            current_index = 0
        else:
            units = model.layers[0].units

            w_size = input_dim * units
            W = np.array(individual[:w_size]).reshape(input_dim, units)
            b = np.array(individual[w_size:w_size + units])

            model.layers[0].set_weights([W, b])
            current_index = (input_dim + 1) * units

        # Iterate over remaining layers and slice from `individual` accordingly
        for i in range(1, len(model.layers)):
            layer = model.layers[i]

            if isinstance(layer, layers.Dense):
                # Dense: weights of shape (units_prev, units) and bias of shape (units)
                out_units = layer.units
                w_size = units * out_units

                W = np.array(
                    individual[current_index:current_index + w_size]
                ).reshape(units, out_units)
                b = np.array(
                    individual[current_index + w_size:
                               current_index + w_size + out_units]
                )

                layer.set_weights([W, b])
                current_index += (units + 1) * out_units
                units = out_units

            elif isinstance(layer, layers.BatchNormalization):
                # BatchNorm: gamma, beta, moving_mean, moving_variance
                if self.flag_GF:
                    # Fixed mean and std; only gamma and beta are trainable
                    gamma = np.array(
                        individual[current_index:current_index + units]
                    )
                    beta = np.array(
                        individual[current_index + units:
                                   current_index + 2 * units]
                    )
                    mean = np.zeros(units)
                    std = np.ones(units)
                    current_index += 2 * units
                else:
                    gamma = np.array(
                        individual[current_index:
                                   current_index + 1 * units]
                    )
                    beta = np.array(
                        individual[current_index + 1 * units:
                                   current_index + 2 * units]
                    )
                    mean = np.array(
                        individual[current_index + 2 * units:
                                   current_index + 3 * units]
                    )
                    std = np.array(
                        individual[current_index + 3 * units:
                                   current_index + 4 * units]
                    )
                    current_index += 4 * units

                layer.set_weights([gamma, beta, mean, std])

            elif isinstance(layer, QuantumLayer):
                # QuantumLayer: single parameter tensor of shape (1, n_params)
                q_params = layer.weight_shape[1]
                W_q = np.array(
                    individual[current_index:current_index + q_params]
                ).reshape(1, q_params)

                layer.set_weights([W_q])
                current_index += q_params

                # After the QVC, the next layer sees n_qubits features
                units = self.n_qubits

        return model


    def get_individual(self, model_weights):
        """Flatten model weights into a single 1D parameter vector ("individual").

        Args:
            model_weights: iterable of TensorFlow variables / weight tensors
                           (e.g., model.trainable_weights).

        Returns:
            list or np.ndarray: flattened parameter vector, in the same order
            as Keras returns the weights.
        """
        individual = []
        for weights in model_weights:
            # Each `weights` may be a vector or matrix; stack and flatten
            individual.extend(np.hstack(tup=weights.numpy()))
        return individual

    def _parallel_run(self, kwargs):
        """Worker function for parallel prediction with multiprocessing.

        Args:
            kwargs: dict with keys:
                - "individual": flat parameter vector.
                - "X_k":        batch of input data.

        Returns:
            1D numpy array: model predictions on X_k.
        """
        # Construct a fresh model and load the individual's weights
        model = self.map_individual(kwargs["individual"], self.get_model())
        # Forward pass (no training) on the sub-batch
        return model(kwargs["X_k"], training=False).numpy().flatten()

    def _parallel_helper(self, X_data, individual):
        """Compute model predictions for X_data, optionally in parallel.

        If a GPU is available or max_threads == 1, predictions are computed in
        a single process via model.predict. Otherwise, X_data is split into
        batches and evaluated using multiprocessing.Pool.

        Args:
            X_data:    input data, shape (n_samples, input_dim).
            individual: flat parameter vector representing model weights.

        Returns:
            1D numpy array of predictions for all samples in X_data.
        """
        # Single-process execution if GPU is present or threading disabled
        if self.flag_GPU or self.max_threads == 1:
            model = self.map_individual(individual, self.get_model())
            return model.predict(
                X_data, batch_size=self.batchsize, verbose=0
            ).flatten()

        # Multi-process execution on CPU
        n_batches = len(X_data) // self.batchsize
        X_splits = np.array_split(X_data, n_batches)

        kwargs_list = [
            {"individual": individual, "X_k": X_splits[k]}
            for k in range(n_batches)
        ]

        with Pool(min(n_batches, self.max_threads)) as pool:
            jobs = pool.map(self._parallel_run, kwargs_list)
            y_predict = list(itertools.chain.from_iterable(jobs))

        return np.asarray(y_predict)


    def bce(self, y_pred, y_true, eps=1e-15):
        """Binary cross-entropy (elementwise), with numerical clipping.

        Args:
            y_pred: predicted probabilities in [0, 1], array-like.
            y_true: binary labels {0, 1}, same shape as y_pred.
            eps:    small constant to avoid log(0).

        Returns:
            Array of per-sample BCE values.
        """
        # Convert to arrays compatible with pennylane.numpy
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, eps, 1.0 - eps)

        # Standard binary cross-entropy: -[y log p + (1-y) log (1-p)]
        return -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))

    def evaluate_single_final(self, X_data, y_labels, individual):
        """Evaluate a single individual on a given dataset.

        This helper is used in the evolutionary optimization loop. It:
          * maps the flat parameter vector `individual` onto a model,
          * computes predictions on `X_data` (possibly in parallel),
          * returns mean BCE and accuracy, rounded to 4 decimals.

        Args:
            X_data:     input data, shape (n_samples, input_dim).
            y_labels:   corresponding binary labels {0, 1}; may be a
                        TensorFlow tensor or NumPy / pennylane.numpy array.
            individual: flat parameter vector specifying model weights.

        Returns:
            (bce, acc): tuple of mean binary cross-entropy and accuracy.
        """
        # Forward pass to obtain predictions for all samples
        y_pred = np.asarray(self._parallel_helper(X_data, individual))

        # Convert labels to NumPy / pennylane.numpy
        if hasattr(y_labels, "numpy"):
            y_true = np.asarray(y_labels.numpy())
        else:
            y_true = np.asarray(y_labels)

        # Mean BCE over all samples
        bce_values = self.bce(y_pred, y_true)
        bce = float(np.mean(bce_values))

        # Threshold predictions at 0.5 to obtain hard labels
        y_hat = (y_pred > 0.5).astype(int).flatten()

        # Classification accuracy
        acc = float(np.mean(y_true.flatten() == y_hat))

        # Round for reporting
        return round(bce, 4), round(acc, 4)
