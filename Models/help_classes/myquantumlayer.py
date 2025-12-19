import numpy as np
import tensorflow as tf


class QuantumLayer(tf.keras.layers.Layer):
    """Keras layer that wraps a quantum circuit evaluated via a PennyLane-style QNode.

    The layer:
      * Calls a user-supplied circuit `circuit(x, weights)` in the forward pass
        using `tf.py_function` (eager execution).
      * Implements a custom gradient w.r.t. the quantum weights using
        the parameter-shift rule with ±π/2 shifts.
      * Does not propagate gradients w.r.t. the layer inputs (dL/dX = 0).
    """

    def __init__(self, circuit, n_qubits, weight_shape, **kwargs):
        """
        Args:
            circuit: callable with signature `circuit(x, weights)` returning
                a collection of expectation values per sample.
            n_qubits: number of qubits / expectation values (output dimension).
            weight_shape: shape of the trainable weight tensor expected by `circuit`.
        """

        super().__init__(**kwargs)

        # Number of qubits / output dimension of the quantum circuit
        self.n_qubits     = n_qubits

        # Reference to the quantum circuit (typically a PennyLane QNode)
        self.circuit      = circuit

        # Shape of the trainable parameters for the quantum circuit
        self.weight_shape = weight_shape

        # Constant π in float64 for parameter-shift updates
        self.pi           = tf.constant(np.pi, dtype=tf.float64)

    def build(self, input_shape):
        """
        Create the trainable quantum weights once the input shape is known
        """

        # Trainable weights for the quantum circuit
        self.weights_qc = self.add_weight(
            name="weights_qc",
            shape=self.weight_shape,
            initializer=tf.keras.initializers.RandomUniform(
                minval=0.0, maxval=2 * np.pi
            ),
            trainable=True,
        )

        super().build(input_shape)

    def set_quantum_weights(self, new_weights):
        """
        Set the quantum circuit weights from an external array or tensor
        """

        new_weights = tf.convert_to_tensor(new_weights, dtype=self.weights_qc.dtype)
        self.weights_qc.assign(new_weights)

    @tf.custom_gradient
    def my_caller(self, X_data):
        """Forward pass through the quantum circuit with an explicit custom gradient.

        Args:
            X_data: input tensor of shape (batch_size, input_dim).

        Returns:
            output: tensor of shape (batch_size, n_qubits).
            grad_fn: function implementing dL/dX_data and dL/d(weights_qc).
        """

        def qnode_wrapper(x, weights):
            """Eager wrapper around the quantum circuit.

            Runs the (typically PennyLane) circuit on a batch of inputs `x`
            and reshapes the result to (batch_size, n_qubits).
            """

            # `circuit` usually returns one result per sample
            output = self.circuit(x, weights)
            output = tf.stack(output, axis=1)
            output = tf.reshape(output, (len(x), self.n_qubits))
            return tf.cast(output, tf.float64)

        # Forward pass: call the circuit via tf.py_function to keep it in eager mode
        output = tf.py_function(
            func=qnode_wrapper,
            inp=[X_data, self.weights_qc],
            Tout=tf.float64,
        )
        output = tf.reshape(output, (-1, self.n_qubits))

        def grad_calculations(upstream, *, variables):
            """Custom gradient implementing the parameter-shift rule.

            Args:
                upstream: dL/d(output), shape (batch_size, n_qubits).
                variables: list of variables used in the forward pass
                    (here: [weights_qc]).

            Returns:
                dL/dX_data (set to 0) and dL/d(weights_qc) as required by Keras.
            """
            
            upstream            = tf.cast(upstream, tf.float64)

            # Extract flattened weight vector; shape (n_params,)
            weights             = variables[0][0]
            n_params            = weights.shape[0]

            # Batch size remains symbolic here
            batch_size          = tf.shape(X_data)[0]

            # Parameter-shift offsets: W ± (π/2) e_i for each parameter i
            W_col               = tf.cast(weights, tf.float64)
            eye                 = tf.cast(tf.eye(n_params), tf.float64)
            angles_plus         = W_col + (self.pi / 2.0) * eye
            angles_minus        = W_col - (self.pi / 2.0) * eye

            # Collect gradients dL/dθ_i in a single row vector
            angles_gradients     = tf.zeros((1, n_params), dtype=tf.float64)

            for i in range(n_params):
                # Parameter vectors shifted in the i-th direction; shape (1, n_params)
                phi_plus         = tf.expand_dims(angles_plus[i], axis=0)
                phi_minus        = tf.expand_dims(angles_minus[i], axis=0)

                # Evaluate circuit at θ_i ± π/2 using the same eager wrapper
                plus_pi2         = tf.py_function(
                    func=qnode_wrapper,
                    inp=[X_data, phi_plus],
                    Tout=tf.float64,
                )
                plus_pi2         = tf.reshape(plus_pi2, (batch_size, self.n_qubits))

                minus_pi2        = tf.py_function(
                    func=qnode_wrapper,
                    inp=[X_data, phi_minus],
                    Tout=tf.float64,
                )
                minus_pi2        = tf.reshape(minus_pi2, (batch_size, self.n_qubits))

                # Parameter-shift derivative doutput/dθ_i
                angle_gradient   = (plus_pi2 - minus_pi2) / 2.0
                angle_gradient   = tf.reshape(angle_gradient, (batch_size, self.n_qubits))

                # Chain rule: sum over batch and qubit outputs
                one_grad         = tf.reduce_sum(angle_gradient * upstream)  # scalar

                angles_gradients = tf.tensor_scatter_nd_update(
                    tensor=angles_gradients,
                    indices=[[0, i]],
                    updates=[one_grad],
                )

            # No gradient is propagated w.r.t. inputs (dL/dX_data = 0)
            input_gradients       = tf.zeros_like(X_data, dtype=tf.float64)

            return (
                tf.convert_to_tensor(input_gradients, dtype=tf.float64),
                [angles_gradients],
            )

        return output, grad_calculations

    def call(self, inputs):
        """
        Standard Keras call method
        """

        return self.my_caller(inputs)

    def compute_output_shape(self, input_shape):
        """
        Output shape is (batch_size, n_qubits)
        """

        return tf.TensorShape((input_shape[0], self.n_qubits))
