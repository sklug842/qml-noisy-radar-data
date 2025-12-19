import tensorflow                    as     tf
from   tqdm                          import tqdm
from   help_classes.abstract_classes import AbstractOptimizer_GB

# Use float64 to match PennyLane's default and avoid dtype mismatches
tf.keras.backend.set_floatx("float64")


class ADAM(AbstractOptimizer_GB):
    """Gradient-based optimization of a hybrid QML model using Adam.

    This optimizer:
      * builds the hybrid quantum–classical model from `wrapper.get_model()`,
      * trains it with standard mini-batch Adam on training data,
      * records test accuracy after each epoch via the wrapper's evaluation
        routine (which expects a flattened parameter vector, "individual").
    """

    def __init__(self, wrapper, n_epochs, eta) -> None:
        """Initialize the Adam optimizer.

        Args:
            wrapper: model wrapper providing data, model, and evaluation.
            n_epochs: number of training epochs.
            eta: learning rate for the Adam optimizer.
        """
        # n_evals_per_epoch is not used for gradient-based methods,
        # but kept for interface consistency with the abstract base class.
        super().__init__(wrapper=wrapper,
                         n_evals_per_epoch=0,
                         n_epochs=n_epochs,
                         eta=eta)

        self.wrapper  = wrapper
        self.n_epochs = n_epochs

    def run(self):
        """Run Adam training and return test accuracy after each epoch.

        Returns:
            List of accuracies on the test set, one value per epoch.
        """
        # Build and compile the hybrid model (quantum or classical)
        model = self.wrapper.get_model()

        # Adam optimizer with learning rate from base class
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.eta)

        # Binary cross-entropy loss for probabilistic outputs in [0, 1]
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        @tf.function
        def train_step(x_batch, y_batch):
            """Single mini-batch optimization step."""
            with tf.GradientTape() as tape:
                # Forward pass
                y_pred = model(x_batch, training=True)

                # Reshape labels and predictions to 1D vectors [batch_size]
                y_true_vec = tf.reshape(y_batch, [tf.shape(x_batch)[0]])
                y_pred_vec = tf.reshape(y_pred,  [tf.shape(x_batch)[0]])

                # Compute batch loss
                loss_value = loss_fn(y_true_vec, y_pred_vec)

            # Backward pass: gradients w.r.t. trainable weights
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Parameter update via Adam
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Store test accuracy after each epoch
        accuracies = []

        # Epoch loop
        pbar = tqdm(range(1, self.n_epochs + 1))
        for epoch in pbar:
            pbar.set_description(f"Currently running epoch {epoch}")

            # Iterate over mini-batches of the training set
            n_batches = len(self.wrapper.X_train) // self.wrapper.batchsize
            for i in range(n_batches):
                start = i * self.wrapper.batchsize
                end   = start + self.wrapper.batchsize

                x_batch = self.wrapper.X_train[start:end]
                y_batch = self.wrapper.y_train[start:end]

                train_step(x_batch, y_batch)

            # Convert current model weights to a flat "individual" vector,
            # since evaluate_single_final expects this representation.
            individual = self.wrapper.get_individual(model.weights)

            # Evaluate on test set and store accuracy
            acc = self.wrapper.evaluate_single_final(
                X_data     = self.wrapper.X_test,
                y_labels   = self.wrapper.y_test,
                individual = individual,
            )[1]
            accuracies.append(acc)

        return accuracies