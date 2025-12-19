import numpy                         as     np
from   tqdm                          import tqdm
from   help_classes.abstract_classes import AbstractOptimizer_GF


class Hashemi(AbstractOptimizer_GF):
    """Gradient-free genetic optimizer following Hashemi et al.

    The optimizer maintains a single incumbent individual θ and repeatedly:
      1. Selects a contiguous segment of θ on a ring (i.e., with wrap-around).
      2. Perturbs entries in this segment with Gaussian noise (σ = sigma_gs)
         independently, with a crossover-like probability p_cx = 1 / (1 + log g),
         where g is the segment length.
      3. Accepts the mutated "larva" if it yields a lower training BCE.

    Test accuracy is recorded after each epoch (n_evals_per_epoch iterations).
    """

    def __init__(self, wrapper, n_evals_per_epoch, n_epochs, sigma_init, sigma_gs) -> None:
        """Initialize the Hashemi optimizer.

        Args:
            wrapper:            model wrapper providing evaluation and ind_size.
            n_evals_per_epoch:  number of fitness evaluations per epoch.
            n_epochs:           number of epochs (outer loops).
            sigma_init:         standard deviation of the initial Gaussian
                                distribution for the individual.
            sigma_gs:           standard deviation of the Gaussian mutation noise.
        """
        super().__init__(wrapper=wrapper,
                         n_evals_per_epoch=n_evals_per_epoch,
                         n_epochs=n_epochs)

        self.sigma_init   = sigma_init
        self.sigma_gs     = sigma_gs

        # Dimensionality of the individual (flat parameter vector)
        self.ind_size     = self.wrapper.get_ind_size()

        # Total number of mutation–selection steps
        self.n_iterations = n_epochs * n_evals_per_epoch

    def run(self):
        """Run the Hashemi optimizer.

        Returns:
            List of test accuracies, one entry per epoch
            (i.e., after every `n_evals_per_epoch` iterations).
        """

        # ------------------------------------------------------------------
        # 1. Initialize incumbent individual θ ~ N(0, sigma_init^2 I)
        # ------------------------------------------------------------------
        individual = np.random.normal(loc=0.0,
                                      scale=self.sigma_init,
                                      size=self.ind_size)

        # Evaluate incumbent fitness (training BCE)
        fit_individual = self.wrapper.evaluate_single_final(
            X_data     = self.wrapper.X_train,
            y_labels   = self.wrapper.y_train,
            individual = individual,
        )[0]

        # Store test-set accuracy after each epoch
        accuracies = []

        # ------------------------------------------------------------------
        # 2. Main mutation–selection loop
        # ------------------------------------------------------------------
        pbar = tqdm(range(1, self.n_iterations + 1))
        for generation in pbar:
            pbar.set_description(f"Currently running generation {generation}")

            # Copy current individual to form the "larva" candidate
            larva = np.copy(individual)

            # Sample segment length g and ending index on the ring [0, ind_size)
            g            = np.random.choice(a=np.arange(1, self.ind_size + 1))
            ending_point = np.random.choice(a=np.arange(0, self.ind_size))

            # Crossover-like probability controlling how many genes are perturbed
            p_cx = 1.0 / (1.0 + np.log(g))

            # ------------------------------------------------------------------
            # 2a. Apply Gaussian mutation to a contiguous segment on the ring
            # ------------------------------------------------------------------
            if ending_point - g < 0:
                # Segment wraps around: indices [0, ending_point) and
                # [ind_size - (g - ending_point), ind_size)

                # Bernoulli masks for whether to perturb each gene
                p1 = np.random.uniform(low=0.0, high=1.0, size=ending_point)
                p2 = np.random.uniform(low=0.0, high=1.0, size=g - ending_point)

                # Additive Gaussian noise on the two parts of the segment
                gaussian_noise1 = np.random.normal(loc=0.0,
                                                   scale=self.sigma_gs,
                                                   size=ending_point)
                gaussian_noise2 = np.random.normal(loc=0.0,
                                                   scale=self.sigma_gs,
                                                   size=g - ending_point)

                larva[0:ending_point]       += (p1 < p_cx) * gaussian_noise1
                larva[-(g - ending_point):] += (p2 < p_cx) * gaussian_noise2

            else:
                # Segment does not wrap: indices [ending_point - g, ending_point)

                p              = np.random.uniform(low  = 0.0, 
                                                   high = 1.0,
                                                   size = g  )

                gaussian_noise = np.random.normal(loc   = 0.0          ,
                                                  scale = self.sigma_gs,
                                                  size  = g            )

                larva[ending_point - g:ending_point] += (p < p_cx) * gaussian_noise

            # ------------------------------------------------------------------
            # 2b. Evaluate larva and accept if fitness improves
            # ------------------------------------------------------------------
            fit_larva = self.wrapper.evaluate_single_final(
                X_data     = self.wrapper.X_train,
                y_labels   = self.wrapper.y_train,
                individual = larva,
            )[0]

            # Minimization of BCE: accept if larva has lower fitness
            if fit_larva < fit_individual:
                individual     = np.copy(larva)
                fit_individual = np.copy(fit_larva)

            # ------------------------------------------------------------------
            # 2c. At the end of each epoch, record test-set accuracy
            # ------------------------------------------------------------------
            if generation % self.n_evals_per_epoch == 0:
                acc = self.wrapper.evaluate_single_final(
                    X_data     = self.wrapper.X_test,
                    y_labels   = self.wrapper.y_test,
                    individual = individual,
                )[1]
                accuracies.append(acc)

        return accuracies