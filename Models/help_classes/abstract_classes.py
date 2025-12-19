import abc

class AbstractWrapper():

    def __init__(self) -> None:
        """
        the constructor of a wrapper
        """

        self.flag_GF = False
    
    @abc.abstractmethod
    def evaluate_single_final(self, X_data, y_labels, individual: list[float]) -> float:
        """
        evaluation of the individual on X_data 
        """


class AbstractOptimizer():

    def __init__(self, wrapper: AbstractWrapper, n_evals_per_epoch: int, n_epochs: int) -> None:
        """
        the constructor of an optimizer
        """

        self.wrapper           = wrapper
        self.n_evals_per_epoch = n_evals_per_epoch
        self.n_epochs          = n_epochs

    @abc.abstractmethod
    def run(self) -> float:
        """
        runs the algorithm and returns the final evaluation
        """


class AbstractOptimizer_GB(AbstractOptimizer):

    def __init__(self, wrapper: AbstractWrapper, n_evals_per_epoch: int, n_epochs: int, eta: float) -> None:
        """
        the constructor of an gradient based optimizer
        """

        super().__init__(wrapper, n_evals_per_epoch, n_epochs)

        self.eta             = eta
        self.wrapper.flag_GF = False


class AbstractOptimizer_GF(AbstractOptimizer):

    def __init__(self, wrapper: AbstractWrapper, n_evals_per_epoch: int, n_epochs: int) -> None:
        """
        the constructor of an gradient free optimizer
        """

        super().__init__(wrapper, n_evals_per_epoch, n_epochs)

        self.wrapper.flag_GF   = True