from enum import Enum
from typing import Any, Callable, Iterable, Optional, Type
from warnings import deprecated

import numpy as np
import torch
from torch import Tensor, nn


class SelectionMethod(Enum):
    ROULETTE_WHEEL = 0
    TOURNAMENT = 1


class GradientProjection(object):
    @staticmethod
    @torch.no_grad()
    @torch.jit.script
    def transform(
        params: Tensor,
        grad: Tensor,
        grad_norm: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Transform the parameters into parallel and orthogonal components
        with respect to the gradient.

        Args:
            params (Tensor): Shape of (n,)
            grad (Tensor): Shape of (n,)
            grad_norm (Tensor): Norm of the parameters `grad`

        Returns:
            tuple[Tensor, Tensor]: Tuple containing the parallel and orthogonal components.
        """
        # if params.dim() != 1 or grad.dim() != 1:
        #     raise ValueError("Both params and grad must be 1D tensors.")

        # Normalize the gradient
        normalized_grad: Tensor = grad / grad_norm

        parallel: Tensor = (params @ normalized_grad) * normalized_grad
        orthogonal: Tensor = params - parallel

        return parallel, orthogonal

    @staticmethod
    @torch.no_grad()
    @torch.jit.script
    def inverse_transform(parallel: Tensor, orthogonal: Tensor) -> Tensor:
        """
        Inverse transform the parallel and orthogonal components back to the original parameters.

        Args:
            parallel (Tensor): Shape of (n,)
            orthogonal (Tensor): Shape of (n,)

        Returns:
            Tensor: The reconstructed parameters of shape (n,).
        """

        return parallel + orthogonal


class Individual(object):
    def __init__(
        self,
        model: nn.Module,
        compile: bool,
        dtype: torch.dtype,
        device: torch.device,
        name: Optional[str] = None,
    ):
        self._model: nn.Module = (
            model.to(device=device, dtype=dtype)
            if not compile
            else torch.compile(
                model.to(device=device, dtype=dtype),
            )
        )  # type: ignore
        self.dtype = dtype
        self.device = device
        self._name = name or f"ID-{id(self)}"

        self._is_fitness_valid: bool = False
        self._fitness: float = 0.0  # Inverse of the training loss
        self._train_loss: float = float("inf")
        self._outputs: Optional[Tensor] = None

    def __call__(self, *args, **kwargs) -> Tensor:
        """
        Forward pass through the model.
        Invoke nn.Module's __call__ (i.e., forward) method.

        Args:
            *args: Positional arguments for the model.
            **kwargs: Keyword arguments for the model.

        Returns:
            Tensor: Output of the model.
        """
        return self._model(*args, **kwargs)

    @property
    def name(self) -> str:
        return self._name

    @property
    def fitness(self) -> float:
        """
        Get the fitness value of the individual.
        Fitness is the inverse of the training loss; higher is better.

        Returns:
            float: The fitness value.
        """
        # if not self._is_fitness_valid:
        #     raise ValueError("Fitness is not computed yet. Call compute_fitness first.")
        return self._fitness

    def update_fitness(self, train_loss: float):
        """
        Update the fitness value of the individual.
        This method is used to set the fitness directly.

        Args:
            train_loss (float): The training loss to set the fitness from.
        """
        # if self._is_fitness_valid:
        #     raise ValueError(f"Fitness is already computed for {self._name}.")

        # if self._fitness == 0.0:
        #     self._fitness = 1.0 / train_loss
        # else:
        #     _new_fitness: float = 1.0 / train_loss
        #     w = 1.0  # Weight for the new fitness
        #     self._fitness = w * _new_fitness + (1 - w) * self._fitness
        self._fitness = 1.0 / train_loss

        self._is_fitness_valid = True

    @property
    def train_loss(self) -> float:
        return self._train_loss

    @property
    def outputs(self) -> Tensor:
        """
        Get the outputs of the model from the last forward pass.

        Returns:
            Tensor: The outputs of the model, or None if not computed.
        """
        if self._outputs is None:
            raise ValueError("Outputs are not computed yet. Call forward first.")
        return self._outputs

    def zero_grad(self):
        """
        Alias of nn.Module.zero_grad()
        """
        self._model.zero_grad()

    def train(self):
        """
        Alias of nn.Module.train()
        """
        self._model.train()

    def eval(self):
        """
        Alias of nn.Module.eval()
        """
        self._model.eval()

    def to(self, device: torch.device):
        """
        Move the model to the specified device.

        Args:
            device (torch.device): The device to move the model to.
        """
        self._model.to(device)
        self.device = device

    @torch.no_grad()
    def get_flat_params(self) -> Tensor:
        """
        Get flattened parameters of the model.

        Returns:
            Tensor: Flattened parameters of the model.
        """
        params: list[Tensor] = [
            p.data.view(-1) for p in self._model.parameters() if p.requires_grad
        ]
        # if not params:
        #     raise ValueError(
        #         f"No parameters with gradients found in {self._name}."
        #         " Ensure model has learnable parameters."
        #     )
        return torch.cat(params)

    @torch.no_grad()
    def set_flat_params(self, flat_params: Tensor) -> "Individual":
        """
        Set parameters from flattened tensor in-place.
        Note that this will not instantiate a new model, but will
        update the existing model's parameters directly.

        Args:
            flat_params (Tensor): Flattened parameters to set in the model.
        Returns:
            Individual: The individual with updated parameters.
        """
        learnable_params: list[nn.Parameter] = [
            p for p in self._model.parameters() if p.requires_grad
        ]
        offset = 0
        for param in learnable_params:
            numel: int = param.numel()
            param.data.copy_(flat_params[offset : offset + numel].view(param.shape))
            offset += numel

        # Parameters are set, so fitness is no longer valid
        self._is_fitness_valid = False
        self._outputs = None
        self._train_loss = float("inf")
        self.zero_grad()

        return self

    @torch.no_grad()
    def get_flat_gradients(self) -> Tensor:
        """
        Get flattened gradients.

        Returns:
            Tensor: Flattened gradients of the model
        """
        learnable_params: list[nn.Parameter] = [
            p
            for p in self._model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        grads = []
        for param in learnable_params:
            # if param.grad is None:
            #     raise ValueError(
            #         f"Gradient is None for parameter in {self._name}. Ensure backward has been invoked."
            #     )
            grads.append(param.grad.data.view(-1))  # type: ignore

        # if not grads:
        #     raise ValueError(
        #         f"No learnable parameters found in {self._name}."
        #         " Ensure the model has learnable parameters."
        #     )

        return torch.cat(grads)

    def forward(
        self,
        criterion: Callable[[Tensor, Tensor], Tensor],
        inputs: Tensor,
        targets: Tensor,
        compute_gradients: bool,
        update_fitness: bool,
    ) -> Tensor:
        """
        Single forward step of the model using a single batch of data
        and compute the loss. Optionally compute gradients.
        This method also updates the fitness and train loss of the individual.

        Args:
            criterion (Callable): Loss function to compute the loss.
            inputs (Tensor): Batch of input data for the model.
            targets (Tensor): Batch of target labels for the model.
            compute_gradients (bool): Whether to compute gradients after the forward pass.
            update_fitness (bool): Whether to update the fitness of the individual.
        Returns:
            Tensor: Outputs of the model that can be used for backpropagation.
        """
        self.zero_grad()  # Prevent gradient accumulation

        outputs: Tensor = self._model(inputs)
        loss: Tensor = criterion(outputs, targets)
        if compute_gradients:
            loss.backward()
        self._outputs = outputs.detach()

        self._train_loss = loss.item() + 1e-8
        if update_fitness:
            self.update_fitness(self._train_loss)

        return outputs


class Population(list[Individual]):
    def __init__(self, individuals: Optional[Iterable[Individual]] = None):
        super().__init__(individuals if individuals is not None else [])

    @torch.no_grad()
    def compute_fitnesses(
        self,
        criterion: Callable[[Tensor, Tensor], Tensor],
        inputs: Tensor,
        targets: Tensor,
    ):
        """
        Evaluate all individuals in the population using a single batch of data.
        Does not compute gradients, only the loss and the corresponding fitness.

        Args:
            criterion (Callable): Loss function to compute the loss.
            inputs (Tensor): Batch of input data.
            targets (Tensor): Batch of target labels.
        """
        for individual in self:
            # if individual._is_fitness_valid:
            #     raise ValueError(f"Fitness is already computed for {individual.name}.")
            individual.train()  # TODO: train or eval mode?
            individual.forward(
                criterion,
                inputs,
                targets,
                compute_gradients=False,
                update_fitness=True,
            )

    def fitnesses(self) -> list[float]:
        """
        Get the fitness values of all individuals in the population.
        This method assumes that fitnesses have already been computed.

        Returns:
            list[float]: List of fitness values.
        """
        return [ind.fitness for ind in self]

    def sort_by_fitness(self):
        """
        Sort the individuals in descending order based on their fitness.
        This method assumes that fitnesses have already been computed.
        """
        self.sort(key=lambda ind: ind.fitness, reverse=True)


class GPGAEnhanced(object):

    def __init__(
        self,
        model_class: Type[nn.Module],
        model_kwargs: dict[str, Any],
        criterion: Callable[[Tensor, Tensor], Tensor],
        init_population_size: int = 32,
        elite_size: int = 4,
        selection_method: SelectionMethod = SelectionMethod.ROULETTE_WHEEL,
        offspring_size: int = 24,
        tau: float = 3.0,
        mutation_rate: float = 0.25,
        mutation_rate_decay: float = 0.999,
        parallel_mutation_strength: float = 0.1,
        parallel_mutation_decay: float = 1.0,
        orthogonal_mutation_strength: float = 0.2,
        orthogonal_mutation_decay: float = 0.999,
        compile_model: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the GPGA optimizer.

        Args:
            model_class (Type[nn.Module]): Model instance generator.
            model_kwargs (dict[str, Any]): Keyword arguments for the model.
            criterion (Callable): Loss function to compute the loss.
            init_population_size (int): Initial population size.
            elite_size (int): Number of elite individuals to retain.
            selection_method (SelectionMethod): Method for selecting parents.
            offspring_size (int): Number of offspring to generate.
            tau (float): Selection pressure for the crossover operation; temperature.
            mutation_rate (float): Probability of mutation for each offspring.
            mutation_decay (float): Decay factor for the mutation rate.
            parallel_mutation_strength (float): Strength of parallel mutation.
            parallel_mutation_decay (float): Decay factor for parallel mutation strength.
            orthogonal_mutation_strength (float): Strength of orthogonal mutation.
            orthogonal_mutation_decay (float): Decay factor for orthogonal mutation strength.
            compile_model (bool): Whether to compile the model using torch.compile; may incur overhead.
            dtype (torch.dtype): Data type for the model parameters.
            device (Optional[torch.device]): Device to run the model on. Defaults to CUDA if available, else CPU.
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.criterion = criterion

        self.init_population_size = init_population_size
        self.elite_size = elite_size
        self.selection_method = selection_method
        self.offspring_size = offspring_size
        self.tau = tau
        self.mutation_rate = mutation_rate
        self.mutation_rate_decay = mutation_rate_decay
        self.parallel_mutation_strength = parallel_mutation_strength
        self.parallel_mutation_decay = parallel_mutation_decay
        self.orthogonal_mutation_strength = orthogonal_mutation_strength
        self.orthogonal_mutation_decay = orthogonal_mutation_decay
        self.compile_model = compile_model
        self.dtype = dtype
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.population_size = self.init_population_size
        self.generation: int = 0
        self._offspring_count: int = 0

        self._validate_hyperparameters()

        self.population = Population(
            Individual(
                model=self.model_class(**self.model_kwargs),
                compile=self.compile_model,
                dtype=self.dtype,
                device=self.device,
                name=f"Individual-{i + 1}",
            )
            for i in range(self.population_size)
        )

        print(self)
        self._print_n_parameters()

    def __repr__(self) -> str:
        return (
            f"GPGAEnhanced(model_class={self.model_class.__name__}, "
            f"population_size={self.population_size}, "
            f"elite_size={self.elite_size}, "
            f"selection_method={self.selection_method.name}, "
            f"offspring_size={self.offspring_size}, "
            f"tau={self.tau:.2f}, "
            f"mutation_rate={self.mutation_rate:.2f}, "
            f"mutation_rate_decay={self.mutation_rate_decay:.3f}, "
            f"parallel_mutation_strength={self.parallel_mutation_strength:.2f}, "
            f"parallel_mutation_decay={self.parallel_mutation_decay:.3f}, "
            f"orthogonal_mutation_strength={self.orthogonal_mutation_strength:.2f}, "
            f"orthogonal_mutation_decay={self.orthogonal_mutation_decay:.3f}, "
        )

    def _validate_hyperparameters(self):
        if self.population_size <= 0:
            raise ValueError("population_size must be a positive integer.")
        if self.elite_size < 0:
            raise ValueError("elite_size must be a non-negative integer.")
        if self.selection_method is SelectionMethod.TOURNAMENT:
            print("[!] Warning: Tournament selection is not recommended for GPGA.")
        if self.offspring_size <= 0:
            raise ValueError("offspring_size must be greater than zero.")
        if self.offspring_size + self.elite_size > self.population_size:
            raise ValueError(
                "offspring_size + elite_size must be less than or equal to population_size."
            )
        if not (0 <= self.mutation_rate <= 1):
            raise ValueError("mutation_rate must be between 0 and 1.")
        if (
            self.parallel_mutation_strength <= 0
            or self.orthogonal_mutation_strength <= 0
        ):
            raise ValueError("mutation_strength must be greater than zero.")
        if self.mutation_rate_decay <= 0 or self.mutation_rate_decay > 1:
            raise ValueError("mutation_decay must be between 0 and 1 (inclusive).")
        if self.compile_model:
            print("[!] Compiling the model may incur an additional overhead.")
        if self.device.type == "cpu":
            print("[!] Running on CPU may be slow.")

    def _print_n_parameters(self):
        """
        Print the number of parameters in the model.
        """
        numel = sum(p.numel() for p in self.population[0]._model.parameters())
        if numel < 1_000:
            string = f"{numel}"
        elif numel < 1_000_000:
            string = f"{numel / 1_000:.2f}K"
        elif numel < 1_000_000_000:
            string = f"{numel / 1_000_000:.2f}M"
        elif numel < 1_000_000_000_000:
            string = f"{numel / 1_000_000_000:.2f}B"
        else:
            string = f"{numel / 1_000_000_000_000:.2f}T"
        print(
            f"[i] Number of parameters of the model {self.model_class.__name__}"
            f": {string}"
        )

    def get_best(self) -> Individual:
        """
        Get the individual with the best fitness.
        Returns:
            Individual: The individual with the highest fitness.
        """
        return max(self.population, key=lambda ind: ind.fitness)

    def seed_everything(self, seed: int):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): The seed value to set.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _roulette_wheel_selection(self, n: int = 2) -> list[Individual]:
        """
        Roulette wheel selection of parents based on fitness.

        Args:
            n (int): Number of parents to select. Default is 2.

        Returns:
            list[Individual]: Selected parents for the next generation.
        """
        fitnesses: np.ndarray = np.array(self.population.fitnesses())
        probabilities: np.ndarray = fitnesses / fitnesses.sum()  # Assured to be nonzero
        return [
            self.population[i]
            for i in np.random.choice(len(self.population), size=n, p=probabilities)
        ]

    @deprecated("Tournament selection is not recommended for GPGA.")
    def _tournament_selection(self, n: int = 2, k: int = 24) -> list[Individual]:
        """
        Tournament selection of parents based on fitness.

        Args:
            n (int): Number of parents to select. Default is 2.
            k (int): Number of individuals to select for each tournament. Default is 3.

        Returns:
            list[Individual]: Selected parents for the next generation.
        """
        # Sample k individuals from the population
        tournament_indices: np.ndarray = np.random.choice(
            len(self.population), size=k, replace=False
        )

        # Select the best n individuals from the tournament
        tournament_individuals: list[Individual] = [
            self.population[i] for i in tournament_indices
        ]
        tournament_individuals.sort(key=lambda ind: ind.fitness, reverse=True)
        selected_parents: list[Individual] = tournament_individuals[:n]
        return selected_parents

    @torch.no_grad()
    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        Gradient-Semantic Crossover

        Args:
            parent1 (Individual): First parent individual.
            parent2 (Individual): Second parent individual.

        Returns:
            Individual: Offspring
        """
        params1, params2 = parent1.get_flat_params(), parent2.get_flat_params()
        grad1, grad2 = parent1.get_flat_gradients(), parent2.get_flat_gradients()
        # assert (
        #     grad1 is not None and grad2 is not None
        # ), "Gradients must be computed before crossover."

        grad1_norm: Tensor = grad1.norm() + 1e-8
        grad2_norm: Tensor = grad2.norm() + 1e-8

        # Cosine similarity between gradients
        similarity: Tensor = (grad1 @ grad2) / (grad1_norm * grad2_norm + 1e-8)

        # Inversely proportional to similarity
        # 1 if gradient direction is opposite, 0 if identical, and 0.5 if orthogonal
        selection_pressure: Tensor = (1 - similarity) / 2.0  # [0, 1]

        # Relative grad2 norm to grad1 norm
        mag_ratio_grad2: Tensor = grad2_norm / grad1_norm  # [0, inf]

        # Parent1 dominance: selection pressure * parent2 gradient magnitude ratio
        bias: Tensor = selection_pressure * torch.log(mag_ratio_grad2)  # p*[-inf, inf]

        # Parent1 dominance for gradient-parallel component
        alpha: Tensor = torch.sigmoid(self.tau * bias)
        # print(f"[i] Crossover alpha: {alpha.item():.4f}")

        beta: Tensor = torch.rand_like(
            params1,
            dtype=self.dtype,
            device=self.device,
        )  # Uniform random values in [0, 1)

        parallel1, orthogonal1 = GradientProjection.transform(
            params1, grad1, grad1_norm
        )
        parallel2, orthogonal2 = GradientProjection.transform(
            params2, grad2, grad2_norm
        )

        parallel_offspring: Tensor = alpha * parallel1 + (1 - alpha) * parallel2
        orthogonal_offspring: Tensor = beta * orthogonal1 + (1 - beta) * orthogonal2

        offspring_params: Tensor = GradientProjection.inverse_transform(
            parallel_offspring, orthogonal_offspring
        )

        self._offspring_count += 1
        return Individual(
            model=self.model_class(**self.model_kwargs),
            compile=self.compile_model,
            dtype=self.dtype,
            device=self.device,
            name=f"Offspring-{self._offspring_count}",
        ).set_flat_params(offspring_params)

    @torch.no_grad()
    def _mutate(self, individual: Individual) -> Individual:
        """
        Gradient-Guided Mutation

        Args:
            individual (Individual): The individual to mutate.
        Returns:
            Individual: The mutated individual.
        """
        params: Tensor = individual.get_flat_params()
        grad: Tensor = individual.get_flat_gradients()
        # assert grad is not None, "Gradients must be computed before mutation."

        grad_norm: float = grad.norm().item() + 1e-8
        # print(f"[i] Mutation gradient norm: {grad_norm:.4f}")
        normalized_grad: Tensor = grad / grad_norm

        # Adaptive parallel mutation strength based on gradient norm
        _eps_parallel: Tensor = torch.normal(
            mean=0.0,
            std=self.parallel_mutation_strength / grad_norm,
            size=params.shape,
            dtype=self.dtype,
            device=self.device,
        )
        # Gradient-parallel subspace mutation
        # NOTE: Correct operation is `(_eps_parallel @ normalized_grad) * normalized_grad` for
        #       projection onto the gradient direction, but take this as a generalization of
        #       the scalar `_eps_parallel`.
        delta_parallel: Tensor = _eps_parallel * normalized_grad

        _eps_orthogonal: Tensor = torch.normal(
            mean=0.0,
            std=self.orthogonal_mutation_strength,
            size=params.shape,
            dtype=self.dtype,
            device=self.device,
        )
        # Project noise_orthogonal onto grad direction
        # NOTE: Here, we use the Gram-Schmidt process to ensure that the noise is orthogonal
        #       to the gradient direction, unlike the `delta_parallel`
        delta_orthogonal: Tensor = (
            _eps_orthogonal - (_eps_orthogonal @ normalized_grad) * normalized_grad
        )

        # Use params as param_parallel + param_orthogonal = params
        mutated_params: Tensor = params + delta_parallel + delta_orthogonal
        return individual.set_flat_params(mutated_params)

    def _generate_offspring_population(
        self,
        inputs: Tensor,
        targets: Tensor,
    ) -> Population:
        """
        Generate a new population of offspring individuals based on the current population.

        Args:
            inputs (Tensor): Batch of input data for the model.
            targets (Tensor): Batch of target labels for the model.
        Returns:
            Population: A new population of offspring individuals.
        """
        offspring_population = Population()

        for _ in range(self.offspring_size):
            # if self.selection_method == SelectionMethod.TOURNAMENT:
            #     p1, p2 = self._tournament_selection()
            # elif self.selection_method == SelectionMethod.ROULETTE_WHEEL:
            #     p1, p2 = self._roulette_wheel_selection()
            # else:
            #     raise ValueError(f"Unknown selection method: {self.selection_method}")
            p1, p2 = self._roulette_wheel_selection()

            # Parents must have valid gradients and fitness
            p1.train()
            p1.forward(
                self.criterion,
                inputs,
                targets,
                compute_gradients=True,
                update_fitness=False,
            )
            p2.train()
            p2.forward(
                self.criterion,
                inputs,
                targets,
                compute_gradients=True,
                update_fitness=False,
            )

            offspring: Individual = self._crossover(p1, p2)
            offspring.train()

            if np.random.rand() < self.mutation_rate:
                # Gradient is required for mutation
                offspring.forward(
                    self.criterion,
                    inputs,
                    targets,
                    compute_gradients=True,
                    update_fitness=False,
                )
                offspring = self._mutate(offspring)

            # self._fix_normalization_layers(offspring, p1, p2)
            offspring_population.append(offspring)

        return offspring_population

    def step(self, inputs: Tensor, targets: Tensor):
        """
        Perform one step of the GPGA algorithm, which includes evaluating fitness,
        selecting elites, generating offspring, and applying mutation and crossover.

        Args:
            inputs (Tensor): Batch of input data for the model.
            targets (Tensor): Batch of target labels for the model.
        """
        parent_population: Population = self.population

        # Invalidate fitness for all individuals for the next step
        # This is necessary to ensure that fitness is recomputed
        for individual in parent_population:
            individual._is_fitness_valid = False

        inputs = inputs.to(self.device, dtype=self.dtype)
        targets = targets.to(self.device)
        parent_population.compute_fitnesses(self.criterion, inputs, targets)

        offspring_population: Population = self._generate_offspring_population(
            inputs, targets
        )
        offspring_population.compute_fitnesses(self.criterion, inputs, targets)

        entire_population = Population(parent_population + offspring_population)
        # Parent and offspring populations already have fitness computed
        entire_population.sort_by_fitness()

        elite_population = Population(entire_population[: self.elite_size])
        non_elite_population = Population(entire_population[self.elite_size :])

        non_elite_indices: np.ndarray = np.random.choice(
            len(non_elite_population),
            size=self.population_size - self.elite_size,
            replace=False,
        )

        new_population = Population(elite_population)
        new_population.extend(non_elite_population[i] for i in non_elite_indices)

        self.generation += 1
        self.mutation_rate *= self.mutation_rate_decay
        self.parallel_mutation_strength *= self.parallel_mutation_decay
        self.orthogonal_mutation_strength *= self.orthogonal_mutation_decay
        self.population = new_population
