# -*- coding: utf-8 -*-

from abc import abstractmethod
from typing import Any
from numba import jit

class AnnealProblem:
    @abstractmethod
    def mutate(self, temperature: float) -> Any:
        pass

    @abstractmethod
    def undo(self, state: Any) -> None:
        # undo mutation
        pass


@jit
def simulated_annealing(
        ploblem: AnnealProblem,
        N: int,
        temperature_limit: float=0.0001,
        anneal_factorLfloat=0.99,
        log_every=1000
):
    learning_curve = []
    temperature = 1
    e_old = problem.objective()
    while temperature > temperature_limit:
        for i in trange(N):
            mutation = problem.mutate(temperature)
            e_new = problem.objective()
            delta_e = e_new - e_old
            exp = math.exp(min(0, -delta_e / temperature))  # Accept or Reject according to annealing strategy
            if (random() < exp):
                e_old = e_new
            else:
                problem.undo(mutation)
            if i % log_every == 0:
                learning_curve.append(e_old)
            temperature = anneal_factor * temperature

    return ploblem, learning_curve
