from pyseds.seds import SEDS
import numpy


def generate_demonstrations(n_demos, n_task_dims, n_steps):
    S = numpy.ndarray((n_demos, n_task_dims, n_steps))
    Sd = numpy.ndarray((n_demos, n_task_dims, n_steps))
    for d in range(n_demos):
        pass
        # TODO
    return S, Sd


if __name__ == "__main__":
    S, Sd = generate_demonstrations(2, 2, 10)

    seds = SEDS(2)
    seds.imitate(S, Sd)
