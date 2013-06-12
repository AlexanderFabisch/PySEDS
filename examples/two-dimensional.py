import numpy


class SEDS(object):
    def __init__(self):
        pass

    def step(self, s):
        """Compute next desired velocity from current state.
        
        Parameters
        ----------
        s: array-like, shape (num_task_dimensions)
            current state

        Returns
        -------
        sd: array-like, shape (num_task_dimensions)
            desired next state
        sdd: array-like, shape (num_task_dimensions)
            desired next velocity
        """
        sd = numpy.zeros_like(s)
        sdd = numpy.zeros_like(s)
        return sd, sdd


if __name__ == "__main__":
    seds = SEDS()
