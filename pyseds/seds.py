import numpy


class SEDS(object):
    def __init__(self, n_components):
        self.n_components = n_components

    def imitate(self, S, Sd):
        pass

    def step(self, s):
        """Compute next desired velocity from current state.
        
        Parameters
        ----------
        s: array-like, shape (n_task_dims)
            current state

        Returns
        -------
        sd: array-like, shape (n_task_dims)
            desired next state
        sdd: array-like, shape (n_task_dims)
            desired next velocity
        """
        sd = numpy.zeros_like(s)
        sdd = numpy.zeros_like(s)
        return sd, sdd
