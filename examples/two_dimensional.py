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


class Gaussian(object):
    def __init__(self, mean, covariance):
        self.mean = numpy.asarray(mean)
        self.covariance = numpy.asarray(covariance)
        self.num_task_dimensions = self.mean.shape[0] / 2

    def propability_density(self, s, sd):
        ssd = numpy.hstack((s, sd))
        ssd_mean = ssd - self.mean
        covariance_inv = numpy.linalg.inv(self.covariance)
        return 1 / numpy.sqrt((2*numpy.pi)**(2*self.num_task_dimensions) *
                              numpy.linalg.det(self.covariance)) * \
            numpy.exp(-0.5 * ssd_mean.T.dot(covariance_inv).dot(ssd_mean))


if __name__ == "__main__":
    seds = SEDS()
    gaussian = Gaussian([0, 0], [[1, 0], [0, 1]])
    print gaussian.propability_density([0], [0])
