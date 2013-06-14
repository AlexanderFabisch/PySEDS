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


def mvn_pdf(x, mean, covariance):
    d = x.shape[0]
    covariance_inv = numpy.linalg.inv(covariance)
    x_mean = x - mean
    return 1 / numpy.sqrt((2*numpy.pi)**d * numpy.linalg.det(covariance)) * \
        numpy.exp(-0.5 * x_mean.T.dot(covariance_inv).dot(x_mean))


class Gaussian(object):
    def __init__(self, mean, covariance):
        self.mean = numpy.asarray(mean)
        self.covariance = numpy.asarray(covariance)

        self.num_task_dimensions = self.mean.shape[0] / 2
        # Extract covariance parts
        self.covariance_ss = self.covariance[:self.num_task_dimensions,
                                             :self.num_task_dimensions]
        self.covariance_sdsd = self.covariance[self.num_task_dimensions:,
                                               self.num_task_dimensions:]
        self.covariance_sds = self.covariance[self.num_task_dimensions:,
                                              :self.num_task_dimensions]
        covariance_ssd = self.covariance[:self.num_task_dimensions,
                                         self.num_task_dimensions:]
        assert (self.covariance_sds == covariance_ssd).all()
        # Extract mean parts
        self.mean_s = self.mean[:self.num_task_dimensions]
        self.mean_sd = self.mean[self.num_task_dimensions:]

    def pdf_ssd(self, s, sd):
        return mvn_pdf(numpy.hstack((s, sd)), self.mean, self.covariance)

    def pdf_sd(self, s):
        return mvn_pdf(numpy.asarray(s), self.mean_s, self.covariance_ss)


class GaussianMixture(object):
    def __init__(self, priors, means, covariances):
        self.priors = numpy.asarray(priors)
        self.means = numpy.asarray(means)
        self.covariances = numpy.asarray(covariances)

        assert self.priors.shape[0] == self.covariances.shape[0]
        assert self.means.shape[1] == self.covariances.shape[1]
        assert self.means.shape[1] == self.covariances.shape[2]

        self.num_gaussians = len(priors)
        self.num_task_dimensions = self.means.shape[1] / 2
        self.gaussians = [Gaussian(self.means[k], self.covariances[k])
                          for k in range(self.num_gaussians)]

    def next(self, s):
        h = self.priors * numpy.array([self.gaussians[k].pdf_sd(s)
                                       for k in range(self.num_gaussians)])
        h /= h.sum()
        sd = numpy.zeros(self.num_task_dimensions)
        for k in range(self.num_gaussians):
            sd += h[k] * self.gaussians[k].mean_sd + \
                self.gaussians[k].covariance_sds.dot(
                numpy.linalg.inv(self.gaussians[k].covariance_ss
                                 ).dot(s - self.gaussians[k].mean_s))
        return sd


if __name__ == "__main__":
    seds = SEDS()
    gaussian = Gaussian([0, 0], [[1, 0], [0, 1]])
    numpy.testing.assert_allclose(gaussian.pdf_ssd([0], [0]), 0.159154943092)
    numpy.testing.assert_allclose(gaussian.pdf_sd([0]), 0.398942280401)
    gmm = GaussianMixture([0.5, 0.5],
                          [[0, 0], [1, 1]],
                          [[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    print gmm.next([-2.0])
