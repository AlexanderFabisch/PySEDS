import numpy


class GaussianMixture(object):
    def __init__(self, weights, means, covariances):
        self.priors = numpy.asarray(weights)
        self.means = numpy.asarray(means)
        self.covariances = numpy.asarray(covariances)

        assert self.priors.shape[0] == self.covariances.shape[0]
        assert self.means.shape[1] == self.covariances.shape[1]
        assert self.means.shape[1] == self.covariances.shape[2]

        self.n_components = len(weights)
        self.n_task_dims = self.means.shape[1] / 2
        self.gaussians = [Gaussian(self.means[k], self.covariances[k])
                          for k in range(self.n_components)]

    def next(self, s):
        h = self.priors * numpy.array([self.gaussians[k].pdf_sd(s)
                                       for k in range(self.n_components)])
        h /= h.sum()
        sd = numpy.zeros(self.n_task_dims)
        for k in range(self.n_components):
            sd += h[k] * self.gaussians[k].mean_sd + \
                self.gaussians[k].covariance_sds.dot(
                numpy.linalg.inv(self.gaussians[k].covariance_ss
                                 ).dot(s - self.gaussians[k].mean_s))
        return sd


class Gaussian(object):
    def __init__(self, mean, covariance):
        self.mean = numpy.asarray(mean)
        self.covariance = numpy.asarray(covariance)

        self.n_task_dims = self.mean.shape[0] / 2
        # Extract covariance parts
        self.covariance_ss = self.covariance[:self.n_task_dims,
                                             :self.n_task_dims]
        self.covariance_sdsd = self.covariance[self.n_task_dims:,
                                               self.n_task_dims:]
        self.covariance_sds = self.covariance[self.n_task_dims:,
                                              :self.n_task_dims]
        covariance_ssd = self.covariance[:self.n_task_dims,
                                         self.n_task_dims:]
        assert (self.covariance_sds == covariance_ssd).all()
        # Extract mean parts
        self.mean_s = self.mean[:self.n_task_dims]
        self.mean_sd = self.mean[self.n_task_dims:]

    def pdf_ssd(self, s, sd):
        return mvn_pdf(numpy.hstack((s, sd)), self.mean, self.covariance)

    def pdf_sd(self, s):
        return mvn_pdf(numpy.asarray(s), self.mean_s, self.covariance_ss)


def mvn_pdf(x, mean, covariance):
    d = x.shape[0]
    covariance_inv = numpy.linalg.inv(covariance)
    x_mean = x - mean
    return 1 / numpy.sqrt((2*numpy.pi)**d * numpy.linalg.det(covariance)) * \
        numpy.exp(-0.5 * x_mean.T.dot(covariance_inv).dot(x_mean))


if __name__ == "__main__":
    gaussian = Gaussian([0, 0], [[1, 0], [0, 1]])
    numpy.testing.assert_allclose(gaussian.pdf_ssd([0], [0]), 0.159154943092)
    numpy.testing.assert_allclose(gaussian.pdf_sd([0]), 0.398942280401)
