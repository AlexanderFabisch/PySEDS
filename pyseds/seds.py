from pyseds.gm import GaussianMixture
import numpy
import scipy.optimize
from sklearn.mixture import GMM


def _to_vector(weights, means, covars):
    return numpy.hstack((weights.flatten(), means.flatten(), covars.flatten()))


def _from_vector(x, weights, means, covars):
    weights = x[:weights.size]
    means = x[weights.size:-covars.size].reshape(means.shape)
    covars = x[-covars.size:].reshape(covars.shape)


class SEDS(object):
    def __init__(self, attractor, n_components):
        self.attractor = numpy.asarray(attractor)
        self.n_components = n_components
        self.n_task_dims = len(attractor)

    def imitate(self, S, Sd):
        weights, means, covars = self.__initial_parameters(S, Sd)

        # TODO conditions
        cost = lambda x: self.__cost(x)
        cost_deriv = lambda x: self.__cost_deriv(x)
        x0 = _to_vector(weights, means, covars)
        scipy.optimize.fmin_slsqp(func=cost, fprime=cost_deriv, x0=x0,
                                  eqcons=[], ieqcons=[])

        self.gm = GaussianMixture(weights, means, covars)

    def __initial_parameters(self, S, Sd):
        S = numpy.concatenate(S.swapaxes(1, 2), axis=0).T
        Sd = numpy.concatenate(Sd.swapaxes(1, 2), axis=0).T
        X = numpy.concatenate((S, Sd)).T

        gmm = GMM(self.n_components, covariance_type="full")
        gmm.fit(X)

        n_task_dims = self.n_task_dims
        weights = gmm.weights_
        means = numpy.ndarray(gmm.means_.shape)
        means[:, :n_task_dims] = gmm.means_[:, :n_task_dims]
        # Transform covariances such that they satisfy the optimization
        # constraints:
        covars = numpy.ndarray(gmm.covars_.shape)
        eye = numpy.tile(numpy.eye(n_task_dims), (self.n_components, 1, 1))
        covars[:, :n_task_dims, :n_task_dims] = \
            eye * numpy.abs(gmm.covars_[:, :n_task_dims, :n_task_dims])
        covars[:, n_task_dims:, n_task_dims:] = \
            eye * numpy.abs(gmm.covars_[:, n_task_dims:, n_task_dims:])
        covars[:, n_task_dims:, :n_task_dims] = \
            -eye * numpy.abs(gmm.covars_[:, n_task_dims:, :n_task_dims])
        covars[:, :n_task_dims, n_task_dims:] = \
            -eye * numpy.abs(gmm.covars_[:, :n_task_dims, n_task_dims:])
        # Compute the rest of the means by solving the optimization constraint
        for k in range(self.n_components):
            means[k, n_task_dims:] = covars[k, n_task_dims:, :n_task_dims].dot(
                numpy.linalg.inv(covars[k, :n_task_dims, :n_task_dims]).dot(
                means[k, :n_task_dims] - self.attractor))

        return weights, means, covars

    def __cost(self, x):
        # TODO log-likelihood
        return 0.0

    def __cost_deriv(self, x):
        # TODO derivatives
        return None

    def step(self, s):
        """Compute next desired velocity from current state.
        
        Parameters
        ----------
        s: array-like, shape (n_task_dims)
            current state

        Returns
        -------
        sd: array-like, shape (n_task_dims)
            desired next velocity
        """
        return self.gm.next(s)
