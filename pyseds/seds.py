from pyseds.gm import GaussianMixture
import numpy
import scipy.optimize
from sklearn.mixture import GMM


def _to_vector(weights, means, covars):
    # TODO cholesky
    return numpy.hstack((numpy.log(weights).flatten(), means.flatten(), covars.flatten()))


def _from_optimization_vector(theta, weights, means, covars):
    # TODO
    weights = theta[:weights.size]
    means = theta[weights.size:-covars.size].reshape(means.shape)
    covars = theta[-covars.size:].reshape(covars.shape)


class SEDS(object):
    def __init__(self, attractor, n_components):
        self.attractor = numpy.asarray(attractor)
        self.n_components = n_components
        self.n_task_dims = len(attractor)

    def imitate(self, S, Sd):
        weights, means, covars = self.__initial_parameters(S, Sd)

        # Alternative likelihood optimization
        # The parameters of the SEDS are represented by
        # p1, ..., pK, mu1, ..., muK, L1, ..., LK,
        # where p1, ..., pK are the log-priors, i.e. pk = ln pik,
        # mu1, ..., muK are the means and L1, ..., LK are the Cholesky
        # decompositions of the covariance matrices (these do always exist
        # since covariance matrices are positive definite).
        cost = lambda theta: self.__cost(theta)
        cost_deriv = lambda theta: self.__cost_deriv(theta)
        theta0 = _to_optimization_vector(weights, means, covars)
        # TODO conditions, cost function
        #scipy.optimize.fmin_slsqp(func=self.cost, fprime=self.cost_deriv,
        #                          theta0=theta0, eqcons=[], ieqcons=[])

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
