from pyseds.gm import GaussianMixture
import numpy
import scipy.optimize
from sklearn.mixture import GMM


def _to_optimization_vector(weights, means, covars):
    chol_decomps = numpy.array([numpy.linalg.cholesky(covar) for covar in covars])
    return numpy.hstack((numpy.log(weights), means.flatten(), chol_decomps.flatten()))


def _from_optimization_vector(theta, weights, means, covars):
    assert theta.size == (weights.size + means.size + covars.size)
    weights[:] = numpy.exp(theta[:weights.size])
    means[:,:] = theta[weights.size:-covars.size].reshape(means.shape)
    covars[:, :, :] = numpy.array([chol_decomp.dot(chol_decomp.T)
                                   for chol_decomp in theta[-covars.size:]
                                   .reshape(covars.shape)])


class SEDS(object):
    def __init__(self, attractor, n_components, debug=True):
        self.attractor = numpy.asarray(attractor)
        self.n_components = n_components
        self.n_task_dims = len(attractor)
        self.debug = debug

    def imitate(self, S, Sd):
        """Imitate demonstrations.

        Parameters
        ----------
        S: array-like, shape = (n_demonstrations, n_task_dims, n_steps)
            Positions
        Sd: array-like, shape = (n_demonstrations, n_task_dims, n_steps)
            Velocities
        """
        self.S = S
        self.Sd = Sd

        weights, means, covars = self.__initial_parameters(S, Sd)

        # Alternative likelihood optimization
        # The parameters of the SEDS are represented by
        # p1, ..., pK, mu1, ..., muK, L1, ..., LK,
        # where p1, ..., pK are the log-priors, i.e. pk = ln pik,
        # mu1, ..., muK are the means and L1, ..., LK are the Cholesky
        # decompositions of the covariance matrices (these do always exist
        # since covariance matrices are positive definite).
        theta0 = _to_optimization_vector(weights, means, covars)

        if self.debug:
            # Test packing / unpacking
            w = numpy.copy(weights)
            m = numpy.copy(means)
            c = numpy.copy(covars)
            _from_optimization_vector(theta0, w, m, c)
            assert (w - weights < 1e-5).all()
            assert (m - means < 1e-5).all()
            assert (c - covars < 1e-5).all()

        # TODO conditions, cost function
        def f(theta):
            _from_optimization_vector(theta, weights, means, covars)
            return self.__cost_deriv(weights, means, covars)[0]
        def fp(theta):
            _from_optimization_vector(theta, weights, means, covars)
            return self.__cost_deriv(weights, means, covars)[1]
        scipy.optimize.fmin_slsqp(f, fprime=fp, x0=theta0, eqcons=[], ieqcons=[])

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

    def __cost_deriv(self, weights, means, covars):
        gm = GaussianMixture(weights, means, covars)
        J = -numpy.sum([numpy.log(gm.pdf_ssd(self.S[d, :, t], self.Sd[d, :, t]))
                        for t in range(self.S.shape[2])
                        for d in range(self.S.shape[0])]) / self.S.shape[0]
        print "J = %f" % J
        # TODO implement
        weights_deriv = numpy.zeros_like(weights)
        means_deriv = numpy.zeros_like(means)
        covars_deriv = numpy.zeros_like(covars)
        return 0.0, numpy.hstack((weights_deriv, means_deriv.flatten(), covars_deriv.flatten()))

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
