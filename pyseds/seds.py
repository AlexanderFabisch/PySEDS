from gmr import GMM
import numpy
import scipy.optimize


def _to_optimization_vector(priors, means, covars):
    chol_decomps = numpy.array([numpy.linalg.cholesky(covar)
                                for covar in covars])
    return numpy.hstack((numpy.log(priors), means.flatten(),
                         chol_decomps.flatten()))


def _from_optimization_vector(x, priors, means, covars):
    assert x.size == (priors.size + means.size + covars.size)
    priors[:] = numpy.exp(x[:priors.size])
    means[:,:] = x[priors.size:-covars.size].reshape(means.shape)
    covars[:, :, :] = numpy.array([chol_decomp.dot(chol_decomp.T)
                                   for chol_decomp in x[-covars.size:]
                                   .reshape(covars.shape)])


class SEDS(object):
    """Stable estimators of dynamical systems."""
    def __init__(self, attractor, n_components, verbose=True):
        self.attractor = numpy.asarray(attractor)
        self.n_components = n_components
        self.n_task_dims = len(attractor)
        self.verbose = verbose

    def imitate(self, X, Xd):
        """Imitate demonstrations.

        Parameters
        ----------
        X: array-like, shape = (n_demonstrations, n_task_dims, n_steps)
            Positions
        Xd: array-like, shape = (n_demonstrations, n_task_dims, n_steps)
            Velocities
        """
        self.S = X
        self.Sd = Xd

        priors, means, covars = self.__initial_parameters(self.S, self.Sd)

        # Alternative likelihood optimization
        # The parameters of the SEDS are represented by
        # p1, ..., pK, mu1, ..., muK, L1, ..., LK,
        # where p1, ..., pK are the log-priors, i.e. pk = ln pik,
        # mu1, ..., muK are the means and L1, ..., LK are the Cholesky
        # decompositions of the covariance matrices (these do always exist
        # since covariance matrices are positive definite).
        x0 = _to_optimization_vector(priors, means, covars)

        if self.verbose:
            # Test packing / unpacking
            p = numpy.copy(priors)
            m = numpy.copy(means)
            c = numpy.copy(covars)
            _from_optimization_vector(x0, p, m, c)
            assert (p - priors < 1e-5).all()
            assert (m - means < 1e-5).all()
            assert (c - covars < 1e-5).all()

        # TODO conditions, cost function
        def f(theta):
            _from_optimization_vector(theta, priors, means, covars)
            return self.__cost_deriv(priors, means, covars)[0]
        def fp(theta):
            _from_optimization_vector(theta, priors, means, covars)
            return self.__cost_deriv(priors, means, covars)[1]
        scipy.optimize.fmin_slsqp(f, fprime=fp, x0=theta0, eqcons=[], ieqcons=[])

        self.gm = GaussianMixture(priors, means, covars)

    def __initial_parameters(self, S, Sd):
        S = numpy.concatenate(S.swapaxes(1, 2), axis=0).T
        Sd = numpy.concatenate(Sd.swapaxes(1, 2), axis=0).T
        X = numpy.concatenate((S, Sd)).T

        gmm = GMM(self.n_components)
        gmm.from_samples(X)
        if self.verbose:
            print("Estimated GMM")

        n_task_dims = self.n_task_dims
        priors = gmm.priors
        means = numpy.ndarray(gmm.means.shape)
        means[:, :n_task_dims] = gmm.means[:, :n_task_dims]
        # Transform covariances such that they satisfy the optimization
        # constraints:
        covars = numpy.ndarray(gmm.covariances.shape)
        eye = numpy.tile(numpy.eye(n_task_dims), (self.n_components, 1, 1))
        covars[:, :n_task_dims, :n_task_dims] = \
            eye * numpy.abs(gmm.covariances[:, :n_task_dims, :n_task_dims])
        covars[:, n_task_dims:, n_task_dims:] = \
            eye * numpy.abs(gmm.covariances[:, n_task_dims:, n_task_dims:])
        covars[:, n_task_dims:, :n_task_dims] = \
            -eye * numpy.abs(gmm.covariances[:, n_task_dims:, :n_task_dims])
        covars[:, :n_task_dims, n_task_dims:] = \
            -eye * numpy.abs(gmm.covariances[:, :n_task_dims, n_task_dims:])
        # Compute the rest of the means by solving the optimization constraint
        for k in range(self.n_components):
            means[k, n_task_dims:] = covars[k, n_task_dims:, :n_task_dims].dot(
                numpy.linalg.inv(covars[k, :n_task_dims, :n_task_dims]).dot(
                means[k, :n_task_dims] - self.attractor))

        return priors, means, covars

    def __cost_deriv(self, priors, means, covars):
        gm = GaussianMixture(priors, means, covars)
        J = -numpy.sum([numpy.log(gm.pdf_ssd(self.S[d, :, t], self.Sd[d, :, t]))
                        for t in range(self.S.shape[2])
                        for d in range(self.S.shape[0])]) / self.S.shape[0]
        if self.verbose:
            print("J = %f" % J)
        # TODO implement
        weights_deriv = numpy.zeros_like(priors)
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
