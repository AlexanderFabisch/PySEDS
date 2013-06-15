from pyseds.seds import SEDS
import numpy
import pylab


def generate_demonstrations():
    n_demos = 1
    n_task_dims = 2
    n_steps = 100
    delta_t = 0.01

    Sd = numpy.ndarray((n_demos, n_task_dims, n_steps))
    S = numpy.ndarray((n_demos, n_task_dims, n_steps))
    for d in range(n_demos):
        a = numpy.zeros(n_task_dims)
        v = numpy.zeros(n_task_dims)
        s = numpy.zeros(n_task_dims)
        for t in range(0, n_steps):
            s += v*delta_t
            v += a*delta_t
            a = numpy.array([10.0*(t/(n_steps-2.0)-0.5)**3,
                             numpy.sin(2*numpy.pi*t/(n_steps-2.0))])
            Sd[d, :, t] = v
            S[d, :, t] = s
            print a, v, s

    return S, Sd


if __name__ == "__main__":
    S, Sd = generate_demonstrations()
    pylab.plot(S[0, 0, :], S[0, 1, :])
    pylab.show()

    seds = SEDS(2)
    seds.imitate(S, Sd)
