from pyseds.seds import SEDS
import numpy
import pylab


def generate_demonstrations(delta_t):
    n_demos = 1
    n_task_dims = 2
    n_steps = 10

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
            #print a, v, s

    return S, Sd


if __name__ == "__main__":
    delta_t = 0.1
    S, Sd = generate_demonstrations(delta_t=delta_t)

    seds = SEDS(S[0, :, -1], 10)
    seds.imitate(S, Sd)

    A = numpy.zeros_like(S)
    s = 1*S[0, :, 0]
    for t in range(S.shape[2]-1):
        A[0, :, t] = s
        sd = seds.step(s)
        s += sd * delta_t

    pylab.plot(S[0, 0, :], S[0, 1, :], "o")
    pylab.plot(A[0, 0, :], A[0, 1, :], "o")
    pylab.show()
