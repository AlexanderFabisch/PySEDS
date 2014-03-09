import matplotlib.pyplot as plt
from lasa import load_demos
from pyseds.seds import SEDS


X, Xd, Xdd, dt, name = load_demos(4)

seds = SEDS(attractor=X[:, -1, :].mean(axis=0), n_components=50, verbose=True)
S = X.transpose([2, 0, 1])
Sd = X.transpose([2, 0, 1])
seds.imitate(S, Sd)

for demo_idx in range(X.shape[2]):
    plt.plot(X[0, :, demo_idx], X[1, :, demo_idx], label="%d" % demo_idx)
plt.legend()
plt.show()
