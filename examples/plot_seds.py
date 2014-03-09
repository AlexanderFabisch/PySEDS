import matplotlib.pyplot as plt
from lasa import load_demos


X, Xd, Xdd, dt, name = load_demos(4)



for demo_idx in range(X.shape[2]):
    plt.plot(X[0, :, demo_idx], X[1, :, demo_idx], label="%d" % demo_idx)
plt.legend()
plt.show()
