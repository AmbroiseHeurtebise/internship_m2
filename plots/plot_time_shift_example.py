import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


if __name__ == '__main__':
    s1 = np.arange(-4, 4, 0.1)
    t = np.linspace(-0.2, 0.5, len(s1))
    s1 = norm.pdf(s1, 0, 0.5)
    s2 = np.roll(s1, 2)

    plt.plot(t, s1, label="W^i X^i")
    plt.plot(t, s2, label="(W^i X^i)(1)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig("figures/time_shift_example.pdf")
    plt.show()
