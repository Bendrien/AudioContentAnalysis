import numpy as np
import scipy as sp
from scipy import special, optimize
import matplotlib.pyplot as plt

def main():
    x = sp.linspace(0, 10, 5000);
    plt.plot(np.sin(x));
    plt.show();


if(__name__ == '__main__'):
    main()
