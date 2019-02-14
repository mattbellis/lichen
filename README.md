[![DOI](https://zenodo.org/badge/12589/mattbellis/lichen.svg)](http://dx.doi.org/10.5281/zenodo.17256)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mattbellis/lichen/master)


################################################################################
# README
################################################################################

################################################################################
# INSTALL
################################################################################

    python setup.py install

You may need to run this as root, depending on your permissions.

    sudo python setup.py install

################################################################################
# Hello world!
################################################################################

    import numpy as np
    import matplotlib.pylab as plt

    import lichen as lch

    x = np.random.normal(5,1,1000)
    h = lch.hist(x,bins=50)
    plt.show()


