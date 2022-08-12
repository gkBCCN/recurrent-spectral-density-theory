# recurrent-spectral-density-theory
This repository is a supplement to

<a href=https://journals.aps.org/pre/abstract/10.1103/PhysRevE.105.044411>
Knoll G. & Lindner B. (2022) <i>Information transmission in recurrent networks: Consequences of network noise for synchronous and asynchronous signal encoding.</i> Phys. Rev. E, 105(4):044411
</a>

It provides the code to find the analytical results for the spectral densities of recurrent spiking network outputs: all-spike, activity, and synchrony.

First select a J value (0 or 0.01) in param_file.py.

Then calculate all necessary spectral densities of interest using main.py

Three plots from the paper (Figures 3-5) can be generated by running their designated scripts.  The simulation data used in the paper has been provided for both J values for all three plots.
