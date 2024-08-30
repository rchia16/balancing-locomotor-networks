# balancing-locomotor-networks
Publication: "Balancing Excitation and Inhibition in the Locomotor Spinal Circuits" by R. Chia and C. Lin

This repository provides the python code for describing the presynaptically inhibited ankle flexor spiking neural network (SNN) model. Biophysical parameters were constrained to experiment with reported values. The 'flex_experiment.py' script can be run to reproduce publication figures over different conditions. A high-performance computer cluster is recommended to run the simulations.

## Setup
The code was written using Python 3.10 and the Brian2 simulator. Please refer to the Brian2 documentation for [installation](https://brian2.readthedocs.io/en/stable/introduction/install.html). 

## Additional Files
The recruitment curve data and mean afferent firing rates for ankle flexor/extensor locomotor activity were adopted from [Formento et al., 2018](https://github.com/FormentoEmanuele/MuscleSpindleCircuitsModel).
