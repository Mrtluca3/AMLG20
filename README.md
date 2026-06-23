This is the folder for Group G20 of AML course
## Project Description:

This project implements a hybrid classical-quantum pipeline for model-independent anomaly
detection in high-energy physics, benchmarked on the LHC Olympics 2020 dataset. Following
the strategy proposed by Belis et al. (2024), the workflow aims to detect potential Beyond
the Standard Model (BSM) phenomena without assuming a specific signal hypothesis a priori.

The pipeline combines classical and quantum unsupervised machine learning methods:
1. Preprocessing: High-dimensional jet features are reconstructed from collision events
   using the FastJet library with the anti-kt clustering algorithm. For each event, the highest energy jet is selected.
2. Dimensionality Reduction: These preprocessed particle kinematic features are then
   compressed into a lower-dimensional latent space using a classical Autoencoder (AE).
3. Quantum Kernel Anomaly Detection: An unsupervised Quantum Kernel Machine is trained
   exclusively on Standard Model (QCD) background events within the latent space to map
   typical interactions and isolate rare or unusual BSM processes.

The model's discrimination capability is evaluated using **ROC curves**, **AUC scores**, and
**background rejection metrics**, and is systematically benchmarked against classical baselines
such as Gaussian Mixture Model, K-Means, and reconstruction-loss anomaly scoring.


The primary objective is to deliver a robust, well-documented proof of concept and a
simplified numerical reproduction of the reference study rather than a large-scale hardware
implementation.

## Bibliographic References:
- V. Belis et al., "Quantum anomaly detection in the latent space of proton collision
  events at the LHC," Communications Physics 7, 334 (2024).
- G. Kasieczka et al., "The LHC Olympics 2020: a community challenge for anomaly
  detection in high energy physics," Eur. Phys. J. C 81, 119 (2021).

## Contact:
- Margherita Rossi rossi.2012842@studenti.uniroma1.it
- Floriana Sapio sapio.2017565@studenti.uniroma1.it
- Johnathan Boccia boccia.2022157@studenti.uniroma1.it
- Luca Martini martini.2062769@studenti.uniroma1.it

## Dependencies:
- Data manipulation:
numpy
pandas
- Visualization:
matplotlib
plotly
seaborn

- Machine Learning:
scikit-learn

- Deep Learning:
torch
torchsummary

- Quantum Computing:
qiskit[visualization]
qiskit-machine-learning
qiskit-algorithms

- Physics / Jet clustering:
fastjet

- Utilities:
gdown
pickle  
## Project Notebooks
You can view the notebooks also using the links below:
* [Open in Colab](https://colab.research.google.com/drive/1VRQKZ9OqmHsgNdviqTcbAH-jXenid47i?usp=sharing)
