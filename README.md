# QML: Noisy Radar Data

Code accompanying the manuscript:

**Stefan Klug, Stefan Pickl, Maximilian Moll**  
*Comparing quantum and classical machine learning for radar-based drone classification: a like-for-like benchmark on noisy data*  
(submitted to *Quantum Machine Intelligence*; preprint available at [Research Square](https://www.researchsquare.com/article/rs-8437365/v1), DOI: [10.21203/rs.3.rs-8437365/v1](https://doi.org/10.21203/rs.3.rs-8437365/v1))

## Overview

This repository contains code used to reproduce the experiments reported in the manuscript, including:

- Simulation of radar micro-Doppler data for two multirotor UAVs in the X-band using the Martin–Mulgrew signal model.
- Generation of independent training, validation, and test sets, with model and hyperparameter selection performed exclusively on validation data.
- Generation of three data regimes: noiseless, additive white Gaussian noise (AWGN), and colored AR(1) complex Gaussian noise (cAR(1)–GN).
- Implementations of quantum variational classifiers (QVCs) with  
  - two circuit families (Version 1 and Version 2),  
  - angle and amplitude encodings, and  
  - classical Discrete Fourier Transform (DFT) or Quantum Fourier Transform (QFT) processing,  
  trained with both gradient-based (Adam) and gradient-free (Asexual Reproduction Optimization, ARO) optimizers.
- Classical feed-forward neural networks, including  
  - structure-matched baselines mirroring QVC depth and width, and  
  - tuned architectures obtained via Optuna-based hyperparameter search.
- Experiment scripts to train and evaluate the models over 10 independent runs per configuration and aggregate performance statistics.

## Citation

If you use this repository in your research, please cite both the code and the accompanying manuscript.

### Repository
```bibtex
@misc{Klug2026QMLNoisyRadarData,
  author       = {Klug, Stefan and Pickl, Stefan and Moll, Maximilian},
  title        = {QML: Noisy Radar Data},
  year         = {2026},
  howpublished = {\url{https://github.com/sklug842/qml-noisy-radar-data}},
  note         = {GitHub repository}
}

@article{Klug2026RadarBenchmark,
  author  = {Klug, Stefan and Pickl, Stefan and Moll, Maximilian},
  title   = {Comparing quantum and classical machine learning for radar-based drone classification: a like-for-like benchmark on noisy data},
  journal = {Research Square},
  year    = {2026},
  doi     = {10.21203/rs.3.rs-8437365/v1},
  url     = {https://www.researchsquare.com/article/rs-8437365/v1}
}
