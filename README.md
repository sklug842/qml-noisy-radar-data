# QML: Noisy Radar Data

Code accompanying the manuscript

> Stefan Klug, Stefan Pickl, Maximilian Moll  
> *Comparing quantum and classical machine learning for radar-based drone classification: a like-for-like benchmark on noisy data*  
> (submitted to *Quantum Machine Intelligence*).

## Overview

This repository contains code to reproduce the experiments in the paper, including:

- Simulation of radar micro-Doppler data for two multirotor UAVs in the X-band using the Martin–Mulgrew signal model.
- Generation of three data regimes: noiseless, additive white Gaussian noise (AWGN), and colored AR(1) complex Gaussian noise (cAR(1)–GN).
- Implementations of variational quantum classifiers (VQCs) with
  - two circuit families (Version 1 and Version 2),
  - angle and amplitude encodings, and
  - optional Quantum Fourier Transform (QFT) layers,
  trained with both gradient-based (Adam) and gradient-free (Asexual Reproduction Optimization, ARO) optimizers.
- Classical feed-forward neural networks:
  - structure-matched baselines mirroring the QVC depth/width, and
  - tuned architectures obtained via Optuna-based hyperparameter search.
- Experiment scripts to train all models, run 10 independent restarts per configuration, and aggregate metrics (accuracy, adjusted mean accuracy, and Mann–Whitney U tests).
