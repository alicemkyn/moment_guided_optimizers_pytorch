# Moment-Guided Optimizers for Noisy Deep Learning  
© 2025 Alicem Koyun  

This repository provides the complete **research artifact** for the paper:  
**“Moment-Guided Optimizers for Noisy Deep Learning.”**

It includes full PyTorch implementations of adaptive optimizers (Adam, AdaGrad, RMSProp, AdaMax),  
training scripts for MNIST, CIFAR-10, and IMDB datasets, and figure-generation code reproducing  
the results and plots shown in the paper.

---

## Overview
Moment-Guided Optimizers extend classical first-order optimization by maintaining exponential moving  
averages of gradient means and variances, allowing per-parameter adaptive learning rates that  
improve convergence under noisy or sparse gradient conditions.

---

## Setup
1. Clone or unzip the project:
   ```bash
   git clone https://github.com/alicemkoyun/moment_guided_optimizers_pytorch.git
   cd moment_guided_optimizers_pytorch
   pip install -r requirements.txt

## Run Experiments
You can reproduce the results by running the following experiments:

```bash
python experiments/mnist_train.py
python experiments/cifar10_train.py
python experiments/imdb_train.py
python experiments/beta_sweep.py
python experiments/plot_results.py

##  Citation
If you use this code or findings in academic work, please cite:

@article{koyun2025moment,
  title={Moment-Guided Optimizers for Noisy Deep Learning},
  author={Koyun, Alicem},
  year={2025},
  publisher={Independent Research}
}

For questions or collaboration:
Alicem Koyun
alicemkoyun@gmail.com