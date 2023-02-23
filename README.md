# GMM-NPs
This is the source code accompanying [Volpp et al., "Accurate Bayesian Meta-Learning by Accurate Task Posterior Inference", ICLR 2023](https://openreview.net/forum?id=sb-IkS8DQw2).

## Installation
Clone this repository and run

```python3 -m pip install .```

from the source directory to install all necessary packages.
It is recommended to create a new virtual environment (``python ~= 3.10``) for this purpose.

## Run
We provide a jupyter notebook 

```scripts/run_experiment.ipynb```

to reproduce the results on the synthetic function classes (Sinusoid1D, LineSine1D) within error bounds and to visualize some predictions. 
The script uses the optimal hyperparameters determined according to the experimental protocol described in the paper.

## License
"GMM-NP" is open-sourced under the [MIT license](LICENSE).
