# SJE Extensions

A collection of personal extensions for the [SpikingJelly](https://github.com/fangwei123456/spikingjelly) library.

## Features

* **BTSP Learner**: An implementation of the Behavioral-TimeScale Plasticity learning model.
* **IQIF Neuron**: An implementation of the Integer-Quadratic-Integrate-and-Fire neuron model.

## Installation

To install this package in editable mode for development, first ensure you have a Conda(, virtualenv, or any) environment with SpikingJelly installed. Then, run the following command from the root of this repository:

```bash
pip install -e .
```

## Usage

```python
from sje_extensions.btsp import BTSPLearner
from sje_extensions.iqif import IQIFNode

# ... your code here
```

