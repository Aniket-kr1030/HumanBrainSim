# HumanBrainSim

This repository provides a minimal reference implementation of a hierarchical brain model. It now includes additional components such as attention, meta-learning, energy monitoring, lifelong learning, and a simple dashboard server. The code demonstrates how the different modules described in the system specifications can be composed in Python.

Install dependencies and run the demo with:

```bash
pip install -r requirements.txt
python run_model.py --config config.yaml
```

This will execute a short dummy run of the `HierarchicalBrainModel` using randomly generated data.

For spontaneous cycles using the default mode network, run:

```bash
python run_model.py --config config.yaml --spontaneous
```

To resize the model's neural capacity on startup, pass the `--capacity` flag:

```bash
python run_model.py --config config.yaml --capacity 1.5
```

During execution a minimal dashboard prints updates about reward and EWC penalty.

For basic speech input and output using your microphone and speakers, run:

```bash
python run_model.py --config config.yaml --speech
```

### Hardware acceleration

Use the `--device` option to select the compute backend. Valid choices are:

* `cpu` – run entirely on the CPU (default)
* `mps` – use Apple's Metal backend via PyTorch for GPU acceleration
* `ne` – attempt to run inference layers on the Neural Engine via Core ML

Example:

```bash
python run_model.py --config config.yaml --device mps
```

If `coremltools` or `torch` are not installed, the model falls back to CPU.
