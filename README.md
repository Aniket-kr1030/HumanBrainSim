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

For real-time two-way speech and camera interaction, run:

```bash
python run_model.py --config config.yaml --speech --camera
```
The program will run indefinitely until you press `Ctrl+C`. A small window
shows the live camera feed with the current audio amplitude and TD-error
overlaid as text so you can visually monitor what the model is sensing.
Use the `--steps` flag to limit the number of cycles when testing.
If OpenCV fails to display the camera window (for example when running via
SSH or without GUI support), the program now continues headlessly and you
can simply rely on the printed metrics.

For a simple text-only interface without camera or microphone, launch with
the `--text` option. Type your messages and the model will echo a response
until you type `quit` or `exit`.

```bash
python run_model.py --config config.yaml --text
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
