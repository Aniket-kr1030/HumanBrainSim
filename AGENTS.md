Coding Guidelines
=================

- Follow PEP8 style for all Python code.
- When modifying the repository, run the following commands to ensure the demo
  executes correctly:

```bash
python run_model.py --config config.yaml --steps 1
python run_model.py --config config.yaml --spontaneous --steps 1
```

These quick runs verify that the main loop and spontaneous mode still work.
