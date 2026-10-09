# GraNoM

**Project Structure**
- `pyproject.toml`: Project configuration and Python packaging metadata.
- `README.md`: This file — high-level overview and structure.
- `pf/`: Main Python package with experiment code and utilities.
	- `__init__.py`: Package initializer.
	- `client_app.py`: Client-side application.
	- `server_app.py`: Server-side application.
	- `model_inversion_attack.py`: Implementation and helpers for model inversion attacks.
	- `strategy.py`: Contains strategies used in the experiments (e.g., training/evaluation strategies).
	- `task.py`: Task definitions and orchestration utilities for running experiments.
	- `client/` or other modules: (If present) additional helpers and modules used by the package.

**How to use (quick)**
- Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

- Run scripts from the `pf` package. Example (project-specific entry points may vary):

```bash
flwr run . > <job_name>.log
```

**License**
This software is released under an Evaluation-Only License.
Use is limited to personal, non-commercial evaluation. Academic and research use is explicitly prohibited.

For licensing inquiries or special permissions, contact the developer.

See LICENSE for full terms.
