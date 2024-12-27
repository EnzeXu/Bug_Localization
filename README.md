
# Bug Localization

Welcome to the **Bug Localization** repository! This project focuses on localizing relevant code snippets in response to bug reports. The repository includes tools for dataset generation, training, and evaluation of the BLNT5 model with two variants: **Concat** and **CosSim**.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Dataset Generation](#dataset-generation)
- [Training](#training)
  - [Command](#command)
  - [Arguments](#arguments)
  - [Example](#example)
- [Contributors](#contributors)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Citation](#citation)


## Features
- **Dataset Generation**: Automatically generate datasets for bug report and method pairs from GitHub repositories.
- **Training**: Train BLNT5 models to predict similarity scores between bug reports and methods.
- **Evaluation**: Evaluate model performance using standard classification metrics.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/EnzeXu/Bug_Localization.git
   cd Bug_Localization
   ```

2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset Generation

To generate the dataset, use the `make_one_dataset.py` script. Replace the repository name as needed:

```bash
python -u make_one_dataset.py --repo <repository_name>
```

Example:
```bash
python -u make_one_dataset.py --repo robolectric/robolectric
```

This script processes the specified repository to create a dataset of bug report and method pairs.

---

## Training

Run the `run.py` script to train the BLNT5 model. The script supports several configurable arguments:

### Command

```bash
python run.py --seed <SEED> --gpu_id <GPU_ID> --model <MODEL> [OPTIONS]
```

### Arguments
- `--seed`: Set the random seed (default: `42`).
- `--gpu_id`: Specify the GPU ID to use (default: `3`).
- `--random`: Enable random weights for evaluation without training (default: `False`).
- `--no_wandb`: Disable logging with `wandb` (default: `False`).
- `--model`: Choose the model variant. Options:
  - `BLNT5Concat`
  - `BLNT5Cosine` (default: `BLNT5Concat`).

### Example

Run training with default settings:
```bash
python run.py
```

Train the model with specific parameters:
```bash
python run.py --seed 123 --gpu_id 0 --model BLNT5Cosine
```

Run with random weights for baseline comparison:
```bash
python run.py --random
```

---

## Contributors

- **Enze Xu**: exu03@wm.edu
- **Daoxuan Xu**: dxu05@wm.edu
- **Yi Lin**: ylin13@wm.edu

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this code or dataset in your research, please cite this repository:

```bibtex
@misc{Bug_Localization,
  author = {Enze Xu, Daoxuan Xu and Yi Lin},
  title = {Bug Localization},
  year = {2024},
  howpublished = {\url{https://github.com/EnzeXu/Bug_Localization}}
}
```