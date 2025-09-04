# Flappy Bird NEAT-AI

A Python project that trains a neuro-evolutionary neural network (NEAT) to master the classic **Flappy Bird** game using the `neat-python` library and `pygame` for rendering. You can either **train** the AI from scratch or **play** the game with the best-performing genome (saved as `winner.pkl`).

---

## Table of Contents
1. [Features](#features)
2. [Demo](#demo)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
   * [Training Mode](#training-mode)
   * [Play Mode](#play-mode)
7. [NEAT Configuration](#neat-configuration)
8. [Command-line Arguments](#command-line-arguments)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)
11. [License](#license)

---

## Features
- **Neuro-Evolution**: Utilises NEAT (Neuro-Evolution of Augmenting Topologies) to evolve both weights *and* network topologies.
- **Parallel Evaluation**: Spawns multiple `multiprocessing` workers to evaluate genomes concurrently for faster training.
- **Async Game Loop**: Uses `asyncio` to keep the Pygame render loop non-blocking during evaluation.
- **Configurable Generations & Population**: Easily tweak parameters such as population size, activation functions, mutation rates, etc. via `neat-config.txt`.
- **Persistent Best Genome**: Saves the best-performing genome as `winner.pkl` for later gameplay or analysis.
- **Modular Game Logic**: Clear separation between core game entities (player, pipes, background, score) located in `game/src/`.

## Demo
Coming soon! (GIF / video of the AI achieving high scores.)

---

## Project Structure
```
.
├── game/                 # Game package & assets
│   ├── assets/           # Sprites & audio
│   └── src/              # Game entities & utilities
├── main.py               # Entry-point for training / play
├── args_parser.py        # CLI argument parser
├── args_constants.py     # Enum helpers for modes
├── neat-config.txt       # NEAT algorithm hyper-parameters
├── winner.pkl            # Saved best genome (generated after training)
├── README.md             # ⇦ you are here
├── pyproject.toml        # Python metadata & dependencies
└── .gitignore            # Git ignored files
```

---

## Requirements
- Python **3.10+** (see `.python-version`)
- [Poetry](https://python-poetry.org/) **or** `pip`

Python packages (automatically installed via `pyproject.toml`):
- `neat-python>=0.92`
- `pygame>=2.6.1`
- `numpy>=2.2.6`

Optional dev tools:
- `black` – code formatter
- `icecream` – better print debugging

---

## Installation
### 1. Clone the repo
```bash
git clone https://github.com/<your-user>/flappy-neat-ai.git
cd flappy-neat-ai
```

### 2. Create & activate virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
Using uv (preferred):
```bash
uv sync
```
Or with `pip`:
```bash
pip install -r <(python -c "import tomllib,sys,os;d=tomllib.load(open('pyproject.toml','rb'));print('\n'.join(d['project']['dependencies']))")
```

---

## Quick Start

### Training Mode
Train a new model for **20 generations** (default):
```bash
python main.py train --config neat-config.txt --generations 20
```
Key arguments:
* `--config/-c` – Path to NEAT configuration (default: `neat-config.txt`)
* `--generations/-g` – Number of generations to evolve (default: `20`)

The best genome of the last generation is saved to `winner.pkl`.

### Play Mode
Play Flappy Bird controlled by a **pre-trained winner genome**:
```bash
python main.py play --model winner.pkl --config neat-config.txt
```
The game window will open at a fixed coordinate to avoid overlap if multiple windows spawn.

---

## NEAT Configuration
The file [`neat-config.txt`](./neat-config.txt) defines every hyper-parameter used by the NEAT algorithm, grouped into sections:
- `[NEAT]` – global evolution settings (population size, stagnation, etc.)
- `[DefaultGenome]` – initial connection scheme, activation functions
- `[DefaultSpeciesSet]` – speciation threshold
- `[DefaultReproduction]` – elitism and survival thresholds

You can adjust these values to change the learning dynamics.

---

## Command-line Arguments
| Mode      | Flag              | Default            | Description                                |
|-----------|-------------------|--------------------|--------------------------------------------|
| **train** | `--config, -c`    | `neat-config.txt`  | NEAT config path                           |
|           | `--generations,-g`| `20`               | Number of generations                      |
| **play**  | `--model, -m`     | `winner.pkl`       | Path to saved best genome                  |
|           | `--config, -c`    | `neat-config.txt`  | NEAT config path                           |

---

## Troubleshooting
| Issue                              | Solution |
|------------------------------------|----------|
| *pygame window not opening*        | Ensure you have a functional display (on WSL use `sudo apt install xvfb`). |
| *NEAT training very slow*          | Increase `batch_size` in `run_multiple_gnomes` or reduce `pop_size` in config. |
| *Winner file empty / not found*    | Verify that training completed and `winner.pkl` exists in project root. |

---

## Contributing
Pull requests are welcome! If you would like to add new features (visualizations, improved fitness functions, etc.):
1. Fork the repository & create your branch: `git checkout -b feature/awesome-feature`.
2. Run `black .` before committing.
3. Submit a PR with a clear description.

---

## License
Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.

---

*Happy flapping & evolving!*