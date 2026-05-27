# Code for "Multimodality as Bridge or Hurdle in Human--AI Communication"

This repository contains the prototype code and evaluation scripts for the paper:

**Multimodality as Bridge or Hurdle in Human--AI Communication: Selective Interpretability in Game-Based Multimodal Tasks**  
Noah Schager, Universität Kassel

The paper studies multimodal communication as a sender--receiver--validator process. It introduces a reusable game-based framework for modelling multimodal tasks and instantiates it in an adversarial human--AI captioning task over paired images. In the captioning task, captions are produced under two conditions: a standard condition and an AI-aware condition in which captions should remain understandable for humans while becoming harder for AI to interpret.

## Repository overview

The repository contains two main parts:

1. **Game framework prototype**  
   Code for representing multimodal games through tasks, examples, alphabets, signatures, adapters, agents, and a game loop.

2. **Evaluation scripts and rater files**  
   Code for evaluating human and AI ratings using accuracy, Fleiss' kappa, Cohen's kappa, and logistic models of label assignment.

A lightweight GUI prototype also exists, but it is optional and not required for reproducing the evaluation described in the paper.

## Main files

| File | Purpose |
| `game_interface_and_emoji_trivia_prototype.ipynb` | Main notebook-style prototype of the framework. Defines and demonstrates the reusable components described in the paper. This is the preferred entry point for running and experimenting with the framework interactively. |
| `game_interface_and_emoji_trivia_prototype.py` | Exported Python version of the notebook. Used as an importable module by `game.py` and `chat.py`; not necessarily the main file to run directly. 
| `eval.py` | Evaluation script for the rater files. Computes human and AI Fleiss' kappa, human--AI Cohen's kappa, accuracies against `H0`, and label-B logistic models using GEE and GLMM. |
| `game.py` | Optional NiceGUI wrapper around the game framework. Useful for interactive demos, but not needed for the paper evaluation. |
| `chat.py` | Helper functions for the optional NiceGUI chat interface. |
| `README.md` | Project documentation. |

## Data files for evaluation

The evaluation expects two rater files, one for each captioning condition:

```text
rater/ImageAnnotation_standard_captioning_condition.ods
rater/ImageAnnotation_ai_aware_captioning_condition.ods
```

Both files should follow the same structure, with the following relevant columns:

| Column | Meaning |
|---|---|
| `Pair` | Image-pair identifier. |
| `A`, `B` | Target marker. The intended target image is marked with `X` in either column `A` or column `B`. |
| `Caption` | Caption shown to the raters. |
| `H1`, `H2` | Additional human ratings. |
| `AI1`, `AI2`, `AI3` | AI ratings from repeated model runs. |

The script derives `H0` automatically from the `A`/`B` target marker. `H0` is treated as the reference label for accuracy, and as part of the human-side ratings for agreement analysis.

## Prerequisites

- Python 3.8+
- `uv` installed

### Install `uv`:

```bash
pip install uv
```

### or (recommended):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create and sync the virtual environment:

```bash
uv sync
```

## Running the evaluation

The script `eval.py` is currently configured with a single input path, this can be changed by editing the `file_path` variable:

```python
file_path = "rater/ImageAnnotation_ai_aware_captioning_condition.ods"
```

To evaluate both conditions, run the script once for each rater file by changing this path:

```python
file_path = "rater/ImageAnnotation_standard_captioning_condition.ods"
```

and then:

```python
file_path = "rater/ImageAnnotation_ai_aware_captioning_condition.ods"
```

Run:

```bash
uv run eval.py
```

The script prints:

- Fleiss' kappa among human raters (`H0`, `H1`, `H2`),
- Fleiss' kappa among AI raters (`AI1`, `AI2`, `AI3`),
- Cohen's kappa between human-majority and AI-majority labels,
- accuracy of each rater against `H0`,
- AI-majority accuracy against `H0`,
- a label-assignment model of the form `label_B ~ group_AI` using GEE and GLMM.

The logistic model should be interpreted as a model of **label tendency**, not correctness. It tests whether AI and human raters differ in their tendency to assign label `B` rather than label `A`.

## Running the framework prototype

The main framework code is contained in:

```bash
game_interface_and_emoji_trivia_prototype.py
```

It defines the reusable components described in the paper:

- `Alphabet`: available symbols or modal items,
- `Example`: target message plus encoder, decoder, and validator alphabets,
- `Task`: collection of examples with phase-specific instructions,
- `Signature`: validity rules over an alphabet,
- `Adapter`: interface to a human participant or external AI model,
- `Agent`: phase-specific control logic using a signature and adapter,
- `Game`: sender--receiver--validator loop.

The prototype includes example configurations for:

- Movie Emoji Trivia,
- 4 Images 1 Word,
- paired-image captioning.

The AI adapter uses an OpenAI-compatible API endpoint. Configure the endpoint and API key in `send_request(...)` before using AI-based decoding. Do not commit real API keys.

## Optional GUI prototype

The files `game.py` and `chat.py` provide an optional NiceGUI-based chat interface for the game loop. This interface is useful for demonstrations, but it is not required for reproducing the evaluation results in the paper.

Run the GUI with:

```bash
uv run game.py
```

The GUI expects the image files referenced in the game configuration to exist under an `images/` directory.

## Relation to the paper

- **Paper Section 3** describes the framework implemented in `game_interface_and_emoji_trivia_prototype.py`.
- **Paper Section 4** describes the adversarial human--AI captioning task and the image-pair dataset.
- **Paper Section 5** corresponds to `eval.py` and the two rater files.
- **Paper Section 6** discusses the quantitative results and qualitative observations, including human encoding strategies and AI prior effects.

## Notes

This repository is a research prototype. Some components, especially the notebook-style game prototype and GUI wrapper, are intended for experimentation and demonstration rather than production deployment.
