# QuadVox: A Large-Scale Fine-Grained Benchmark for Robust Audio Deepfake Detection
[![Hugging Face%20-%20QuadVoxBench](https://img.shields.io/badge/🤗%20Hugging%20Face%20-%20QuadVoxBench-blue)](https://huggingface.co/datasets/Lioy/QuadVoxBench)

QuadVox is a large-scale benchmark (392+ hours) designed to evaluate audio deepfake detection across diverse and fine-grained variations. It is structured in four key aspects: **Speech Style**, **Emotional Prosody**, **Acoustic Environment**, and **Manipulation Type**.

This repository includes the full evaluation suite, state-of-the-art baselines, the newly proposed **Relative Audio Proximity Test (RAPT)** baseline, and a modular generation toolkit (TTS + Voice Conversion) to synthesize deepfake audio.

## Features

- **Four-Aspect Structure**: Organized across Speech Style, Emotional Prosody, Acoustic Environment, and Manipulation Type.
- **Comprehensive Evaluation**: A fine-grained evaluation protocol with 4 targeted tests: Domain Generalization, Emotional Uncanny Valley, Sensitivity vs. Robustness, and Cross-Lingual Generalization.
- **State-of-the-art Baselines**: Includes 7 advanced audio deepfake detection models, such as AASIST, RawNet2, and the proposed RAPT.
- **Rich Audio Content**: Over 392 hours of multilingual audio (English and Chinese) with balanced real and fake samples.
- **Integrated Generation**: TTS + VC toolkit to synthesize domain-specific deepfake audio with automatic metadata updates.
- **Flexible Framework**: Easy-to-use, modular pipelines for both evaluation and generation.

## Table of Contents

- **Installation**: [Installation](#installation)
- **Quick Start**: [Quick Start](#quick-start)
- **Evaluation Guide**: [Evaluation Guide](#evaluation-guide)
- **Generation Guide**: [Generation Guide](#generation-guide)
- **Datasets**: [Dataset Overview](#dataset-overview)
- **Baselines**: [Available Baselines](#available-baselines)
- **Metrics**: [Evaluation Metrics](#evaluation-metrics)
- **Advanced Usage**: [Advanced Usage](#advanced-usage)
- **Utils**: [Utils](#utils)
- **Logging and Results**: [Logging and Results](#logging-and-results)
- **Citation**: [Citation](#citation)
- **License**: [License](#license)

## Installation

### Prerequisites

- Python 3.12+
- CUDA 12.4+
- 95GB+ free disk space for full dataset

### Setup

1. **Clone the repository**:

```bash
git clone https://github.com/wtalioy/QuadVox.git
cd QuadVox
```

2. **Install dependencies**:

```bash
conda create -n quadvox python=3.12 -y
conda activate quadvox
pip install -e .
python -m unidic download

cd src/generation/models/tts/vits/monotonic_align
python setup.py build_ext --inplace
cd ../../../../../..
```

3. **Download datasets**:

```bash
mkdir data
cd data
hf download Lioy/QuadVoxBench --repo-type dataset
cd ../..
```

## Quick Start

The recommended way to use QuadVox is through the provided command-line scripts, which become available after installation.

### Run Experiments

The `quadvox-run` command executes the predefined benchmark experiments.

```bash
# Run all four benchmark tests
quadvox-run

# Run a specific test (e.g., test 1)
quadvox-run -t test1

# Run an test with a specific baseline
quadvox-run -t test1 -b aasist rawnet2 rapt
```

### Standalone Evaluation

Use `quadvox-eval` to run a custom evaluation on one or more datasets.

```bash
# Cross-domain evaluation
quadvox-eval -b aasist -s interview publicspeech -m cross

# In-domain train+eval
quadvox-eval -b rawnet2 -s movie -m in
```

### Standalone Generation

Use `quadvox-generate` to synthesize new deepfake audio for a dataset.

```bash
# Generate English podcast samples using XTTSv2
quadvox-generate -d podcast -t xttsv2 -s en
```


## Evaluation Guide

Evaluate baseline models using the `quadvox-eval` command.

### CLI

```bash
quadvox-eval \
  -b aasist rapt rawnet2 \
  -s phonecall publicspeech interview \
  -m in \
  --metric eer
```

### Arguments

- **-b / --baseline**: one or more of: `aasist`, `aasist-l`, `res-tssdnet`, `inc-tssdnet`, `rawnet2`, `rawgat-st`, `rapt`
- **-s / --subset**: one or more of: `publicfigure`, `news`, `podcast`, `partialfake`, `audiobook`, `noisyspeech`, `phonecall`, `interview`, `publicspeech`, `movie`, `emotional`
- **-m / --mode**: `in` or `cross` (for in-domain or cross-domain evaluation)
- **--metric**: one or more metrics, e.g. `eer`, `auroc`
- **--train_only / --eval_only**: restrict to one stage in `in` mode
- **--data_dir**: path to data root (default: `data/QuadVox`)

### Modes

- **In-domain**: trained on QuadVox and evaluated on QuadVox

  ```bash
  # Train only
  quadvox-eval -b rapt -s interview -m in --train_only
  # Eval only (using existing trained checkpoints)
  quadvox-eval -b rapt -s interview -m in --eval_only --metric eer
  # Train + Eval
  quadvox-eval -b rapt -s interview -m in --metric eer
  ```

- **Cross-domain**: trained on ASVspoof 2019 LA and evaluated on QuadVox

  ```bash
  quadvox-eval -b aasist rawnet2 -s phonecall publicspeech interview -m cross --metric eer
  ```

### Outputs

- Metrics printed to console
- Logs written to `logs/eval.log`

## Generation Guide

Generate synthetic audio for raw domains using the `quadvox-generate` command.

### CLI

```en
quadvox-generate \
  -s podcast \
  -t xttsv2 yourtts \
  -v openvoice \
  -s en
```

### Arguments

- **-s / --subset**: one or more of: `news`, `podcast`, `movie`, `phonecall`, `interview`, `publicspeech`, `partialfake`, `noisyspeech`
- **-t / --tts_model**: one or more of: `vits`, `xttsv2`, `yourtts`, `tacotron2`, `bark`, `melotts`, `gpt-40-mini-tts`
- **-v / --vc_model**: optional VC models: `knnvc`, `freevc`, `openvoice`
- **-p / --partition**: partition specific for PhoneCall subset: `en` or `zh-cn`
- **--data_dir**: path to data root (default: `data/QuadVoxBench`)

Notes:

- Some TTS models require VC (their voices are not speaker-conditioned). These are marked internally and will be paired with provided VC models if any.
- TTS models that support reference audio (e.g., `xttsv2`, `yourtts`) can run without VC.

### Dataset expectations

- Each subset directory should contain a `meta.json` describing items and real audio paths, e.g. `Podcast/meta.json` with `audio/real/...` entries.
- Generated audio is saved under `audio/fake/...` and `meta.json` is updated with a mapping per model.
- `phonecall` expects a subfolder by subset: `PhoneCall/en` or `PhoneCall/zh-cn`.
- `partialfake` will build its own `meta.json` by sampling from `Interview`, `Podcast`, and `PublicSpeech` test metadata11. Ensure these exist at `{Subset}/meta_test.json`.

### Examples

- **TTS-only English podcast generation** (reference-speaker TTS):

```bash
quadvox-generate -s podcast -t xttsv2 yourtts -p en
```

- **Chinese news with TTS+VC** (pairs TTS that require VC with a VC model):

```bash
quadvox-generate -s news -t gpt4omini melotts bark -v openvoice
```

- **PartialFake composition across domains**:

```bash
quadvox-generate -s partialfake -t xttsv2 yourtts -v openvoice
```

### Outputs and logs

- Generated files: `{Subset}/audio/fake/...`
- Updated metadata: `{Subset}/meta.json`
- Logs: `logs/generation*.log`

## Experiment Guide

QuadVox includes four predefined benchmark tests to test different aspects of deepfake detection models. Use the `quadvox-run` command to execute them.

### CLI

```bash
# Run all tests for all default baselines
quadvox-run

# Run a single test
quadvox-run -t test1

# Run a single test for a subset of baselines
quadvox-run -t test1 -b aasist rawnet2
```

### Arguments

- **-t / --test**: one of `test1`, `test2`, `test3`, `test4`, or `all` (default).
- **-b / --baseline**: one or more of: `aasist`, `aasist-l`, `res-tssdnet`, `inc-tssdnet`, `rawnet2`, `rawgat-st`, `rapt`
- **--data_dir**: path to the data directory.
- **--device**: compute device (`cuda` or `cpu`, default: `cuda`).

### Test Descriptions

- **`test1`**: **Domain Generalization Test**: Evaluates generalization from *Scripted* audio (control) to *Spontaneous* and *Real-world* audio (targets).
- **`test2`**: **Emotional Uncanny Valley Test**: Evaluates detectors trained on *Neutral* speech (control) against unseen *Emotional* speech (target).
- **`test3`**: **Sensitivity vs. Robustness Test**: Jointly tests sensitivity on *PartialFake* audio (target 1) and robustness on *NoisySpeech* (target 2) against a *CleanSpeech* control.
- **`test4`**: **Cross-Lingual Generalization Test**: Evaluates detectors trained on English (en) vs. Chinese (zh) and vice-versa, testing for language-independent artifact detection.

### Outputs

- Per-test results are saved to `results/test_{timestamp}.json`.
- Detailed logs are saved to `logs/test_{timestamp}.log`.

## Available Baselines

QuadVox includes 7 state-of-the-art audio deepfake detection models evaluated in the paper:

| **Baseline**    | **Description**                                              | **Paper**                                                    |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **AASIST**      | Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks | [ICASSP 2022](https://arxiv.org/abs/2110.01200)              |
| **AASIST-L**    | Lightweight variant of AASIST                                | [ICASSP 2022](https://arxiv.org/abs/2110.01200)              |
| **RawNet2**     | End-to-end anti-spoofing using raw waveforms                 | [ICASSP 2021](https://arxiv.org/abs/2011.01108)              |
| **Res-TSSDNet** | Time-domain synthetic speech detection net (Resnet Net Style) | [IEEE 2021](https://arxiv.org/abs/2106.06341)                |
| **Inc-TSSDNet** | Time-domain synthetic speech detection net (Inception Net Style) | [IEEE 2021](https://arxiv.org/abs/2106.06341)                |
| **RawGAT-ST**   | End-to-End Spectro-Temporal Graph Attention Networks for Speaker Verification Anti-Spoofing and Speech Deepfake Detection | [ASVspoof 2021 Workshop](https://arxiv.org/abs/2107.12710)   |
| **RAPT**        | Relative Audio Proximity Test (MMD based detection)          | [CVPR 2026](https://www.google.com/search?q=https://github.com/wtalioy/MDAD) |

## Evaluation Metrics

QuadVox supports the following evaluation metrics:

- **EER** (Equal Error Rate): Primary metric for audio deepfake detection.
- **AUROC** (Area Under the Receiver Operating Characteristic Curve): Secondary metric.

## Advanced Usage

### Custom Dataset

To add a new dataset, create a class inheriting from `BaseSubset`:

```python
from quadvox_datasets.base import BaseSubset

class MyDataset(BaseSubset):
    def __init__(self, data_dir=None, *args, **kwargs):
        super().__init__(os.path.join(data_dir or "data", "MyDataset"), *args, **kwargs)
        self.name = "MyDataset"
```

### Custom Baseline

To add a new baseline model, inherit from the `Baseline` class:

```python
from quadvox.baselines.base import Baseline

class MyBaseline(Baseline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "MyBaseline"
        self.supported_metrics = ["eer", "acc"]
    
    def evaluate(self, data, labels, metrics, **kwargs):
        # Implementation
        pass
```

### Configuration

Model configurations are stored in `src/eval/baselines/{model}/config/`:

- `model.yaml`: Model architecture configuration
- `train_default.yaml`: Default training configuration

## Utils

Utility scripts for dataset processing and management are available in `src/utils/`, including:

- Dataset splitting and creation scripts for different domains
- Audio duration calculation and metadata management
- Dataset filtering, reduction, and resampling tools
- Backup file cleanup utilities

## Logging and Results

Evaluation results are automatically logged to:

- Console output with detailed metrics
- `logs/eval.log`: Comprehensive evaluation logs with rotation

Example output:

```
(AASIST on Interview) eer: 0.1234
(RawNet2 on PublicSpeech) eer: 0.2345
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use QuadVox in your research, please cite:

```
@inproceedings{quadvox2026,
  title={QuadVox: A Large-Scale Fine-Grained Benchmark with Relative Audio Proximity Test for Robust Audio Deepfake Detection},
  author={Ruiming Wang, et al.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

------

<div align="center">

Made with ❤️ for advancing audio deepfake detection research

</div>
