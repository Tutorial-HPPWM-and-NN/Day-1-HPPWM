# Tutorial: Deep Learning and Embedded Deployment for HPPWM - Day 2

This repository contains the practical material for the second day of the
tutorial, focused on using neural networks to approximate programmed
modulation solution spaces and their subsequent deployment on embedded
systems.

The central objective is to replace massive lookup tables with compact
neural models, trained on offline-generated datasets, in order to finally
run inference on a real SoC system based on the Zybo Z7.

The exercise progression is designed to cover the full implementation
flow: supervised training, model compression, quantization to TensorFlow
Lite, and final deployment on the ARM processor of the embedded platform.

## Repository Structure

The practical material is split into four sequential folders. Each
directory contains a base script (`_base.py`) for in-class work, a script
with the complete solution, and a local instructions file (README).

* **Exercise_1 (Training an MLP):** Implementation of a multilayer neural
  network to approximate the switching angles from the harmonic references
  of the SHC-PWM problem. Includes data splitting, normalization, training,
  and evaluation.
* **Exercise_2 (Model Compression):** Application of complexity-reduction
  strategies, such as weight pruning, to shrink the effective size of the
  network and study the impact on regression error.
* **Exercise_3 (Quantization to TFLite):** Conversion of the trained model
  to quantized TensorFlow Lite. Metrics before and after quantization are
  compared to evaluate the induced degradation.
* **generate_test_sample.py:** Helper script to extract test samples from the dataset, normalize harmonic references with StandardScaler, and format C++ code for `main.cc`.
* **Exercise_4 (Deployment on Zybo Z7):** Execution of the quantized TFLite
  model on the ARM processor of the Zybo Z7 SoC, validating embedded
  inference and execution latency.

## Dataset Used

The dataset used on this day is a CSV file with input harmonic references
and the optimal output angles:

**Inputs**
- `m1`
- `m5`
- `m7`
- `m11`
- `m13`
- `m17`
- `m19`
- `phi5`

**Outputs**
- `alpha_1` through `alpha_17`

## Test Vector Generation for Embedded Deployment

In Exercise 4 (and Day 3), the embedded target (`main.cc`) runs inference on
sample vectors to verify accuracy against ground-truth angles:

* **Sample 0 (default):** $m_1 = 0.85$, $m_5 = 0.01$, $m_7=0, \dots, \phi_5 = -\pi$.
* **StandardScaler Normalization:** Raw inputs are normalized using the scaler
  parameters ($\mu, \sigma$) saved in `exercise_1/artifacts/input_scaler.joblib`.
* **Ground Truth Angles:** Extracted from the dataset for hardware vs software comparison.

To inspect or generate C++ declarations for any dataset index:

```bash
python generate_test_sample.py --sample 0
```

## Requirements and Environment Setup

**Python 3.10 is required** (an exact match, not just "3.10 or newer"), to
avoid compatibility issues between TensorFlow, `tensorflow_model_optimization`,
and the rest of this day's libraries.

### 1. Check that Python 3.10 is installed

```bash
py -0                    # Windows: lists every Python version the launcher finds
python3.10 --version     # macOS/Linux
```

If it doesn't appear, install it from
https://www.python.org/downloads/release/python-31011/ (any 3.10.x build
works) before continuing.

### 2. Create and activate a virtual environment named `Day_2_environment`

**Windows (PowerShell or cmd):**
```bash
py -3.10 -m venv Day_2_environment
Day_2_environment\Scripts\activate
```

**macOS / Linux:**
```bash
python3.10 -m venv Day_2_environment
source Day_2_environment/bin/activate
```

With the environment active (your prompt should show
`(Day_2_environment)`), install the dependencies:

```bash
pip install -r requirements.txt
```

To leave the environment later, run `deactivate` from any terminal.

### 3. Or set it up directly from VS Code (no terminal needed)

1. Open the `Day_2/` folder in VS Code.
2. `Ctrl+Shift+P` (`Cmd+Shift+P` on macOS) → run **"Python: Create
   Environment..."**.
3. Choose **Venv**, select the **Python 3.10** interpreter from the list,
   and when VS Code asks which `requirements.txt` to install, pick the one
   in `Day_2/`.
4. VS Code creates the `Day_2_environment/` folder and automatically
   selects it as the interpreter for this workspace.
5. To confirm or change it later: `Ctrl+Shift+P` → **"Python: Select
   Interpreter"** → choose the entry showing
   `.\Day_2_environment\Scripts\python.exe` (Windows) or
   `./Day_2_environment/bin/python` (macOS/Linux).
6. With that interpreter selected, the ▶ **Run Python File** button (top
   right of the editor) runs any script inside this environment directly —
   no manual activation needed.

### 4. Required environment variable: `TF_USE_LEGACY_KERAS=1`

`tensorflow_model_optimization` (used in Exercise 2, weight pruning) does
not yet support Keras 3, which has been the default Keras since TensorFlow
2.16. The `tf-keras` package (already included in `requirements.txt`)
provides the Keras 2 shim it needs, but TensorFlow must be told to use it
**before** `tensorflow` is imported in any script:

**From the terminal, before running a script:**
```bash
# Windows PowerShell
$env:TF_USE_LEGACY_KERAS = "1"
# Windows cmd
set TF_USE_LEGACY_KERAS=1
# macOS/Linux
export TF_USE_LEGACY_KERAS=1
```

**For this to also apply to VS Code's ▶ Run button** (without exporting
the variable every time), create a `.env` file in the `Day_2/` folder with
this single line:

```
TF_USE_LEGACY_KERAS=1
```

VS Code automatically loads that `.env` file for the selected environment,
so both the integrated terminal and the ▶ Run button are configured with
no extra steps.
