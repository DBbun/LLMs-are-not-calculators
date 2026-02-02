# LLM Education Impact Simulator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Developed by:** DBbun LLC  
**Based on:** Jackson, D. (2025). "LLMs are not calculators: Why educators should embrace AI (and fear it)"

A research-grounded simulation for generating synthetic observational data about how students interact with AI tools in educational settings.

## Overview

This simulator generates realistic data capturing student behaviors when using different AI tools:
- **Search engines** (traditional documentation-driven approach)
- **Explicit-context LLMs** (careful, student-controlled context selection)
- **Agentic LLMs** (automated context selection with less student control)

The simulation models key findings from educational research:
- Context engineering is more critical than prompt engineering
- Reading documentation decreases with LLM usage
- Intentional tool use preserves student agency
- Agentic tools risk module boundary violations
- Verification checks significantly reduce errors

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-education-simulator.git
cd llm-education-simulator

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run with default settings
python llm_education_simulator.py

# Customize simulation
python llm_education_simulator.py --students 200 --tasks 20 --seed 42

# Use custom configuration
python llm_education_simulator.py --config my_config.json
```

### Output

The simulator generates:
- `students.csv` - Student characteristics and behavioral traits
- `tasks.csv` - Educational tasks with varying properties
- `runs.csv` - Complete records of student-task-tool combinations
- `events.csv` - Fine-grained event log of all activities
- `summary.json` - Aggregated statistics
- `config.json` - Configuration used for reproducibility
- `metadata.json` - Dataset metadata for Hugging Face
- `README.md` - Dataset-specific documentation
- `fig*.png` - Visualization figures (8 total)

## Configuration

Create a custom `config.json`:

```json
{
  "seed": 42,
  "n_students": 150,
  "n_tasks": 12,
  "tasks_per_student": 5,
  "task_type_weights": {
    "coding": 0.60,
    "writing": 0.25,
    "reading": 0.15
  },
  "tool_distribution": {
    "search": 0.15,
    "llm_explicit": 0.55,
    "llm_agentic": 0.30
  }
}
```

### Key Configuration Parameters

#### Student Behavioral Traits
- `effort_minimization_mean` (0.62) - Tendency to take shortcuts
- `doc_discipline_mean` (0.42) - Propensity to read documentation
- `context_care_mean` (0.55) - Attention to context selection
- `intentionality_mean` (0.48) - Deliberate vs passive tool use

#### Error Rates
- `llm_hallucination_base` (0.20) - Base LLM factual error rate
- `llm_omission_base` (0.25) - Base LLM omission rate
- `agentic_boundary_violation_base` (0.32) - Agentic tool boundary breaks
- `explicit_boundary_violation_base` (0.10) - Explicit LLM boundary breaks

#### Outcome Weights
- `weight_reading_on_quality` (0.20) - Impact of reading on score
- `weight_intentionality_on_quality` (0.24) - Impact of intentional use
- `weight_hallucinations_on_quality` (0.32) - Impact of uncaught errors

## Data Schema

### Students Table
| Column | Type | Description |
|--------|------|-------------|
| student_id | string | Unique identifier (S0000-S9999) |
| skill | float [0-1] | Base ability level |
| effort_minimization | float [0-1] | Tendency to take shortcuts |
| doc_discipline | float [0-1] | Reads documentation before acting |
| context_care | float [0-1] | Attention to context selection |
| time_pressure | float [0-1] | Stress/urgency level |
| intentionality | float [0-1] | Deliberate vs passive tool use |

### Tasks Table
| Column | Type | Description |
|--------|------|-------------|
| task_id | string | Unique identifier (T000-T999) |
| task_type | string | coding, writing, or reading |
| difficulty | float [0-1] | Task difficulty |
| has_module_boundaries | bool | Requires modularity (coding) |
| rubric_strictness | float [0-1] | Grading precision |
| requires_prior_reading | bool | Needs background reading |
| conceptual_complexity | float [0-1] | Interconnected concepts |

### Runs Table (Main Analysis Table)
| Column | Type | Description |
|--------|------|-------------|
| run_id | string | Unique identifier (R000000-R999999) |
| student_id | string | Foreign key to students |
| task_id | string | Foreign key to tasks |
| tool_mode | string | search, llm_explicit, or llm_agentic |
| **Behaviors** | | |
| prompts | int | LLM prompt rounds |
| doc_opens | int | Quick documentation lookups |
| reading_sessions | int | Sustained reading sessions |
| edits | int | Manual editing rounds |
| test_runs | int | Testing iterations (coding only) |
| verification_checks | int | Checking LLM output |
| **Errors** | | |
| hallucinations | int | LLM factual errors |
| hallucinations_caught | int | Detected/corrected hallucinations |
| omissions | int | LLM omission errors |
| omissions_caught | int | Detected/corrected omissions |
| boundary_violations | int | Module boundary breaks |
| **Outcomes** | | |
| score | float [0-1] | Rubric-based quality |
| passed | int (0/1) | Binary pass/fail |
| tests_failed | int | Failed test count (coding) |
| **Learning Proxies** | | |
| agency_proxy | float [0-1] | Student ownership of work |
| cognitive_engagement | float [0-1] | Deep vs surface learning |
| context_quality | float [0-1] | Context selection quality |

### Events Table
Fine-grained log of all student activities with timestamps, event types, and metadata.

## Research Applications

This dataset enables investigation of:

1. **Tool Effectiveness**: How do different AI tools affect learning outcomes?
2. **Student Characteristics**: Which traits predict successful AI usage?
3. **Context Engineering**: How does context quality relate to errors and outcomes?
4. **Agency Preservation**: What behaviors maintain student ownership?
5. **Intentionality**: Does deliberate use mediate tool effects on learning?
6. **Reading Behavior**: How do LLMs change documentation engagement?
7. **Verification**: What's the ROI of checking LLM outputs?
8. **Boundary Violations**: When do agentic tools break modularity?

## Example Analyses

### Pass Rates by Tool Mode

```python
import pandas as pd
import matplotlib.pyplot as plt

runs = pd.read_csv('output/runs.csv')

# Compare pass rates
pass_rates = runs.groupby('tool_mode')['passed'].mean()
print(pass_rates)

# Visualize
pass_rates.plot(kind='bar', title='Pass Rates by Tool Mode')
plt.ylabel('Pass Rate')
plt.ylim([0, 1])
plt.show()
```

### Agency vs Cognitive Engagement

```python
import seaborn as sns

# Scatter plot by tool mode
sns.scatterplot(data=runs, x='cognitive_engagement', y='agency_proxy', 
                hue='tool_mode', alpha=0.6)
plt.title('Cognitive Engagement vs Student Agency')
plt.xlabel('Cognitive Engagement')
plt.ylabel('Agency Proxy')
plt.show()
```

### Reading Behavior Changes

```python
# Compare reading between tool modes
reading_by_tool = runs.groupby('tool_mode')['reading_sessions'].mean()
doc_opens_by_tool = runs.groupby('tool_mode')['doc_opens'].mean()

print("Reading Sessions by Tool:")
print(reading_by_tool)
print("\nDoc Lookups by Tool:")
print(doc_opens_by_tool)
```

### Verification Effectiveness

```python
llm_runs = runs[runs['tool_mode'].str.startswith('llm')]

# Calculate catch rates
llm_runs['halluc_catch_rate'] = (
    llm_runs['hallucinations_caught'] / 
    llm_runs['hallucinations'].clip(lower=1)
)

# Bin by verification checks
llm_runs['verify_bins'] = pd.cut(llm_runs['verification_checks'], bins=4)
catch_by_verify = llm_runs.groupby('verify_bins')['halluc_catch_rate'].mean()

print("Hallucination Catch Rate by Verification Level:")
print(catch_by_verify)
```

## Theoretical Foundation

The simulation implements findings from:

1. **Jackson (2025)** - Core observations about LLM usage in education
2. **Barba (2025)** - Student behavior with LLMs in programming courses
3. **Salazar-Gómez & Sarma (2025)** - AI in education framework
4. **Kosmyna et al. (2025)** - Cognitive effects of LLM usage
5. **Lee et al. (2025)** - Impact on critical thinking

### Key Mechanisms Modeled

**Reading Behavior**: Students using LLMs read 45% less documentation (based on Jackson's course observations)

**Context Selection**: Quality depends on `context_care` trait and tool mode (explicit vs agentic)

**Error Generation**: Hallucinations and omissions increase with:
- Lower skill and context care
- Higher time pressure and effort minimization
- Reduced intentionality

**Verification Effects**: Checking LLM outputs catches:
- 60% of hallucinations (when skill is high)
- 50% of omissions (when skill is high)

**Agency Calculation**: Combines editing, reading, verification (positive) with prompting and effort minimization (negative)

**Boundary Violations**: Much higher with agentic tools (32% base rate vs 10% explicit)

## Validation & Calibration

Parameter values are derived from:
- Jackson's MIT course experiments (Fall 2024)
- Reported pass rates and behavioral observations
- Error rates from LLM benchmarks
- Student trait distributions from educational psychology

While synthetic, the data reflects real patterns observed in educational settings.

## Deployment

### For GitHub

1. Ensure all files are committed:
```bash
git add llm_education_simulator.py requirements.txt LICENSE README.md
git commit -m "Initial release of LLM education simulator"
git push origin main
```

2. Create a release with version tag:
```bash
git tag -a v1.0 -m "Version 1.0: Enhanced simulation based on Jackson (2025)"
git push origin v1.0
```

### For Hugging Face Datasets

1. Generate dataset:
```bash
python llm_education_simulator.py --seed 42
```

2. The `output/` directory now contains:
   - All CSV files
   - `metadata.json` with Hugging Face-compatible schema
   - `README.md` with dataset card
   - Figures for visualization

3. Upload to Hugging Face:
```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create dataset repo
huggingface-cli repo create llm-education-impact --type dataset

# Upload files
cd output
huggingface-cli upload llm-education-impact . --repo-type dataset
```

4. The dataset card (README.md) includes:
   - Overview and motivation
   - Dataset statistics
   - Variable descriptions
   - Usage examples
   - Citation information

## Citation

If you use this simulator or generated datasets, please cite:

```bibtex
@article{jackson2025llms,
  title={LLMs are not calculators: Why educators should embrace AI (and fear it)},
  author={Jackson, Daniel},
  year={2025},
  month={December}
}

@software{llm_education_simulator,
  title={LLM Education Impact Simulator},
  author={DBbun LLC},
  note={Based on research by Jackson, Daniel},
  year={2026},
  version={1.0},
  url={https://github.com/dbbun/llm-education-simulator}
}
```

## Contributing

Contributions welcome! Areas for enhancement:

- Additional task types (problem-solving, collaborative work)
- More sophisticated error models
- Temporal dynamics (learning over time)
- Peer interaction effects
- Multi-modal tasks
- Alternative assessment methods

Please open issues for bugs or feature requests.

## License

MIT License - See LICENSE file for details

## Acknowledgments

This work builds on:
- Daniel Jackson's educational experiments at MIT
- Insights from Mitchell Gordon, Eagon Meng, and course TAs
- Educational AI research by Lorena Barba, Sanjay Sarma, and colleagues

## Contact

For questions or collaboration:
- **Organization:** DBbun LLC
- Open a GitHub issue
- Based on research by Daniel Jackson (MIT CSAIL)

---

**© 2026 DBbun LLC. All rights reserved.**

**Note**: This is a simulation tool for research purposes. Generated data is synthetic but grounded in real educational observations. Use thoughtfully in educational policy and research contexts.
