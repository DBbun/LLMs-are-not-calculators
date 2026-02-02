# Deployment Guide

**Organization:** DBbun LLC  
**Project:** LLM Education Impact Simulator  
**Version:** 1.0  

## Quick Deployment Checklist

### For GitHub

- [ ] Code is tested and runs without errors
- [ ] README.md is comprehensive
- [ ] LICENSE file is included
- [ ] requirements.txt lists all dependencies
- [ ] .gitignore excludes output files
- [ ] Example configuration provided
- [ ] Version tagged (e.g., v1.0)

### For Hugging Face Datasets

- [ ] Dataset generated with fixed seed for reproducibility
- [ ] All CSV files validated
- [ ] metadata.json includes proper schema
- [ ] README.md follows dataset card format
- [ ] Figures generated for documentation
- [ ] File sizes are reasonable (<100MB for CSV files)

## Step-by-Step Deployment

### 1. Test the Simulator Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run with default settings
python llm_education_simulator.py

# Verify output
ls -lh output/
# Should see: students.csv, tasks.csv, runs.csv, events.csv, 
# summary.json, metadata.json, README.md, config.json, 
# and 8 PNG figures
```

### 2. GitHub Deployment

#### A. Create Repository

```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit: LLM Education Impact Simulator v1.0"

# Create repository on GitHub
# Then:
git remote add origin https://github.com/dbbun/llm-education-simulator.git
git branch -M main
git push -u origin main
```

#### B. Create Release

```bash
# Tag the release
git tag -a v1.0 -m "Version 1.0: Enhanced simulation based on Jackson (2025)"
git push origin v1.0

# On GitHub:
# 1. Go to Releases
# 2. Click "Draft a new release"
# 3. Choose tag v1.0
# 4. Title: "Version 1.0: Jackson (2025) Integration"
# 5. Description: Copy from CHANGELOG section below
# 6. Attach example output files as assets
# 7. Publish release
```

#### C. Repository Settings

**About section:**
- Description: "Research-grounded simulator for educational AI impact analysis"
- Website: Link to paper or documentation
- Topics: `education`, `ai`, `llm`, `simulation`, `research-tool`, `synthetic-data`

**README badges:**
Already included in README.md:
- License badge
- Python version badge

**GitHub Actions (Optional):**
Create `.github/workflows/test.yml` for automated testing:

```yaml
name: Test Simulator

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run simulator
      run: python llm_education_simulator.py --students 10 --tasks 3 --no-figures
    - name: Verify outputs
      run: |
        test -f output/students.csv
        test -f output/runs.csv
```

### 3. Hugging Face Deployment

#### A. Generate Dataset

```bash
# Generate with fixed seed for reproducibility
python llm_education_simulator.py \
  --seed 42 \
  --students 120 \
  --tasks 15 \
  --output-dir output_huggingface

# Verify all files
cd output_huggingface
ls -lh
```

#### B. Prepare Dataset Card

The simulator automatically generates `README.md` in the output directory.
For Hugging Face, you may want to customize:

```bash
# Copy the Hugging Face card template
cp HUGGINGFACE_CARD.md output_huggingface/README.md

# Edit to add:
# - Actual dataset statistics from summary.json
# - Link to your GitHub repository
# - Your contact information
```

#### C. Upload to Hugging Face

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login (one-time)
huggingface-cli login
# Enter your access token from https://huggingface.co/settings/tokens

# Create dataset repository
huggingface-cli repo create llm-education-impact --type dataset --organization dbbun

# Upload files
cd output_huggingface
huggingface-cli upload dbbun/llm-education-impact . --repo-type dataset

# Or use Python API:
```

**Python Upload Script:**

```python
from huggingface_hub import HfApi, create_repo

api = HfApi()

# Create repository
create_repo(
    repo_id="dbbun/llm-education-impact",
    repo_type="dataset",
    private=False
)

# Upload entire directory
api.upload_folder(
    folder_path="output_huggingface",
    repo_id="dbbun/llm-education-impact",
    repo_type="dataset"
)
```

#### D. Dataset Card Metadata

The YAML frontmatter in `HUGGINGFACE_CARD.md` configures:
- License (MIT)
- Task categories (educational-analysis, behavioral-modeling)
- Language (English)
- Tags for discoverability
- Size category

Make sure these match your dataset!

### 4. Documentation

#### Update Citations

In both README files, update:
- GitHub repository URL
- Your contact information
- Organization name (if applicable)

#### Add Examples

Consider adding:
- Jupyter notebook with analysis examples
- Sample visualizations
- Research questions the dataset addresses

**Example notebook structure:**

```markdown
# LLM Education Impact Analysis

## 1. Load Data
## 2. Exploratory Analysis
## 3. Tool Comparison
## 4. Student Trait Effects
## 5. Agency Preservation
## 6. Verification Effectiveness
```

### 5. Quality Checks

#### Before GitHub Release

```bash
# Run tests
python llm_education_simulator.py --students 20 --tasks 5

# Check code quality (optional)
pip install pylint
pylint llm_education_simulator.py

# Format code (optional)
pip install black
black llm_education_simulator.py
```

#### Before Hugging Face Upload

```bash
# Validate CSVs
python -c "
import pandas as pd
df = pd.read_csv('output_huggingface/runs.csv')
print(f'Runs: {len(df)} rows, {len(df.columns)} columns')
print(f'Missing values: {df.isnull().sum().sum()}')
print(f'Tool modes: {df.tool_mode.value_counts()}')
"

# Check file sizes
du -h output_huggingface/*.csv

# Verify metadata
python -c "
import json
with open('output_huggingface/metadata.json') as f:
    meta = json.load(f)
print(json.dumps(meta, indent=2))
"
```

### 6. Maintenance

#### Versioning Strategy

- **Major version (X.0)**: Breaking changes to data schema
- **Minor version (x.Y)**: New features, additional fields
- **Patch version (x.y.Z)**: Bug fixes, parameter tweaks

#### Updating the Dataset

When releasing new version:

```bash
# Update version in config
# Regenerate dataset
python llm_education_simulator.py --seed 42

# Tag new release
git tag -a v2.1 -m "Version 2.1: Added temporal dynamics"
git push origin v2.1

# Update Hugging Face
huggingface-cli upload dbbun/llm-education-impact output/ \
  --repo-type dataset --revision v2.1
```

#### Responding to Issues

Monitor:
- GitHub Issues for bug reports
- Hugging Face Community tab for questions
- Paper citations for feedback

## Common Issues & Solutions

### Issue: Figures Not Generating

**Cause**: matplotlib not installed or display not available

**Solution**:
```bash
pip install matplotlib seaborn
# OR run without figures
python llm_education_simulator.py --no-figures
```

### Issue: CSV Files Too Large for GitHub

**Cause**: GitHub has 100MB file size limit

**Solution**: Use Git LFS or host on Hugging Face only
```bash
git lfs install
git lfs track "*.csv"
```

### Issue: Hugging Face Upload Fails

**Cause**: Authentication or network issues

**Solution**:
```bash
# Re-authenticate
huggingface-cli whoami
huggingface-cli login --token YOUR_TOKEN

# Upload with verbose logging
huggingface-cli upload YOUR_ORG/llm-education-impact . \
  --repo-type dataset --verbose
```

### Issue: Results Not Reproducible

**Cause**: Random seed not set or dependencies differ

**Solution**:
- Always specify `--seed` parameter
- Pin exact versions in requirements.txt:
```txt
numpy==1.24.3
pandas==2.0.2
```

## Publishing Checklist

### Pre-Publication

- [ ] Code reviewed and tested
- [ ] Documentation complete and accurate
- [ ] Example outputs verified
- [ ] License approved
- [ ] Citations checked
- [ ] Contact information current

### GitHub Release

- [ ] Repository created
- [ ] README.md comprehensive
- [ ] Code committed and pushed
- [ ] Version tagged
- [ ] Release notes written
- [ ] Example files attached
- [ ] Topics/tags added

### Hugging Face Release

- [ ] Dataset generated with fixed seed
- [ ] All CSVs validated
- [ ] README.md formatted as dataset card
- [ ] Metadata complete
- [ ] Files uploaded
- [ ] Community tab monitored
- [ ] Dataset card reviewed

### Post-Publication

- [ ] Announce on relevant channels
- [ ] Monitor for issues
- [ ] Respond to questions
- [ ] Plan future updates
- [ ] Track citations/usage

## Support & Community

### Getting Help

- **GitHub Issues**: Bug reports, feature requests
- **Hugging Face Discussions**: Dataset usage questions
- **Paper Authors**: Theoretical questions

### Contributing

Encourage contributions:
- Documentation improvements
- Bug fixes
- New features
- Analysis examples
- Validation studies

## Example Announcement

**For Twitter/X:**

```
📊 New dataset: LLM Education Impact Simulator

Based on @DJackson's research on AI in education, this synthetic dataset models how students use LLMs vs search engines for learning tasks.

🔬 120 students, 600 runs, 12K events
📈 Tracks agency, engagement, errors
🎯 Great for ed-tech research

GitHub: [URL]
HF: [URL]

#EdTech #AI #Education
```

**For Reddit/Forums:**

```
I've released a research-grounded dataset simulating student interactions 
with AI tools in education, based on Daniel Jackson's recent paper "LLMs 
are not calculators."

The dataset includes behavioral traits, tool choices, errors, and learning 
outcomes for 120 synthetic students completing 600 tasks.

Key features:
- 3 tool modes (search, explicit LLM, agentic LLM)
- Agency and engagement metrics
- Verification behavior tracking
- Module boundary violations

Useful for:
- Ed-tech research
- AI impact analysis
- Pedagogical strategy development

Available on GitHub and Hugging Face. Feedback welcome!

[Links]
```

## Legal & Ethical Considerations

### License

MIT License allows:
- ✓ Commercial use
- ✓ Modification
- ✓ Distribution
- ✓ Private use

Requires:
- License and copyright notice inclusion

### Citation Requirements

While not legally required, academic courtesy suggests:
- Cite Jackson (2025) paper
- Cite dataset if used in publications
- Acknowledge any modifications

### Ethical Use

The dataset should:
- Support educational improvement
- Inform evidence-based policy
- Respect student privacy (no real data)

The dataset should NOT:
- Stereotype students based on traits
- Make high-stakes decisions without validation
- Replace real educational research

## Resources

### Documentation
- Jackson (2025) paper
- Simulation code comments
- Generated README files

### Tools
- Hugging Face CLI: https://huggingface.co/docs/huggingface_hub/
- Git LFS: https://git-lfs.github.com/
- Python packaging: https://packaging.python.org/

### Examples
- Pandas documentation
- Matplotlib gallery
- Educational data science tutorials

---

**© 2026 DBbun LLC. All rights reserved.**
