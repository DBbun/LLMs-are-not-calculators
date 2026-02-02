# DBbun LLC - LLM Education Impact Simulator

**Quick Reference Card**

---

## Organization Details

**Company:** DBbun LLC  
**Website:** https://github.com/dbbun  
**Project:** LLM Education Impact Simulator  
**Version:** 1.0  
**License:** apache-2.0  

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/dbbun/llm-education-simulator.git
cd llm-education-simulator

# Install dependencies
pip install -r requirements.txt

# Run simulation
python llm_education_simulator.py

# Or customize
python llm_education_simulator.py --students 200 --tasks 20 --seed 42
```

---

## What It Does

Generates synthetic data simulating how students interact with AI tools in education:
- **3 tool modes:** Search, Explicit LLM, Agentic LLM
- **120 students** with behavioral traits
- **600+ runs** of task completion
- **12,000+ events** tracking all activities

Based on research by Daniel Jackson (MIT, 2025) on LLM usage in undergraduate courses.

---

## Key Outputs

- `students.csv` - Student characteristics
- `tasks.csv` - Educational tasks
- `runs.csv` - Complete interaction records
- `events.csv` - Fine-grained activity log
- `summary.json` - Statistics
- `metadata.json` - Schema and documentation
- `fig1-8.png` - Visualizations

---

## Research Applications

1. How do different AI tools affect learning outcomes?
2. Which student traits predict successful AI usage?
3. How does context selection quality relate to errors?
4. What behaviors preserve student agency?
5. Does intentionality mediate tool effects on learning?

---

## Citation

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

---

## File Checklist for Deployment

### Core Files
- [x] llm_education_simulator.py (main code)
- [x] requirements.txt (dependencies)
- [x] LICENSE (apache-2.0 with DBbun LLC)
- [x] README.md (comprehensive docs)
- [x] .gitignore (proper exclusions)

### Documentation
- [x] DEPLOYMENT.md (step-by-step guide)
- [x] example_config.json (example)

### Generated on Run
- [ ] students.csv
- [ ] tasks.csv
- [ ] runs.csv
- [ ] events.csv
- [ ] summary.json
- [ ] metadata.json
- [ ] config.json
- [ ] README.md (dataset-specific)
- [ ] fig1-8.png (8 figures)

---

## Version History

### v1.0 (2026-02-02) - Current
- Complete rewrite based on Jackson (2025)
- Added reading behavior, intentionality, context quality
- Enhanced theoretical grounding
- Production-ready deployment package
- **© 2026 DBbun LLC**

### v1.0 (Previous)
- Initial concept and basic simulation

---

## Legal

**Copyright:** © 2026 DBbun LLC. All rights reserved.

**License:** MIT License - See LICENSE file for details

**Disclaimer:** This is a research tool generating synthetic data. Use thoughtfully in educational policy and research contexts.

---

**Last Updated:** 2026-02-02  
**Status:** ✅ Production Ready  
**Maintainer:** DBbun LLC
