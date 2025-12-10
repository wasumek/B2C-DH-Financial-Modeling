# Early Viability Assessment of a Business-to-Consumer (B2C) model for Digital Diabetes Screening in Switzerland

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey.svg)](https://osf.io/7dbgn)

A comprehensive Monte Carlo financial model for evaluating digital health subscription businesses in the Swiss market, with focus on diabetes prevention services.

**Author:** Wasu Mekniran  
**Institution:** ETH Zurich, Department of Management, Technology, and Economics  
**Contact:** wmekniran@ethz.ch  
**Date:** December 9, 2025  
**License:** MIT

---

## 🎯 Overview

This model provides rigorous financial analysis for digital health subscription services targeting diabetes prevention in Switzerland. It combines:

- **Monte Carlo simulation** (5,000+ paths) to quantify uncertainty
- **Epidemiological modeling** of Swiss at-risk population
- **Realistic adoption dynamics** based on Swiss digital health data
- **Comprehensive cost structure** including Swiss regulatory compliance
- **Publication-ready visualizations** for academic and business use

### Key Features

✅ **Academically rigorous** - Triangular distributions, proper NPV/IRR calculations  
✅ **Swiss market calibrated** - USPSTF guidelines, CE MDR, FADP compliance  
✅ **Fully reproducible** - Fixed random seed, documented assumptions  
✅ **Extensible architecture** - Modular design for easy customization  
✅ **Professional outputs** - Excel reports, high-resolution PDF plots

---

## 🚀 Quick Start

### Installation

```bash
# Clone or download the repository
# Install dependencies
pip install numpy numpy_financial pandas matplotlib seaborn openpyxl
```

### Run Analysis

```bash
python main.py
```

**Output:**
- Console: Detailed statistics and tables
- `model_results.xlsx`: Complete results workbook
- `plots/`: 9 publication-ready PDF visualizations

**Runtime:** 2-5 minutes on standard hardware

---

## 📁 Repository Structure

```
.
├── financial_model.py      # Core calculation engine
├── results_output.py        # Output formatting and printing
├── visualization.py         # Plotting functions
├── main.py                  # Main execution script
├── README.md               # This file
├── LICENSE                 # MIT License    
└── requirements.txt        # Python dependencies
```

---

## 📊 Model Overview

### Business Model

**Target Population:** Swiss adults at risk for diabetes
- USPSTF eligible population (30-50% of adults)
- Impaired Fasting Glucose (IFG): 10-12%
- Undiagnosed diabetes: 11-80%

**Service:** Digital coaching + monitoring subscription
- Price: CHF 20-60/month
- Retention: 30-60% annual (45% churn base case)

### Financial Projections

**Time Horizon:** 7 years with terminal value  
**Approach:** Monte Carlo with triangular distributions  
**Key Metrics:**
- Net Present Value (NPV)
- Internal Rate of Return (IRR)
- Breakeven year probability

### Cost Structure

**Variable Costs:**
- Customer acquisition (CAC): CHF 150-350
- Call center: CHF 15-30/user/year
- Backend services: CHF 25-45/user/year

**Fixed Costs:**
- Technician salaries: CHF 85k-110k
- Manager salaries: CHF 100k-140k
- App development: CHF 250k-400k (Year 1)
- CE MDR certification: CHF 10k-30k (one-time)
- FADP compliance: CHF 15k-40k/year

---

## 📈 Key Results (Base Case)

**Monte Carlo Summary** (5,000 simulations, seed=42):

| Metric | Median | 95% CI |
|--------|--------|--------|
| NPV | [Run model for results] | [2.5%, 97.5%] |
| IRR | [Run model for results] | [2.5%, 97.5%] |
| Breakeven Probability | [Run model for results] | - |

**Sensitivity Analysis:**
- Most influential parameter: [See tornado diagram]
- Key elasticities: Price × Churn, CAC × Screening

---

## 🔧 Customization

### Modify Parameters

Edit `ASSUMPTIONS` dictionary in `financial_model.py`:

```python
ASSUMPTIONS = {
    "price_per_month": {
        "min": 25,           # Adjust ranges
        "most_likely": 45,
        "max": 65,
        "distribution": "triangular",
    },
    # ... other parameters
}
```

### Run Custom Scenarios

```python
from financial_model import run_deterministic

# Test higher price, lower churn
npv, irr, breakeven = run_deterministic(
    overrides={
        "price_per_month": 50,
        "annual_churn_rate": 0.40
    }
)
```

### Extend Time Horizon

```python
# In financial_model.py
YEARS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Extend to 10 years
```

See the **Wiki** for comprehensive customization guide.

---

## 📚 Documentation

Complete documentation available in `WIKI.md`:

1. **Getting Started** - Installation and quick start
2. **Model Architecture** - Detailed methodology
3. **Parameter Reference** - All 20+ parameters explained
4. **Running the Model** - Standard and custom analyses
5. **Understanding Outputs** - Interpretation guide
6. **Customization Guide** - Extend and modify
7. **Troubleshooting** - Common issues and solutions
8. **Academic Citation** - How to cite this work

---

## 📖 Citation

### BibTeX

```bibtex
@software{mekniran2025_digital_health_model,
  author = {Mekniran, Wasu},
  title = {Early Viability Assessment of a Business-to-Consumer (B2C) model for Digital Diabetes Screening in Switzerland},
  year = {2025},
  month = {12},
  version = {1.0},
  publisher = {Open Science Framework},
  doi = {[DOI will be assigned by OSF]},
  institution = {ETH Zurich},
}
```

### APA

Mekniran, W. (2025). *Early Viability Assessment of a Business-to-Consumer (B2C) model for Digital Diabetes Screening in Switzerland* (Version 1.0) [Computer software]. Open Science Framework. https://osf.io/7dbgn

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

1. **Additional analyses** - Real options, scenario trees
2. **Visualization enhancements** - Interactive dashboards
3. **Documentation** - Tutorials, case studies
4. **Validation** - Comparison with real market data

**Process:**
1. Open an issue to discuss proposed changes
2. Fork the repository
3. Create a feature branch
4. Submit pull request with clear description

---

## 📄 License

MIT License

Copyright (c) 2025 Wasu Mekniran, ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 📞 Contact

**Wasu Mekniran**  
ETH Zurich  
Department of Management, Technology, and Economics  
Email: wmekniran@ethz.ch

**OSF Repository:** https://osf.io/7dbgn

---

## 🙏 Acknowledgments

- ETH Zurich & University of St.Gallen for institutional support
- Swiss diabetes prevention literature for epidemiological parameters
- Digital health market research for adoption rate calibration

---

*Last updated: December 10, 2025*  
*Version: 1.0*
