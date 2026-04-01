# Machine Learning-Based Misbehavior Detection for Connected Vehicles in Ottawa's V2X Network

**5G Vehicle Misbehavior Detection Using F2MD Framework**

[![Course](https://img.shields.io/badge/Course-ITEC%205910W-red)](https://carleton.ca)
[![University](https://img.shields.io/badge/University-Carleton-black)](https://carleton.ca)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![SUMO](https://img.shields.io/badge/SUMO-1.26.0-green)](https://sumo.dlr.de)
[![OMNeT++](https://img.shields.io/badge/OMNeT++-5.6.2-orange)](https://omnetpp.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [ML Results](#ml-results)
- [Key Findings](#key-findings)
- [Team](#team)
- [References](#references)
- [License](#license)

---

## Overview

This project builds an **end-to-end pipeline** for detecting misbehaving Connected and Autonomous Vehicles (CAVs) in Ottawa's downtown V2X network. We simulate realistic Ottawa traffic using real OpenStreetMap data, inject cyberattacks into 10% of vehicles using the F2MD framework, generate a dataset of **475,534 BSM messages**, and train Machine Learning models to detect attackers.

**Key Achievement:** XGBoost achieves **61.5% recall** with **0.915 AUC-ROC**, representing a **+28% improvement** over baseline models.

### Pipeline Flow

```
OpenStreetMap --> SUMO (Traffic) --> VEINS (Bridge) --> OMNeT++ (Network) --> F2MD (Attacks) --> ML (Detection)
```

---

## Problem Statement

Connected vehicles exchange **Basic Safety Messages (BSMs)** 10 times per second, sharing position, speed, heading, and acceleration. An **insider attacker** with valid PKI credentials can broadcast fake data — and standard authentication **cannot detect this** because the message is properly signed.

### Attack Types Supported

| Category | Attacks |
|----------|---------|
| Position Falsification | ConstPos, ConstPosOffset, **RandomPos** (primary), RandomPosOffset |
| Speed Falsification | ConstSpeed, ConstSpeedOffset, RandomSpeed, RandomSpeedOffset |
| Denial of Service | DoS, DoSRandom |
| Identity Attacks | Sybil, GridSybil |
| Replay/Disruption | DataReplay, StaleMessages, Disruptive, EventualStop |

**RandomPos** is the primary attack type used in our simulation.

---

## System Architecture

```
+---------------+     +----------------+     +-----------+     +----------------+
|    SUMO       |     |  OMNeT++ /     |     |   F2MD    |     |  ML Pipeline   |
|   Traffic     |---->|   VEINS        |---->|  Attack   |---->|  Detection     |
|  Simulation   |     |  Network Sim   |     | Injection |     |  (Python)      |
+---------------+     +----------------+     +-----------+     +----------------+
  Ottawa OSM          5.9 GHz, 20mW         19 attack          RF, XGBoost,
  200+ vehicles       IEEE 802.11p          types               GB, MLP
  300s duration       ~420m range           VeReMi output       SMOTE + tuning
```

### Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Map Source | OpenStreetMap (Ottawa Downtown) |
| Map Dimensions | 2,514m x 2,662m |
| Coordinates | 45.41N-45.44N, 75.68W-75.71W |
| Vehicles | 200 (ML dataset) / 213 (F2MD simulation) |
| Duration | 300s (ML) / 3,600s (F2MD) |
| Protocol | ITS-G5 (IEEE 802.11p) at 5.890 GHz |
| Tx Power | 20 mW |
| Beacon Rate | 10 Hz (0.1s interval) |
| Attacker Density | 10% (20 vehicles) |
| Attack Type | RandomPos (Type A3) |

---

## Project Structure

```
ottawa-cav-misbehavior-detection/
|
|-- README.md                          # This file
|-- requirements.txt                   # Python dependencies
|-- .gitignore                         # Git ignore rules
|
|-- scripts/                           # All executable scripts
|   |-- ottawa_map_generator.py        # Generate Ottawa SUMO map from OSM
|   |-- ml_misbehavior_detection.py    # Phase 3: Baseline ML models
|   |-- ml_optimization.py            # Phase 4: Optimized ML models
|
|-- configs/                           # Configuration files
|   |-- sumo/                          # SUMO traffic simulation configs
|   |   |-- ottawa.net.xml             # Ottawa road network (2514m x 2662m)
|   |   |-- ottawa.rou.xml             # Vehicle routes
|   |   |-- ottawa.poly.xml            # Building polygons
|   |   |-- ottawa.sumocfg             # SUMO configuration
|   |   |-- ottawa.trips.xml           # Trip definitions
|   |
|   |-- f2md/                          # F2MD/VEINS simulation configs
|       |-- OttawaScenario.ned         # OMNeT++ network definition
|       |-- omnetpp.ini                # Simulation + attack parameters
|       |-- ottawa.launchd.xml         # VEINS launch configuration
|       |-- ottawa.sumo.cfg            # SUMO config for F2MD
|       |-- config.xml                 # Radio propagation model
|       |-- antenna.xml                # Antenna configuration
|
|-- datasets/                          # Dataset files (generated, not in repo)
|   |-- .gitkeep
|
|-- models/                            # Trained ML models
|   |-- phase4/                        # Optimized models (.joblib)
|       |-- XGB_optimized.joblib       # Best model (XGBoost)
|       |-- GB_optimized.joblib        # Gradient Boosting
|       |-- MLP_optimized.joblib       # MLP Neural Network
|       |-- scaler_phase4.joblib       # Feature scaler
|
|-- results/                           # Experimental results
|   |-- phase3/
|   |   |-- ml_detection_results.json
|   |-- phase4/
|       |-- ml_optimization_results_phase4.json
|       |-- phase4_summary.txt
|
|-- plots/                             # Generated visualizations
|   |-- phase3/                        # Baseline result plots
|   |   |-- confusion_matrices.png
|   |   |-- feature_importance.png
|   |   |-- model_comparison.png
|   |   |-- performance_summary.png
|   |-- phase4/                        # Optimized result plots
|       |-- confusion_matrices_phase4.png
|       |-- performance_summary_phase4.png
|       |-- phase3_vs_phase4_comparison.png
|       |-- precision_recall_curves_phase4.png
|       |-- precision_recall_phase4.png
|       |-- recall_improvement.png
|       |-- roc_curves_phase4.png
|
|-- docs/                              # Documentation
    |-- .gitkeep
```

---

## Prerequisites

### Hardware Requirements

- **RAM:** 8 GB minimum (we used 7.1 GB + 4 GB swap)
- **Disk:** 20 GB free space
- **OS:** Ubuntu 20.04+ or 22.04 (tested on 22.04)

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| SUMO | 1.26.0 | Traffic simulation |
| OMNeT++ | 5.6.2 | Network simulation |
| VEINS | 5.2 | Vehicular networking framework |
| F2MD | latest | Misbehavior detection framework |
| Python | 3.10+ | ML pipeline |

---

## Installation

### Option 1: Quick Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/ottawa-cav-misbehavior-detection.git
cd ottawa-cav-misbehavior-detection

# Install Python dependencies
pip3 install -r requirements.txt
```

### Option 2: Full Environment Setup

**1. Install system dependencies:**

```bash
sudo apt-get update
sudo apt-get install -y build-essential gcc g++ bison flex perl \
  python3 python3-pip python3-venv qtbase5-dev qtchooser qt5-qmake \
  qtbase5-dev-tools libqt5opengl5-dev libxml2-dev zlib1g-dev \
  default-jre wget curl git unzip cmake
```

**2. Install SUMO:**

```bash
sudo add-apt-repository -y ppa:sumo/stable
sudo apt-get update && sudo apt-get install -y sumo sumo-tools sumo-doc
export SUMO_HOME="/usr/share/sumo"
```

**3. Install OMNeT++ 5.6.2:**

```bash
cd ~
wget https://github.com/omnetpp/omnetpp/releases/download/omnetpp-5.6.2/omnetpp-5.6.2-src-linux.tgz
tar xvfz omnetpp-5.6.2-src-linux.tgz
cd omnetpp-5.6.2
./configure WITH_TKENV=no WITH_QTENV=yes WITH_OSG=no
make -j$(nproc)
```

**4. Install VEINS 5.2:**

```bash
cd ~
wget https://veins.car2x.org/download/veins-5.2.zip
unzip veins-5.2.zip && cd veins-5.2
./configure && make -j$(nproc)
```

**5. Install F2MD:**

```bash
cd ~
git clone https://github.com/josephkamel/F2MD.git
cd F2MD && ./buildF2MD
```

**6. Install Python dependencies:**

```bash
pip3 install -r requirements.txt
```

---

## Usage

### Phase 1-2: Generate Ottawa Map and Dataset

```bash
cd scripts/
python3 ottawa_map_generator.py --area downtown --vehicles 200 --duration 300
```

### Phase 3: Run Baseline ML Models

```bash
python3 ml_misbehavior_detection.py
```

Output: Trained models in `models/phase3/`, results JSON, and plots in `plots/phase3/`

### Phase 4: Run Optimized ML Models

```bash
python3 ml_optimization.py
```

Output: Optimized models (including XGBoost), results JSON, and 7 plots in `plots/phase4/`

### Phase 5: Run VEINS/F2MD Simulation

**Terminal 1 - Start SUMO daemon:**

```bash
cd ~/F2MD
./launchSumoTraciDaemon
# Choose "Without GUI" for faster simulation
```

**Terminal 2 - Run simulation:**

```bash
cd ~/F2MD/veins-f2md/f2md-networks/OttawaScenario
opp_run -r 0 -m -u Cmdenv -c General \
  -n .:../../src/veins:../../src \
  -l ../../src/veins omnetpp.ini
```

Output: VeReMi trace files (2.8 GB), ground truth JSON (13 MB), OMNeT++ .vec/.sca files

---

## Dataset

### Dataset Summary

| Property | Value |
|----------|-------|
| Total BSM Messages | 475,534 |
| Total Vehicles | 200 (180 normal + 20 attackers) |
| Normal Messages | 426,814 (89.8%) |
| Attacker Messages | 48,720 (10.2%) |
| Class Imbalance | 8.7 : 1 |
| Features | 19 |
| File Size | 222 MB (CSV) |
| Time Range | 0.1s - 300.0s |

### 19 Features in 3 Categories

**Kinematic (10):** posX, posY, speed, acceleration, spdX, spdY, aclX, aclY, hedX, hedY

**Plausibility (6):** pos_delta, time_delta, implied_speed, speed_consistency, speed_delta, accel_plausible

**Noise (3):** pos_noise_mag, spd_noise_mag, msg_frequency

> **Note:** The dataset file is too large for GitHub (222 MB). Generate it using the provided scripts or contact the team for access.

---

## ML Results

### Phase 3 - Baseline (Default Settings, 3 Models)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 0.946 | 0.990 | 0.481 | 0.648 | 0.934 |
| SVM | 0.474 | 0.109 | 0.578 | 0.184 | 0.516 |
| MLP Neural Net | 0.947 | 0.925 | 0.521 | 0.667 | 0.902 |

**Issue:** Low recall — models missed 48-52% of attackers due to 8.7:1 class imbalance.

### Phase 4 - Optimized (SMOTE + XGBoost + Threshold Tuning)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Threshold |
|-------|----------|-----------|--------|----------|---------|-----------|
| Random Forest | 0.942 | 0.841 | 0.536 | 0.655 | 0.905 | 0.65 |
| **XGBoost** | **0.931** | **0.678** | **0.615** | **0.645** | **0.915** | **0.85** |
| Gradient Boosting | 0.936 | 0.759 | 0.545 | 0.635 | 0.893 | 0.55 |
| MLP Neural Net | 0.933 | 0.804 | 0.451 | 0.578 | 0.861 | 0.85 |

**Best Model: XGBoost** with +28% recall improvement (0.481 -> 0.615).

### Optimization Techniques Applied

| Technique | Why | Impact |
|-----------|-----|--------|
| SMOTE | Fix 8.7:1 class imbalance | +8-15% recall |
| Threshold Tuning | Better precision-recall balance | +5-12% recall |
| Class Weights | Penalize minority misclassification | +3-8% recall |
| Hyperparameter Search | Optimize model config | +5% recall (RF) |
| Model Replacement | Replace failed SVM | XGB: best overall |

### Phase 5 - F2MD Simulation Results

| Parameter | Value |
|-----------|-------|
| Map | Ottawa Downtown (2514m x 2662m) |
| Duration | 3600 seconds |
| Protocol | ITS-G5 (IEEE 802.11p) |
| Total Vehicles | 213 |
| Genuine (A0) | 180 (90%) |
| Attackers (A3) | 20 (10%) |
| Attack Type | RandomPos (position falsification) |
| Output Size | 2.8 GB (VeReMi format) |
| Ground Truth | 13 MB labeled JSON |

### Top 10 Feature Importances

| Rank | Feature | Importance | Category |
|------|---------|-----------|----------|
| 1 | posY | 0.125 | Kinematic |
| 2 | posX | 0.098 | Kinematic |
| 3 | hedX | 0.088 | Kinematic |
| 4 | spd_noise_mag | 0.082 | Noise |
| 5 | pos_noise_mag | 0.078 | Noise |
| 6 | hedY | 0.075 | Kinematic |
| 7 | speed | 0.070 | Kinematic |
| 8 | spdY | 0.062 | Kinematic |
| 9 | spdX | 0.058 | Kinematic |
| 10 | pos_delta | 0.045 | Plausibility |

---

## Key Findings

1. **Class imbalance is the #1 challenge.** Without SMOTE and threshold tuning, even 94% accuracy hides that half of attackers go undetected. Always check recall for security tasks.

2. **XGBoost is the best model.** Built-in class weighting, regularization, and well-calibrated probabilities make it ideal. Threshold 0.85 gives the best recall-precision balance.

3. **Position features are the strongest indicators.** posX + posY = 22.3% combined importance. Ottawa's road grid constrains where vehicles can be, exposing fake positions.

4. **Full pipeline works end-to-end.** SUMO to OMNeT++/VEINS to F2MD to ML successfully generates Ottawa-specific V2X misbehavior data and detects attacks.

---

## Future Work

- Train on F2MD output directly using the full 2.8 GB VeReMi simulation data
- Test more attack types including Sybil, DoS, DataReplay, and speed falsification
- Implement real-time detection by running XGBoost during live F2MD simulation
- Extend to 5G NR-V2X via Simu5G for actual 5G New Radio simulation
- Evaluate deep learning architectures (LSTM and Transformer)
- Multi-city comparison with Toronto, Montreal, and other Canadian cities

---

## Team

| Member | Role |
|--------|------|
| Monthe Tomta Freddy Parimi | Problem definition, system design, V2X background |
| Venkata Naga Teja Sai | Ottawa map generation, SUMO setup, dataset and features |
| Patel Shulankkumar Prakashkumar | ML pipeline (Phase 3 and 4), model optimization |
| Soni Krushant Hemantkumar | VEINS/F2MD simulation, analysis, future work |

**Supervised by:** Prof. Jun (Steed) Huang and Stephen Rayment

**Course:** ITEC 5910W - Selected Topics in Network Technologies (5G Networks)

**Institution:** Carleton University - School of Information Technology - Winter 2026

---

## References

1. R. van der Heijden et al., "Survey on misbehavior detection in cooperative ITS," IEEE COMST, 2019.
2. R. van der Heijden et al., "VeReMi: A dataset for misbehavior detection in VANETs," SecureComm, 2018.
3. J. Kamel et al., "CaTch: Confidence range tolerant misbehavior detection," IEEE WCNC, 2019.
4. J. Kamel et al., "VeReMi extension dataset," IEEE ICC, 2020.
5. C. Sommer et al., "Bidirectionally coupled network and road traffic simulation," IEEE TMC, 2011.
6. P. A. Lopez et al., "Microscopic traffic simulation using SUMO," IEEE ITSC, 2018.
7. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," ACM SIGKDD, 2016.
8. N. V. Chawla et al., "SMOTE: Synthetic minority over-sampling technique," JAIR, 2002.
9. 3GPP TS 23.287, "Architecture enhancements for 5G System to support V2X services," v17.5.0, 2022.
10. ETSI TS 103 759, "Misbehavior Detection in ITS; Release 2," 2021.

---

## License

This project is developed for academic purposes as part of ITEC 5910W at Carleton University.
Licensed under the MIT License.

---

<p align="center">
  <b>Carleton University - School of Information Technology - Winter 2026</b>
</p>
