# Physics-Informed Battery Health Prediction

This repository contains code and data for developing **physics-informed machine learning models** to estimate key lithium-ion battery health metrics—**Remaining Useful Life (RUL)**, **State of Health (SoH)**, and **State of Charge (SoC)**—with a focus on electric vertical take-off and landing (**eVTOL**) aircraft applications.

## 📊 Dataset

- The dataset is based on a realistic battery duty profile designed for eVTOL operations, using high-energy-density pouch cells.
- It includes telemetry from **22 lithium-ion cells**, covering **21,392 charge/discharge cycles** under varied temperature, load, and operational conditions.
- Signals include **voltage**, **current**, **temperature**, and **capacity**, used to predict:
  - Remaining Useful Life (RUL)
  - State of Health (SoH)
  - State of Charge (SoC)

This dataset forms the basis for hybrid physics-informed and data-driven modeling approaches.

## 🧠 Scripts Overview

| Script                          | Description                                                             |
|----------------------------------|-------------------------------------------------------------------------|
| `allFiles_wCycleNum.py`         | Processes all dataset files with embedded cycle number metadata         |
| `allFiles_woCycleNum.py`        | Processes all dataset files without cycle number metadata               |
| `individualFile_wCycleNum_20.py`| Processes individual files with cycle numbers in 20-cycle segments      |
| `individualFile_wCycleNum_50.py`| Processes individual files with cycle numbers in 50-cycle segments      |
| `individualFile_woCycleNum_20.py`| Processes individual files without cycle numbers in 20-cycle segments   |
| `individualFile_woCycleNum_50.py`| Processes individual files without cycle numbers in 50-cycle segments   |

These scripts enable modular experimentation and batch analysis depending on dataset structure and granularity.

## ⚙️ Usage

1. **Clone the Repository**
    ```bash
    git clone https://github.com/mananrathi/battery-metric-prediction.git
    cd battery-metric-prediction
    ```

2. **Extract the Dataset**  
   Unzip the `Carnegie Melon Dataset.zip` into the root project directory.

3. **Install Dependencies**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

4. **Run a Processing Script**  
   For example, to process all files with cycle numbers:
    ```bash
    python allFiles_wCycleNum.py
    ```

## 🧾 Acknowledgments

- Dataset and original experimental framework developed by **Carnegie Mellon University**.
- This project contributes to research on **physics-informed machine learning pipelines** for battery monitoring and prognostics in electric aviation.

For questions, suggestions, or contributions, feel free to open an issue or submit a pull request.