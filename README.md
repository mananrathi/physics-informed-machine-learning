# Physics-Informed Machine Learning  
## Battery Health Prediction

This repository contains code and data for developing **physics-informed machine learning models** to estimate key lithium-ion battery health metrics **Remaining Useful Life (RUL)**, **State of Health (SoH)**, and **State of Charge (SoC)** with a focus on electric vertical take-off and landing (**eVTOL**) aircraft applications.

---

## Data Set

- The dataset is based on a realistic battery duty profile designed for eVTOL operations, using high-energy-density pouch cells.
- This includes telemetry from **22 lithium-ion cells**, covering **21,392 charge/discharge cycles** under varied temperature, load, and operational conditions.
- Collected signals include:
  - **Voltage**
  - **Current**
  - **Temperature**
  - **Capacity**
  - **Cycle Number**
  - **Time Stamp**

This dataset forms the basis for hybrid physics-informed and data-driven modeling approaches.

---

## Machine Learning Models

This project applies supervised machine learning to predict battery capacity degradation and lifecycle metrics using:

- **Random Forest Regressor** (`scikit-learn`)
  
### Model Configuration
- **Algorithm**: RandomForestRegressor  
- **Hyper-Parameters**:
  - `n_estimators = 100`
  - `random_state = 42`
  - Train/Test Split: 50/50  
- **In-Put Features**: 
  - Time, Voltage, Current, Temperature, Cycle Number  
- **Target Variable**: 
  - Capacity

### Evaluation Metrics
- **Mean Squared Error (MSE)**
- **R² Score**

---

## Scripts

| Script                          | Description                                                             |
|----------------------------------|-------------------------------------------------------------------------|
| `allFiles_wCycleNum.py`         | all dataset files with embedded cycle number metadata         |
| `allFiles_woCycleNum.py`        | all dataset files without cycle number metadata               |
| `individualFile_wCycleNum_20.py`| individual files with cycle numbers in 20-cycle segments      |
| `individualFile_wCycleNum_50.py`| individual files with cycle numbers in 50-cycle segments      |
| `individualFile_woCycleNum_20.py`| individual files without cycle numbers in 20-cycle segments   |
| `individualFile_woCycleNum_50.py`| individual files without cycle numbers in 50-cycle segments   |

These scripts enable modular experimentation and batch analysis depending on dataset structure and modeling granularity.

---

## Usage

1. **Clone The Repository**
    ```bash
    git clone https://github.com/mananrathi/physics-informed-machine-learning.git
    cd physics-informed-machine-learning
    ```

2. **Extract The Data Set**  
   Unzip the `Carnegie Melon Dataset.zip` into the root project directory.

3. **Install Dependencies**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

4. **Run Processing Script**  
   For example, to process all files with cycle numbers:
    ```bash
    python allFiles_wCycleNum.py
    ```

---

## Acknowledgments

- Dataset and original experimental framework developed by **Carnegie Mellon University**.
- This project contributes to research on **physics-informed machine learning pipelines** for battery monitoring and prognostics in electric aviation.

For questions, suggestions, or contributions, feel free to open an issue or submit a pull request.
