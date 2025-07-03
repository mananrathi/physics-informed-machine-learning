# Physics-Informed Machine Learning 
## Battery Degradation & Performance Prediction

This project implements physics-informed machine learning models to predict lithium-ion battery degradation and performance metrics, including Remaining Useful Life (RUL), State of Charge (SoC), and State of Health (SoH). The models are trained on comprehensive battery cycling data from high-density pouch cells under varied operational conditions, incorporating advanced feature engineering and real-time inference capabilities.

## Project

This repository contains multiple python scripts that implement machine learning models for battery health monitoring and degradation prediction. The project explores different training strategies using cycle-level telemetry data from voltage, current, temperature, and capacity signals. The models are designed for real-time inference pipelines suitable for onboard integration in electric aviation and automotive applications.

## Dataset

The project utilizes comprehensive battery cycling data from 22 high-density pouch cells, encompassing over 21,000 charge-discharge cycles under varied operational conditions. This dataset provides rich telemetry for developing robust battery health monitoring models suitable for real-world applications including electric aviation.

**Data Features:**
- **Time**: Timestamp of the measurement
- **Voltage**: Battery voltage (V) - critical for SoC estimation
- **Current**: Battery current (A) - key for power and thermal modeling
- **Temperature**: Operating temperature (°C) - affects degradation patterns
- **Cycle Number**: Charge/discharge cycle count - essential for RUL prediction
- **Capacity**: Battery capacity (Ah) - primary indicator of State of Health (SoH)

## File Structure

```
├── allFiles_wCycleNum.py             # Multi-file training with cycle number
├── allFiles_woCycleNum.py            # Multi-file training without cycle number
├── individualFile_wCycleNum_20.py    # Single-file training (80/20 split) with cycle number
├── individualFile_wCycleNum_50.py    # Single-file training (50/50 split) with cycle number
├── individualFile_woCycleNum_20.py   # Single-file training (80/20 split) without cycle number
├── individualFile_woCycleNum_50.py   # Single-file training (50/50 split) without cycle number
├── Carnegie Melon Dataset.zip        # Battery dataset
├── requirements.txt                  # Python dependencies
└── README.md                       
```

## Scripts Description

### Multi-File Training Scripts

1. **`allFiles_wCycleNum.py`**
   - Implements multi-battery training for cross-cell generalization
   - Uses engineered features: time, voltage, current, temperature, cycle number
   - Incorporates temporal normalization and anomaly rejection techniques
   - Evaluates model performance across different battery chemistries and conditions

2. **`allFiles_woCycleNum.py`**
   - Physics-based approach excluding cycle-specific information
   - Focuses on real-time measurable parameters for onboard applications
   - Uses features: time, voltage, current, temperature
   - Tests model robustness without historical cycle data

### Single-File Training Scripts

3. **`individualFile_wCycleNum_[20|50].py`**
   - Single-cell training for cell-specific degradation modeling
   - Incorporates cycle-level telemetry for detailed health assessment
   - Supports different data split strategies for validation (80/20 and 50/50)
   - Optimized for Remaining Useful Life (RUL) prediction

4. **`individualFile_woCycleNum_[20|50].py`**
   - Real-time inference pipeline using only instantaneous measurements
   - Designed for onboard State of Charge (SoC) and State of Health (SoH) estimation
   - Eliminates dependency on historical cycle data for deployment flexibility

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running The Scripts

Each script can be run independently:

```bash
# Multi-file training with cycle number
python allFiles_wCycleNum.py

# Multi-file training without cycle number  
python allFiles_woCycleNum.py

# Individual file training (various configurations)
python individualFile_wCycleNum_20.py
python individualFile_wCycleNum_50.py
python individualFile_woCycleNum_20.py
python individualFile_woCycleNum_50.py
```

### Model Parameters

All scripts implement optimized hyperparameters for battery health prediction:
- **Algorithm**: Random Forest Regressor (suitable for real-time inference)
- **Number of Estimators**: 100
- **Random Seed**: 42 (for reproducibility)
- **Feature Engineering**: Temporal normalization and anomaly rejection
- **Inference Optimization**: Designed for low-latency onboard applications

### Output

Each script provides comprehensive battery health metrics:
1. **Performance Metrics**: Mean Squared Error (MSE) and R² score for model validation
2. **Battery Health Indicators**:
   - Remaining Useful Life (RUL) predictions
   - State of Charge (SoC) estimation accuracy
   - State of Health (SoH) degradation tracking
3. **Visualization Plots**:
   - Predicted vs True capacity scatter plot with confidence intervals
   - Degradation curves over cycling history
   - Real-time inference performance metrics
4. **Model Diagnostics**: Feature importance analysis and prediction uncertainty quantification

## Results Interpretation

The scripts enable comprehensive analysis of battery health prediction approaches:

- **Feature Engineering Impact**: Compare models with and without cycle history to optimize feature selection for different deployment scenarios
- **Cross-Cell Generalization**: Evaluate model performance across different battery chemistries and operational conditions
- **Real-Time Performance**: Assess inference speed and accuracy for onboard integration requirements
- **Degradation Modeling**: Analyze RUL prediction accuracy and SoH estimation reliability
- **Temporal Analysis**: Understanding how model performance evolves with battery aging

## Key Features

- **Physics-Informed Architecture**: Incorporates domain knowledge from electrochemical principles and thermal dynamics
- **Real-Time Inference Pipeline**: Optimized for low-latency onboard applications in electric aviation
- **Advanced Feature Engineering**: Temporal normalization, anomaly rejection, and cycle-level telemetry processing
- **Multi-Scale Analysis**: Cross-cell generalization and individual cell-specific modeling
- **Battery Health Metrics**: Comprehensive RUL, SoC, and SoH estimation capabilities
- **Production-Ready Design**: Modular architecture suitable for integration into battery management systems
- **Robust Validation**: Multiple training strategies to ensure model reliability across operational conditions

## Dependencies

- `numpy`: High-performance numerical computations for signal processing
- `pandas`: Advanced data manipulation and time-series analysis
- `matplotlib`: Professional visualization for battery performance analysis
- `scikit-learn`: Machine learning algorithms optimized for regression tasks
- `tqdm`: Progress monitoring for large-scale battery data processing

*Note: Production deployment may require PyTorch for advanced neural network architectures and real-time inference optimization.*

## Future Enhancements

Research directions for advanced battery health monitoring:
- **Physics-Informed Neural Networks (PINNs)**: Incorporating electrochemical constraints into deep learning models
- **PyTorch Integration**: Advanced neural architectures for complex degradation pattern recognition
- **Real-Time Optimization**: Edge computing deployment for ultra-low latency inference
- **Multi-Modal Fusion**: Integration with thermal imaging and acoustic monitoring data
- **Uncertainty Quantification**: Bayesian approaches for reliable confidence intervals in safety-critical applications
- **Transfer Learning**: Domain adaptation across different battery chemistries and form factors
- **Federated Learning**: Privacy-preserving model updates across distributed battery fleets
