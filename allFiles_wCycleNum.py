import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# hyperparameters
seed = 42
num_estimators = 100

# read data from csv
file_paths = ["VAH25_r2.csv", "VAH16_r2.csv"]
# select one file for testing and use the rest for training
test_file = "VAH07_r2.csv"
train_files = [f for f in file_paths if f != test_file]

# placeholder for combined training data
combined_X_train = np.empty((0, 5))
combined_y_train = np.empty((0, 1))

# combine the training data from multiple files
for file_path in tqdm(file_paths, desc="processing files"):
    data = pd.read_csv("Carnegie Melon Dataset/" + file_path)
    time = data.iloc[:, 0].values.reshape(-1, 1)
    voltage = data.iloc[:, 1].values.reshape(-1, 1)
    current = data.iloc[:, 2].values.reshape(-1, 1)
    temperature = data.iloc[:, 7].values.reshape(-1, 1)
    cycle_number = data.iloc[:, 11].values.reshape(-1, 1)
    capacity = data.iloc[:, 10].values.reshape(-1, 1)

    X = np.concatenate((time, voltage, current, temperature, cycle_number), axis=1)
    y = capacity
    
    combined_X_train = np.vstack((combined_X_train, X))
    combined_y_train = np.vstack((combined_y_train, y))

# train the model using the combined training data
rf_regressor = RandomForestRegressor(n_estimators=num_estimators, random_state=seed)
rf_regressor.fit(combined_X_train, combined_y_train.ravel())

# read the test data
test_data = pd.read_csv("Carnegie Melon Dataset/" + test_file)
X_test = np.concatenate((test_data.iloc[:, 0].values.reshape(-1, 1),
                         test_data.iloc[:, 1].values.reshape(-1, 1),
                         test_data.iloc[:, 2].values.reshape(-1, 1),
                         test_data.iloc[:, 7].values.reshape(-1, 1),
                         test_data.iloc[:, 11].values.reshape(-1, 1)), axis=1)
y_test = test_data.iloc[:, 10].values.reshape(-1, 1)

# make predictions on the test set
y_pred = rf_regressor.predict(X_test)

# evaluate the model's performance on the test set
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'test results for {test_file}:')
print(f'mse: {mse}')
print(f'r^2 score: {r2}')

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, label='predicted vs true capacity')
plt.xlabel('true capacity')
plt.ylabel('predicted capacity')
plt.title(f'predicted vs true capacity for {test_file}')
plt.legend()

plt.gcf().set_size_inches(8, 6)

plt.subplots_adjust(bottom=0.35)

textstr = '\n'.join((
    f'r^2: {r2}',
    f'mse: {mse}',
    f'num estimators: {num_estimators}',
    f'seed: {seed}',
    f'input features: time, voltage, current, temperature, cycle_number',
    f'files used to train: {train_files}'))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

plt.figtext(0.5, 0.05, textstr, ha="center", fontsize=10, bbox=props)
plt.show()