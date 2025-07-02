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
file_paths = ["VAH01_r2.csv", "VAH02_r2.csv", "VAH05_r2.csv", "VAH06_r2.csv", "VAH07_r2.csv", "VAH09_r2.csv",
              "VAH10_r2.csv", "VAH11_r2.csv", "VAH12_r2.csv", "VAH13_r2.csv", "VAH15_r2.csv", "VAH16_r2.csv",
              "VAH17_r2.csv", "VAH20_r2.csv", "VAH22_r2.csv", "VAH23_r2.csv", "VAH24_r2.csv", "VAH25_r2.csv",
              "VAH26_r2.csv", "VAH27_r2.csv", "VAH28_r2.csv", "VAH30_r2.csv"]
              
# iterate over each file
for file_path in tqdm(file_paths, desc="processing files"):
    # read data from csv
    data = pd.read_csv("Carnegie Melon Dataset/" + file_path)
    
    # extract relevant columns
    time = data.iloc[:, 0].values.reshape(-1, 1)
    voltage = data.iloc[:, 1].values.reshape(-1, 1)
    current = data.iloc[:, 2].values.reshape(-1, 1)
    temperature = data.iloc[:, 7].values.reshape(-1, 1)
    cycle_number = data.iloc[:, 11].values.reshape(-1, 1)
    capacity = data.iloc[:, 10].values.reshape(-1, 1)
    
    # concatenate input features
    X = np.concatenate((time, voltage, current, temperature, cycle_number), axis=1)

    # set target labels
    y = capacity
    
   # split the data into training and testing sets
    indices = np.arange(len(y))
    x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.5, random_state=seed)
    
    # create and train the model
    rf_regressor = RandomForestRegressor(n_estimators=num_estimators, random_state=seed)
    rf_regressor.fit(x_train, y_train.ravel())
    
    # make predictions
    y_pred = rf_regressor.predict(x_test)
    
    # get evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'for {file_path}:')
    print(f'mse: {mse}')
    print(f'r^2 score: {r2}')
    
    # plot results
    plt.scatter(y_test, y_pred)
    plt.xlabel('true capacity')
    plt.ylabel('predicted capacity')
    plt.title(f'true vs predicted capacity ({file_path})')
   
    plt.gcf().set_size_inches(8, 6)

    plt.subplots_adjust(bottom=0.35)

    textstr = '\n'.join((
        f'r^2: {r2}',
        f'mse: {mse}',
        f'num estimators: {num_estimators}',
        f'train/test split: 50/50',
        f'seed: {seed}',
        f'input features: time, voltage, current, temperature, cycle_number'))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    plt.figtext(0.5, 0.05, textstr, ha="center", fontsize=10, bbox=props)
    
    # the following code save the plot as an image in your desired location
    # save_directory = "/users/mananrathi/desktop"
    # filename = f"plot_{file_path.replace('.csv', '')}.png"
    # full_path = os.path.join(save_directory, filename)

    # plt.savefig(full_path)


    plt.show()
    
    # plot the measured capacity (true values) versus time
    plt.plot(time[indices_test], y_test, label='measured capacity', color='blue', marker='o')

    # plot the predicted capacity versus time
    plt.plot(time[indices_test], y_pred, label='predicted capacity', color='red', linestyle='--', marker='x')

    plt.xlabel('time')
    plt.ylabel('capacity')
    plt.title(f'capacity over time ({file_path})')
    plt.legend()
    
    plt.gcf().set_size_inches(8, 6)  # width, height in inches

    plt.subplots_adjust(bottom=0.35)

    textstr = '\n'.join((
        f'r^2: {r2}',
        f'mse: {mse}',
        f'num estimators: {num_estimators}',
        f'train/test split: 50/50',
        f'seed: {seed}',
        f'input features: time, voltage, current, temperature, cycle number'))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    plt.figtext(0.5, 0.05, textstr, ha="center", fontsize=10, bbox=props)

    # the following code save the plot as an image in your desired location
    # save_directory = "/users/mananrathi/desktop"
    # filename = f"plot_capacity_vs_time_{file_path.replace('.csv', '')}.png"
    # full_path = os.path.join(save_directory, filename)
    # plt.savefig(full_path)
    
    plt.show()