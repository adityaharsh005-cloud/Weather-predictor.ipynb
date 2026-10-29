# Weather Prediction Using Machine Learning

## 1. Project Overview

This project uses machine learning to predict future air temperature from historical weather observations. It explores different regression algorithms, compares their predictive performance, and develops a forecasting system that predicts temperatures at multiple points over the next seven days.

The project uses hourly weather data obtained from the Open-Meteo Historical Weather API. Python libraries are used for data preprocessing, feature engineering, model training, evaluation, and visualization.

The main objective is to investigate how historical temperature patterns, weather variables, and time-based features can help predict future temperatures.

## 2. Project Objectives

- Collect and analyze historical hourly weather data.
- Clean and preprocess weather observations.
- Create time-based features and lagged weather variables.
- Train and compare different machine learning models.
- Evaluate model performance using MAE and RMSE.
- Visualize actual versus predicted temperatures.
- Analyze feature importance using XGBoost.
- Generate temperature forecasts at eight-hour intervals for up to seven days ahead.

## 3. Dataset

The project uses historical hourly weather data from the Open-Meteo Archive API.

**Weather variables collected:**
- `temperature_2m`: Air temperature at 2 metres above ground.
- `relative_humidity_2m`: Relative humidity at 2 metres.
- `precipitation`: Precipitation.
- `pressure_msl`: Mean sea-level pressure.
- `wind_speed_10m`: Wind speed at 10 metres.

The initial dataset covers **1 January 2015 to 31 December 2024**, with hourly observations.

The notebook also retrieves more recent weather data for forecasting experiments.

**Data source:** [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)

## 4. Data Preprocessing and Feature Engineering

### Data preprocessing

The notebook performs several preprocessing steps:

1. Loads the weather dataset into a Pandas DataFrame.
2. Checks the dataset's dimensions, column names, data types, and missing values.
3. Converts the time column into a datetime format.
4. Sorts observations chronologically.
5. Creates features from the date and time.
6. Removes rows containing missing values introduced by lag and target creation.

### Time-based features

The following features represent temporal patterns:

- Hour of the day
- Day of the month
- Month
- Day of the year

These features help the model learn recurring daily and seasonal temperature patterns.

### Lag features

Lag features provide the model with information about previous weather conditions.

For example, `temp_lag_24` represents the temperature 24 hours earlier, while `temp_lag_168` represents the temperature 168 hours earlier.

The notebook uses previous temperature measurements and historical values of humidity, precipitation, pressure, and wind speed.

These features help the models learn relationships between past weather conditions and future temperature.

### Target variable

For the initial forecasting experiment, the target is the temperature 24 hours after the observation time.

For the seven-day forecasting experiment, separate target variables are created for forecast horizons ranging from 8 hours to 168 hours ahead, in eight-hour intervals.

## 5. Training and Testing Strategy

The notebook uses a chronological split rather than randomly shuffling the observations.

- **Training period:** Observations before 1 January 2024.
- **Testing period:** Observations from 2024.

This approach tests the models on a later time period than the one used for training, which is more appropriate for evaluating time-dependent forecasting tasks.

The models learn patterns from the training data and then generate predictions for the test data. Their predictions are compared with the actual observed temperatures.

## 6. Machine Learning Models

### A. Random Forest Regressor

The first model is a `RandomForestRegressor` from Scikit-learn.

Random Forest combines the predictions of multiple decision trees to produce a regression estimate. Each tree learns relationships between the input features and the target temperature, and their predictions are combined.

**Initial configuration:**
- Number of trees: 100
- Random state: 42
- Parallel processing: Enabled using `n_jobs=-1`

This model serves as the initial machine learning benchmark for the project.

### B. Improved Random Forest Regressor

The second experiment uses another Random Forest model with an expanded feature set.

Additional lag features include temperature measurements from 48, 72, and 168 hours earlier, along with additional historical humidity and pressure measurements.

The number of trees is increased from 100 to 200.

This experiment investigates whether providing more historical information and increasing the number of trees improves predictive performance.

### C. XGBoost Regressor

The project also uses `XGBRegressor` from the XGBoost library.

XGBoost builds decision trees sequentially, with each new stage helping to correct errors made by the existing ensemble.

**Configuration used in the initial experiment:**
- Number of estimators: 500
- Learning rate: 0.05
- Maximum tree depth: 6
- Subsample: 0.8
- Column subsample: 0.8
- Random state: 42

The XGBoost model is evaluated against both Random Forest experiments.

The notebook also examines XGBoost feature importance to identify which input variables contribute most to the model's predictions.

### D. LSTM Neural Network

The project includes an LSTM (Long Short-Term Memory) neural network for time-series forecasting.

Unlike the tree-based models, an LSTM is designed to learn sequential patterns in data.

The notebook prepares sequences containing the previous 24 hours of five weather variables:
- Temperature
- Relative humidity
- Precipitation
- Pressure
- Wind speed

The network contains:
- An LSTM layer with 64 units
- A Dropout layer with a rate of 0.2
- A Dense layer with 32 units and ReLU activation
- A final Dense layer that predicts temperature

The model uses the Adam optimizer and mean squared error as its loss function. Predictions are converted back to the original temperature scale before evaluation.

## 7. Model Evaluation

The project evaluates its regression models using two metrics.

**Mean Absolute Error (MAE)**

MAE measures the average absolute difference between predicted and actual temperatures. A lower MAE indicates smaller average prediction errors.

**Root Mean Squared Error (RMSE)**

RMSE measures prediction errors while penalizing larger errors more strongly. A lower RMSE generally indicates better predictive performance.

Both metrics are reported in the same temperature units as the target, degrees Celsius.

## 8. Results

The following values are recorded in the notebook's model-comparison table for the 24-hour-ahead forecasting experiment.

| Model | MAE (°C) | RMSE (°C) |
|---|---:|---:|
| Random Forest Regressor | 1.128 | 1.510 |
| Improved Random Forest Regressor | 1.116 | 1.492 |
| XGBoost Regressor | 1.109 | 1.477 |
| LSTM Neural Network | 4.859 | 5.584 |

Among these recorded experiments, XGBoost achieves the lowest MAE and RMSE.

The improved Random Forest also performs slightly better than the initial Random Forest, suggesting that the expanded lag-feature set and revised model configuration helped in this experiment.

The LSTM has higher recorded errors in this comparison. This result illustrates that a neural network is not automatically more accurate than a tree-based model; its performance also depends on data preparation, training, architecture, and evaluation methodology.

The notebook also evaluates a simple baseline that uses the temperature from 24 hours earlier as the prediction. Its recorded results are:

- Baseline MAE: approximately 1.464 °C
- Baseline RMSE: approximately 1.973 °C

The notebook reports an approximately 24.30% reduction in MAE for XGBoost relative to this baseline.

*Note: These results belong to the notebook's recorded experiments. They should be rechecked by running the notebook from start to finish before being treated as final, directly comparable benchmark results.*

## 9. Visualization and Interpretation

The notebook includes visualizations to help interpret model behavior.

- **Actual versus predicted temperature:** Compares model predictions with observed temperatures over a portion of the test period.
- **Feature importance:** Uses XGBoost's feature-importance scores to rank input variables.
- **Seven-day forecast plot:** Displays predicted temperatures at eight-hour intervals across the forecast horizon.

These visualizations help assess prediction quality and understand the contribution of historical weather information.

## 10. Seven-Day Temperature Forecasting

The notebook extends the initial 24-hour prediction experiment to a seven-day forecast.

It creates 21 forecast horizons:

`8, 16, 24, 32, ... , 160, 168 hours ahead`

A separate XGBoost model is trained for each forecast horizon. The models use the same general set of 20 time-based and lagged weather features.

The predictions are combined into a forecast table containing:
- Forecast horizon in hours
- Predicted date and time
- Predicted temperature in degrees Celsius

The notebook also visualizes the resulting forecast.

This approach allows the project to predict temperature at multiple points over the next seven days rather than relying on only one 24-hour-ahead prediction.

## 11. Tools and Technologies

- **Python** — Main programming language
- **Pandas** — Data manipulation and preprocessing
- **NumPy** — Numerical operations
- **Requests** — Retrieving weather data from the API
- **Matplotlib** — Data visualization
- **Scikit-learn** — Random Forest regression, preprocessing, and evaluation metrics
- **XGBoost** — Gradient-boosted tree regression
- **TensorFlow / Keras** — LSTM neural network
- **Joblib** — Saving trained models and feature lists
- **Google Colab / Jupyter Notebook** — Development and execution environment

## 12. How to Run the Project

1. Download or clone the GitHub repository.
2. Open the notebook in Google Colab or Jupyter Notebook.
3. Install the required Python libraries if they are not already available.
4. Provide the weather dataset when prompted by the notebook, or run the API data-collection cells.
5. Execute the preprocessing and feature-engineering cells.
6. Train the models and evaluate their predictions.
7. Run the forecasting cells to generate and visualize future temperature predictions.

For reproducible results, execute the notebook cells in the intended order and ensure that the required input data and dependencies are available.

## 13. Future Improvements

Potential next steps include:

- Re-evaluating every forecast horizon separately.
- Comparing model performance across different seasons.
- Tuning model hyperparameters systematically.
- Checking for data leakage and ensuring that all features would be available at the actual forecast time.
- Comparing against stronger seasonal and persistence baselines.
- Improving LSTM training and evaluation.
- Testing predictions on additional unseen periods.
- Adding prediction intervals to represent forecast uncertainty.
- Saving a reproducible pipeline for preprocessing, training, and inference.

## Author
Yasmini Dharne(ms24262),Aditya Harsh(ms24264),Krish Gautam(ms24228)
