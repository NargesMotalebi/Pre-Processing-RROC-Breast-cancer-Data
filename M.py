
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load and preprocess your dataset
X, y = ...  # Load your dataset

# Split the data
X_train, X_calibration, y_train, y_calibration = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the calibration set
y_pred = model.predict(X_calibration)

# Calculate nonconformity scores
# A simple nonconformity score could be the absolute error
nonconformity_scores = np.abs(y_calibration - y_pred)

# Construct prediction intervals
# Sort the nonconformity scores and find the quantile corresponding to the desired coverage level
sorted_scores = np.sort(nonconformity_scores)
alpha = 0.1  # Desired coverage level
quantile = np.quantile(sorted_scores, alpha)

# Use the quantile to construct prediction intervals for new data
X_test, y_test = ...  # Load your test data
y_test_pred = model.predict(X_test)
lower_bound = y_test_pred - quantile
upper_bound = y_test_pred + quantile

# Validate the intervals
# Check that the true outcomes are within the prediction intervals with the desired probability
coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound))
print(f"Coverage: {coverage}")

#For each uncensored observation in the calibration set, the nonconformity score might be the absolute difference between observed and predicted times.
#For each censored observation, the nonconformity score could be the survival probability at the censoring time.
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

# Example survival data with censored observations
data = {
    'time': [5, 6, 6, 2, 4, 7, 10, 10],
    'event': [1, 0, 1, 1, 1, 0, 0, 1]  # 1: event occurred, 0: censored
}

# Create a DataFrame
df = pd.DataFrame(data)

# Split the data into training and calibration sets
train_df, cal_df = train_test_split(df, test_size=0.5, random_state=42)

# Fit the Kaplan-Meier estimator on the training set
kmf = KaplanMeierFitter()
kmf.fit(durations=train_df['time'], event_observed=train_df['event'])

# Function to predict survival time for a new data point
def predict_survival_time(kmf):
    # Example of a simple prediction: median survival time
    median_survival_time = kmf.median_survival_time_
    return median_survival_time

# Compute nonconformity scores for the calibration set
cal_nonconformity_scores = []
for time, event in zip(cal_df['time'], cal_df['event']):
    predicted_time = predict_survival_time(kmf)
    if event == 1:  # uncensored observation
        nonconformity_score = abs(time - predicted_time)
    else:  # censored observation
        survival_prob = kmf.survival_function_at_times(time).values[0]
        nonconformity_score = survival_prob
    cal_nonconformity_scores.append(nonconformity_score)

# Define a function to compute prediction intervals
def conformal_prediction_interval(kmf, alpha=0.1):
    # Sort the calibration nonconformity scores
    sorted_nonconformity_scores = np.sort(cal_nonconformity_scores)
    # Determine the quantile for the prediction interval
    quantile = np.ceil((1 - alpha) * len(sorted_nonconformity_scores)) / len(sorted_nonconformity_scores)
    #quantile = np.quantile(sorted_scores, alpha)
    threshold = sorted_nonconformity_scores[int(quantile)]
    return threshold

# Compute the threshold for the prediction interval
alpha = 0.1   # Desired coverage level
threshold = conformal_prediction_interval(kmf, alpha)

# Function to predict and generate prediction intervals for new data
def predict_and_generate_intervals(time, kmf, threshold):
    predicted_time = predict_survival_time(kmf)
    lower_bound = max(0, predicted_time - threshold)
    upper_bound = predicted_time + threshold
    return lower_bound, upper_bound

# Apply conformal prediction to new observations
new_times = [3, 8, 12]  # Example new times for prediction
for time in new_times:
    lower_bound, upper_bound = predict_and_generate_intervals(time, kmf, threshold)
    print(f"Time: {time}, Prediction Interval: [{lower_bound:.2f}, {upper_bound:.2f}]")




# Replace null values in BiopsyDate with SurgeryDate if the treatment protocol is adjuvant
# adjuvant_mask = df_GData['TreatmentProtocol'] == 'adjuvant'
# # df_GData.loc[adjuvant_mask, 'Gregorian_BiopsiDate'] = df_GData.loc[adjuvant_mask, 'Gregorian_BiopsiDate'].fillna(df_GData.loc[adjuvant_mask, 'Gregorian_SurgeryDate'])


# def partial_likelihood_cox(X, T, E, beta):
#     """
#     Calculate the partial likelihood of Cox regression.
    
#     Parameters:
#     X : np.ndarray
#         The covariates (predictor variables), shape (n_samples, n_features).
#     T : np.ndarray
#         The times of events or censoring, shape (n_samples,).
#     E : np.ndarray
#         The event indicator (1 if event occurred, 0 if censored), shape (n_samples,).
#     beta : np.ndarray
#         The coefficients for the covariates, shape (n_features,).
    
#     Returns:
#     float
#         The partial likelihood value.
#     """
#     n_samples, n_features = X.shape

#     # Calculate the linear predictor (X * beta)
#     linear_predictor = np.dot(X, beta)

#     # Initialize the partial likelihood
#     partial_lik = 0.0

#     # Iterate over each event time
#     for i in range(n_samples):
#         if E[i] == 1:  # Only consider observed events
#             # The risk set is all samples with T >= T[i]
#             risk_set = np.where(T >= T[i])[0]
            
#             # Calculate the log-partial likelihood for this event time
#             numerator = linear_predictor[i]
#             denominator = np.log(np.sum(np.exp(linear_predictor[risk_set])))
#             partial_lik += numerator - denominator
    
#     return partial_lik
# #The coefficients are estimated by maximizing the log-partial likelihood function
# # Example usage
# # Covariates (predictor variables)
# X = np.array([[1, 2],
#               [3, 4],
#               [5, 6]])

# # Times of events or censoring
# T = np.array([5, 10, 15])

# # Event indicator (1 if event occurred, 0 if censored)
# E = np.array([1, 1, 0])

# # Coefficients for the covariates
# beta = np.array([0.1, 0.2])

# # Calculate the partial likelihood
# partial_lik_value = partial_likelihood_cox(X, T, E, beta)
# print(f'Partial Likelihood: {partial_lik_value}')
