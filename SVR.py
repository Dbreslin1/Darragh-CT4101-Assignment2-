import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


def main():
	base_dir = os.path.dirname(__file__)
	csv_path = os.path.join(base_dir, 'steel .csv')

	df = pd.read_csv(csv_path)

	X = df.drop(columns=['tensile_strength'])
	y = df['tensile_strength']

	
	# PREPROCESSING 
	df = pd.concat([X, y], axis=1)
	df = df.dropna(subset=['tensile_strength'])
	X = df.drop(columns=['tensile_strength'])
	y = df['tensile_strength']

	# clip numeric columns at 1st/99th percentiles
	for col in X.select_dtypes(include=['float64', 'int64']).columns:
		lower = X[col].quantile(0.01)
		upper = X[col].quantile(0.99)
		X[col] = X[col].clip(lower, upper)

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42
	)

	# compute medians on training data 
	train_medians = X_train.median()
	X_train = X_train.fillna(train_medians)
	X_test = X_test.fillna(train_medians)

	# Fit scaler on training data only
	scaler = StandardScaler().fit(X_train)
	X_train_s = scaler.transform(X_train)
	X_test_s = scaler.transform(X_test)

	# my 2 hyperparameters to tune 
	param_grid = {
		'C': [1, 10, 100],
		'gamma': ['scale', 'auto', 0.01, 0.001]
	}

	grid_search = GridSearchCV(
		SVR(kernel='rbf'), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1
	)

	print('Gridsearch')
	grid_search.fit(X_train_s, y_train)

	print('\nBest params:')
	print(grid_search.best_params_)

	best_model = grid_search.best_estimator_

	y_pred = best_model.predict(X_test_s)

	mse = mean_squared_error(y_test, y_pred)
	mae = mean_absolute_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)

	print('\nEvaluation on test set:')
	print(f'Mean squared error: {mse:.4f}')
	print(f'Mean absolute error: {mae:.4f}')
	print(f'R2: {r2:.4f}')

	# Save model and scaler separately
	out_model = os.path.join(base_dir, 'svr_model.joblib')
	out_scaler = os.path.join(base_dir, 'scaler.joblib')
	joblib.dump(best_model, out_model)
	joblib.dump(scaler, out_scaler)
	print(f'Model saved to: {out_model}')
	print(f'Scaler saved to: {out_scaler}')


if __name__ == '__main__':
	main()

