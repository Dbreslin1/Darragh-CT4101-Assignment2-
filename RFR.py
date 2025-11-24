import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
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

    for col in X.select_dtypes(include=['float64', 'int64']).columns:
        lower = X[col].quantile(0.01)
        upper = X[col].quantile(0.99)
        X[col] = X[col].clip(lower, upper)
   

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_medians = X_train.median()   # compute medians on train
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians) 

 
    # Tune n_estimators and max_features
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_features': ['sqrt', 'log2', 0.5]
    }

    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    print('Starting GridSearchCV ')
    grid_search.fit(X_train, y_train)

    print('\nBest params:')
    print(grid_search.best_params_)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('\nEvaluation on test set:')
    print(f'Mean squared error: {mse:.4f}')
    print(f'Mean absolute error: {mae:.4f}')
    print(f'R2: {r2:.4f}')
    # Convert cv_results_ to DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)

    # Save to CSV
    out_csv = os.path.join(base_dir, 'rfr_grid_search_results.csv')
    results_df.to_csv(out_csv, index=False)

    print(f"Grid search results saved to: {out_csv}")


    

if __name__ == '__main__':
    main()
