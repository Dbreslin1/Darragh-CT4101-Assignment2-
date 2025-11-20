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

    # same processing as svr
    df = pd.concat([X, y], axis=1)
    df = df.dropna(subset=['tensile_strength'])
    X = df.drop(columns=['tensile_strength'])
    y = df['tensile_strength']
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

 
    # Tune n_estimators and max_features
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_features': ['auto', 'sqrt', 0.5]
    }

    gs = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    print('Starting GridSearchCV ')
    gs.fit(X_train, y_train)

    print('\nBest params:')
    print(gs.best_params_)

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('\nEvaluation on test set:')
    print(f'MSE: {mse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R2: {r2:.4f}')

    # Save trained model
    out_model = os.path.join(base_dir, 'rfr_model.joblib')
    joblib.dump(best_model, out_model)
    print(f'Model saved to: {out_model}')


if __name__ == '__main__':
    main()
