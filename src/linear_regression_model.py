import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

def main():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'linear_regression_data.csv')
    data_path = os.path.abspath(data_path)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    X = df[['SquareFeet']]
    y = df['Price']
    
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X, y)
    
    predictions = model.predict(X)
    
    print("\n--- Model Metrics ---")
    print(f"Coefficient (Slope): {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y, predictions):.2f}")
    print(f"R-squared (R2): {r2_score(y, predictions):.4f}")
    
    new_square_feet = 2250
    predicted_price = model.predict([[new_square_feet]])
    print(f"\nPredicted price for {new_square_feet} sq ft: ${predicted_price[0]:,.2f}")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', label='Actual Data')
    plt.plot(X, predictions, color='red', linewidth=2, label='Regression Line')
    plt.xlabel('Square Feet')
    plt.ylabel('Price ($)')
    plt.title('Linear Regression: House Price vs Square Feet')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(os.path.dirname(__file__), '..', 'linear_regression_plot.png')
    plt.savefig(plot_path)
    print(f"\nSaved regression plot to {os.path.abspath(plot_path)}")
    plt.show()

if __name__ == "__main__":
    main()
