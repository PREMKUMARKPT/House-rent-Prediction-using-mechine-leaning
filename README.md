# House-rent-Prediction-using-mechine-leaning

# **Housing Price Prediction**

## **Project Overview**

This project aims to predict housing prices based on various factors such as location, number of rooms, square footage, and other relevant features. The model uses machine learning techniques to analyze historical housing data and estimate prices for new listings.

## **Dataset**

- The dataset contains features like:
  - Location
  - Number of bedrooms
  - Number of bathrooms
  - Square footage
  - Year built
  - Property type
  - Price
- Data preprocessing steps include handling missing values, feature scaling, and encoding categorical variables.

## **Technologies Used**

- Python
- Jupyter Notebook
- Pandas, NumPy (Data Processing)
- Matplotlib, Seaborn (Data Visualization)
- Scikit-learn (Machine Learning)

## **Installation & Setup**

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/housing-price-prediction.git
   cd housing-price-prediction
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Open the Jupyter Notebook:
   ```sh
   jupyter notebook housing_price_prediction.ipynb
   ```

## **Project Steps**

### **1️⃣ Data Preprocessing**

- Load dataset:
  ```python
  import pandas as pd
  df = pd.read_csv('housing_data.csv')
  ```
- Check missing values:
  ```python
  df.isnull().sum()
  ```
- Handle missing values:
  ```python
  df.fillna(df.median(), inplace=True)
  ```
- Convert categorical data:
  ```python
  from sklearn.preprocessing import LabelEncoder
  encoder = LabelEncoder()
  df['location'] = encoder.fit_transform(df['location'])
  ```
- Feature Scaling:
  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  df[['sqft', 'price']] = scaler.fit_transform(df[['sqft', 'price']])
  ```

### **2️⃣ Exploratory Data Analysis (EDA)**

- Visualizing distributions:
  ```python
  import seaborn as sns
  import matplotlib.pyplot as plt
  sns.histplot(df['price'], kde=True)
  plt.show()
  ```
- Correlation heatmap:
  ```python
  sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
  plt.show()
  ```

### **3️⃣ Model Training & Evaluation**

- Split dataset:
  ```python
  from sklearn.model_selection import train_test_split
  X = df.drop(columns=['price'])
  y = df['price']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  ```
- Train Linear Regression:
  ```python
  from sklearn.linear_model import LinearRegression
  model = LinearRegression()
  model.fit(X_train, y_train)
  ```
- Train Random Forest Regressor:
  ```python
  from sklearn.ensemble import RandomForestRegressor
  rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
  rf_model.fit(X_train, y_train)
  ```
- Model Evaluation:
  ```python
  from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
  y_pred = rf_model.predict(X_test)
  print("MAE:", mean_absolute_error(y_test, y_pred))
  print("MSE:", mean_squared_error(y_test, y_pred))
  print("R² Score:", r2_score(y_test, y_pred))
  ```

### **4️⃣ Hyperparameter Tuning**

- Use GridSearchCV:
  ```python
  from sklearn.model_selection import GridSearchCV
  param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
  grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
  grid_search.fit(X_train, y_train)
  print("Best Parameters:", grid_search.best_params_)
  ```

### **5️⃣ Model Deployment**

- Save trained model:
  ```python
  import joblib
  joblib.dump(rf_model, 'housing_price_model.pkl')
  ```
- Deploy using Flask:
  ```python
  from flask import Flask, request, jsonify
  import joblib
  import numpy as np

  app = Flask(__name__)
  model = joblib.load('housing_price_model.pkl')

  @app.route('/predict', methods=['POST'])
  def predict():
      data = request.get_json()
      features = np.array(data['features']).reshape(1, -1)
      prediction = model.predict(features)
      return jsonify({'predicted_price': prediction.tolist()})

  if __name__ == '__main__':
      app.run(debug=True)
  ```

## **Future Improvements**

- Adding more features like crime rates, school ratings, etc.
- Implementing deep learning models (ANNs).
- Deploying on cloud platforms (AWS, GCP).

## **Contributing**

Feel free to fork this repository, make modifications, and submit pull requests.

## **License**

This project is open-source and available under the MIT License.


