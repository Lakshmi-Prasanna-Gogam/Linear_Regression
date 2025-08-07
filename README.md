# Linear_Regression

# Housing Price Prediction Using Linear Regression
Linear Regression, a supervised machine learning algorithm, to predict the furnishing status of a house based on various features like price, area, number of bedrooms, bathrooms, and amenities. The goal is to implement and understand both simple and multiple linear regression using real housing data.

# Objective
Implement and understand Simple and Multiple Linear Regression.

Preprocess real-world housing data.

Train and evaluate a regression model.

Visualize results and interpret the regression coefficients.

# Tools and Libraries
Python 3.x

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

# Data Preprocessing

Missing Value Handling:

Used dropna() and fillna(0) as needed.

Encoding Categorical Variables:

Used LabelEncoder from sklearn.preprocessing.

Encoded columns:

mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus

# Train-Test Split
Used train_test_split() with test_size=0.2 and random_state=42.

# Model Training
Model Used: LinearRegression from sklearn.linear_model

model = LinearRegression()

model.fit(x_train, y_train)

# Model Evaluation

Metrics:

MAE	

MSE

R² Score	

# Interpretation:
The R² score is very low, suggesting the model doesn’t explain the variance in the target variable well. Linear Regression may not be suitable for predicting categorical values like furnishing status.

# Model Interpretation

Positive coefficients indicate that the feature increases the likelihood of a higher furnishing status.

Negative coefficients suggest an inverse relationship.
