Indian Auto Price Predictor
A full-stack machine learning application to accurately predict the resale value of used cars in the Indian market.

Live Application Demo
Project Overview
This project is an end-to-end implementation of a machine learning pipeline for predicting used car prices. It starts with raw data, proceeds through data cleaning and feature engineering, trains a predictive model, and culminates in a user-friendly web application where users can get real-time price estimates. The model is specifically tailored to the nuances of the Indian automobile market.

Key Features
Data Processing Pipeline: Engineered a robust data pipeline using Pandas to clean and transform raw data, which included creating a key Car_Age feature.

High-Accuracy Model: Developed and trained a Random Forest Regressor model using Scikit-learn, achieving a 96.8% R-squared score on test data.

Feature Importance Analysis: Identified Present_Price (current showroom price) as the most significant price predictor.

Interactive Web Interface: Deployed the trained model using a Flask server and a responsive HTML/Tailwind CSS frontend, allowing for real-time predictions.

Tech Stack
Backend: Python, Flask

ML & Data Processing: Pandas, Scikit-learn, Joblib

Frontend: HTML, Tailwind CSS

Environment: VS Code, Git & GitHub

Project Workflow
Data Collection: Used a well-structured dataset of Indian used car sales from Kaggle.

Data Exploration: Analyzed the data to understand its structure, features, and identify any missing values.

Feature Engineering: Cleaned the data and created the Car_Age feature from the Year column to improve model performance.

Model Training: Trained a Random Forest Regressor on the prepared data, which proved to be highly effective.

Model Evaluation: Achieved an R-squared score of 0.9680, indicating the model explains 96.8% of the price variance. The Mean Absolute Error was approximately ₹64,000.

Deployment: Built a Flask application to serve the trained model through a simple and intuitive web interface.

Installation and Usage
To run this project on your local machine, please follow these steps:

1. Clone the Repository
(Assuming your GitHub repository name is Indian-Auto-Price-Predictor)

git clone [https://github.com/YOUR_USERNAME/Indian-Auto-Price-Predictor.git](https://github.com/YOUR_USERNAME/Indian-Auto-Price-Predictor.git)
cd Indian-Auto-Price-Predictor

2. Create and Activate a Virtual Environment

# Create the environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\Activate

# Activate on macOS/Linux
source venv/bin/activate

3. Install Dependencies
All required libraries are listed in requirements.txt.

pip install -r requirements.txt

4. Run the Flask Application

python app.py

5. Access the Application
Open your web browser and navigate to http://127.0.0.1:5000.

Acknowledgments
The dataset used for this project was sourced from the Vehicle Dataset on Kaggle.