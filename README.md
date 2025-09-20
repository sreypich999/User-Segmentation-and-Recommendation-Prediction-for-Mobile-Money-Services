# 📊 Focused on Clustering + Prediction: Mobile Money User Segmentation & Recommendation Predictor

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-lightgrey.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 🚀 **An advanced machine learning project combining unsupervised clustering and supervised prediction to analyze mobile money user behavior and predict recommendation likelihood.**

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Features](#-features)
- [🛠️ Installation](#️-installation)
- [📖 Usage](#-usage)
- [🔬 Methodology](#-methodology)
- [📈 Results & Insights](#-results--insights)
- [🧠 Technologies Used](#-technologies-used)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🎯 Overview

This project leverages **machine learning techniques** to segment mobile money users and predict their likelihood of recommending the service to others. By combining **K-Means clustering** for user segmentation and **Random Forest classification** for recommendation prediction, the project provides actionable insights into user behavior patterns and factors influencing service adoption.

The analysis is based on a comprehensive dataset covering user demographics, usage patterns, trust levels, security awareness, and motivational factors. The project includes both a detailed Jupyter notebook for analysis and an interactive web application for real-time predictions.

### 🎯 Key Objectives

- **User Segmentation**: Identify distinct user groups based on behavioral patterns
- **Recommendation Prediction**: Predict user recommendation likelihood (A-D grading system)
- **Feature Importance Analysis**: Understand which factors most influence recommendations
- **Interactive Web App**: Provide user-friendly interface for predictions and insights

## ✨ Features

### 🔍 Data Analysis & Visualization
- **Comprehensive EDA**: Exploratory data analysis with statistical summaries and correlations
- **Interactive Visualizations**: PCA plots, cluster heatmaps, and feature importance charts
- **Variance Analysis**: Identify and handle low-variance features

### 🤖 Machine Learning Pipeline
- **K-Means Clustering**: Automated elbow method for optimal cluster selection
- **Dimensionality Reduction**: PCA for 2D/3D visualization of user segments
- **Supervised Prediction**: Logistic Regression and Random Forest models
- **SMOTE Oversampling**: Handle class imbalance in recommendation data

### 🌐 Interactive Web Application
- **Modern UI**: Responsive design with glassmorphism effects and animations
- **Real-time Predictions**: Instant recommendation predictions with explanations
- **Personalized Insights**: User-centric feature impact analysis and suggestions
- **QR Code Generation**: Mobile-friendly access via ngrok tunneling
- **Feature Importance Plots**: Visual representation of model insights

### 📊 User Segmentation Categories
- **🚀 Power Users**: High usage, strong trust, frequent money sending
- **🛡️ Trusted & Secure Users**: High trust, security-conscious, few issues
- **⚠️ Cautious & Problematic Users**: Low usage with experienced issues
- **🌟 Enthusiastic Recommenders**: High recommendation likelihood
- **⚖️ Moderate Users**: Balanced usage patterns

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook
- pip package manager

### Step-by-Step Setup


2. **Install required packages**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn plotly flask pyngrok qrcode[pil]
   ```

3. **Download the dataset**:
   - Place `Mobile_Money.xlsx` in the same directory as the notebook
   - Ensure the dataset contains the required columns for analysis

4. **For the web application**:
   ```bash
   # Additional installations
   pip install flask-ngrok
   ```

5. **Set up ngrok** (for web app deployment):
   - Sign up at [ngrok.com](https://ngrok.com)
   - Get your authtoken
   - Replace the authtoken in the notebook code

## 📖 Usage

### Running the Jupyter Notebook

1. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```

2. **Open the notebook**:
   - Navigate to `Focused_on_Clustering_+_Prediction.ipynb`
   - Run cells sequentially from top to bottom

3. **Key Sections**:
   - **Data Loading & Preprocessing**: Load and clean the mobile money dataset
   - **Feature Engineering**: Transform categorical variables to numerical
   - **Clustering Analysis**: Perform K-Means clustering and visualize segments
   - **Prediction Modeling**: Train and evaluate recommendation prediction models
   - **Web App Deployment**: Launch the interactive Flask application

### Using the Web Application

1. **Run the web app section** of the notebook
2. **Access the application** via the generated ngrok URL
3. **Input user characteristics** through the intuitive form
4. **Receive instant predictions** with personalized insights and suggestions

### Example Usage

```python
# Load and preprocess data
df = pd.read_excel("Mobile_Money.xlsx")

# Perform clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Make predictions
prediction = rf_model.predict(user_input)
```

## 🔬 Methodology

### 1. Data Preprocessing
- **Categorical Encoding**: Map ordinal variables (frequency, amounts, trust levels)
- **Feature Selection**: Include usage patterns, motivations, and demographic factors
- **Missing Value Handling**: Fill NaNs with appropriate defaults
- **Standardization**: Scale features for clustering and modeling

### 2. Clustering Approach
- **Algorithm**: K-Means with Euclidean distance
- **Optimal K Selection**: Elbow method and silhouette analysis
- **Dimensionality Reduction**: PCA for visualization
- **Cluster Interpretation**: Analyze centroids and feature distributions

### 3. Prediction Modeling
- **Algorithms**: Logistic Regression (multinomial) and Random Forest
- **Class Balancing**: SMOTE oversampling for minority classes
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score
- **Feature Importance**: Permutation importance and Gini importance

### 4. Web Application Architecture
- **Backend**: Flask with RESTful API endpoints
- **Frontend**: HTML5 with CSS3 animations and JavaScript
- **Deployment**: ngrok for public tunneling
- **Visualization**: Matplotlib and Plotly for charts

## 📈 Results & Insights

### Clustering Results
- **Optimal Clusters**: 2 distinct user segments identified
- **Segment Characteristics**:
  - Cluster 0: Moderate users with balanced features
  - Cluster 1: High-engagement users with strong positive indicators

### Prediction Performance
- **Random Forest Accuracy**: 91% on test set
- **Key Predictive Features**:
  1. Usage Frequency (most important)
  2. Trust Level
  3. Weekly Spending Amount
  4. Security Awareness
  5. Experience with Issues

### Business Insights
- **Recommendation Drivers**: High usage frequency and trust are strongest predictors
- **Risk Factors**: Experienced issues significantly reduce recommendation likelihood
- **Opportunity Areas**: Security-conscious users show high potential for positive recommendations
- **Target Segments**: Power users and trusted users are most valuable for marketing

## 🧠 Technologies Used

### Core Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **matplotlib** & **seaborn** - Data visualization
- **plotly** - Interactive charts

### Specialized Tools
- **imbalanced-learn** - SMOTE for class balancing
- **Flask** - Web application framework
- **pyngrok** - Public tunneling for web app
- **qrcode** - QR code generation

### Development Environment
- **Jupyter Notebook** - Interactive development
- **Python 3.8+** - Programming language
- **Git** - Version control





## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**📧 Contact**: For questions or collaborations, please reach out via GitHub issues.

**⭐ Star this repo** if you find it helpful for your mobile money analysis projects!

---

*Built with ❤️ for advancing financial inclusion through data-driven insights.*
