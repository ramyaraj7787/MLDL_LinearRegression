from flask import Flask, render_template, request, url_for, session, flash


app = Flask(__name__)


@app.route('/')  # To render Homepage
def home_page():
    return render_template('index.html')


@app.route('/LinearRegression')  # To render Linear Regression Page
def linreg():
    return render_template('linreg.html')


@app.route('/LogisticRegression')  # To render Logistic Regression Page
def logreg():
    return render_template('logreg.html')


@app.route('/DecisionTree')  # To render Decision Tree Page
def dtree():
    return render_template('dectree.html')


@app.route('/RandomForest')  # To render Random Forest Page
def randfrst():
    return render_template('randfrst.html')


@app.route('/XGBoost')  # To render XGBoost Page
def xgboost():
    return render_template('xgboost.html')


@app.route('/TimeSeries')  # To render Time Series Page
def timeseries():
    return render_template('timeseries.html')



if __name__ == '__main__':
    app.run(debug=True)