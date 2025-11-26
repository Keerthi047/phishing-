from flask import Flask, render_template , request
import pickle 
import numpy as np
import shap
import os
import re

app = Flask(__name__)

vector = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("phishing.pkl", 'rb'))


@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url_input = request.form['url']
    
    # Transform input
    features = vector.transform([url_input])
    
    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    confidence = np.max(probability) * 100

    # --- SHAP EXPLANATION ---
    background_data = model.predict_proba(vector.transform(["example"]))
    explainer = shap.LinearExplainer(model, vector.transform(["http"]))  # dummy to initialize
    shap_values = explainer.shap_values(features)
    shap.initjs()
    
    # Generate SHAP force plot for this input
    shap_html = shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        features=vector.get_feature_names_out(),
        matplotlib=False
    )
    shap_plot_path = os.path.join("static", "shap_force_plot.html")


    shap.save_html(shap_plot_path, shap_html)

    #result = "Phishing Site ⚠️" if prediction == "bad" else "Legitimate Site ✅"
    # Correct label mapping
    prediction = model.predict(features)[0]

    if prediction == "bad":  # Phishing is class 0
        result = "Phishing Site ⚠️"
    else:
        result = "Legitimate Site ✅"


    return render_template(
        'result.html',
        url=url_input,
        result=result,
        confidence=round(confidence, 2),
        shap_plot_path=shap_plot_path
    )



if __name__=="__main__":
    app.run(debug=True)