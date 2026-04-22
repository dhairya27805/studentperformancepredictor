
from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

app = Flask(__name__)

df = pd.read_csv("student-mat.csv", sep=';')

X = df.drop('G3', axis=1)
y = df['G3']

original_cols = X.columns.tolist()

# Detect categorical columns
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(exclude='object').columns.tolist()

# Unique values for dropdowns
options = {col: sorted(df[col].dropna().unique()) for col in cat_cols}

# Encode
X_encoded = pd.get_dummies(X, drop_first=True)

model = Ridge(alpha=201.85086292982749)
model.fit(X_encoded, y)

model_cols = X_encoded.columns

HTML = '''
<h2>Student Performance Predictor (G3)</h2>
<form method="post">

<h3>Categorical Inputs</h3>
{% for col in cat_cols %}
    {{col}}:
    <select name="{{col}}">
        {% for val in options[col] %}
            <option value="{{val}}">{{val}}</option>
        {% endfor %}
    </select><br>
{% endfor %}

<h3>Numeric Inputs</h3>
{% for col in num_cols %}
    {{col}}:
    <select name="{{col}}">
        {% for val in range(0, 21) %}
            <option value="{{val}}">{{val}}</option>
        {% endfor %}
    </select><br>
{% endfor %}

<input type="submit">
</form>

{% if prediction %}
<h3>Predicted G3: {{prediction}}</h3>
{% endif %}
'''

@app.route("/", methods=["GET","POST"])
def home():
    if request.method == "POST":
        user_input = {}

        for col in original_cols:
            val = request.form.get(col)
            if col in num_cols:
                user_input[col] = float(val)
            else:
                user_input[col] = val

        df_input = pd.DataFrame([user_input])

        df_input = pd.get_dummies(df_input)

        for col in model_cols:
            if col not in df_input:
                df_input[col] = 0

        df_input = df_input[model_cols]

        pred = model.predict(df_input)[0]

        return render_template_string(HTML, 
                                      cat_cols=cat_cols, 
                                      num_cols=num_cols,
                                      options=options,
                                      prediction=round(pred,2))

    return render_template_string(HTML, 
                                  cat_cols=cat_cols, 
                                  num_cols=num_cols,
                                  options=options)

if __name__ == "__main__":
    app.run(debug=True)
