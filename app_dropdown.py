from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.linear_model import Ridge

app = Flask(__name__)

# Load data
df = pd.read_csv("student-mat.csv", sep=';')

X = df.drop('G3', axis=1)
y = df['G3']

# Encoding
X_encoded = pd.get_dummies(X, drop_first=True)

model = Ridge(alpha=201.85086292982749)
model.fit(X_encoded, y)

model_cols = X_encoded.columns

HTML = """
<h2>Student Performance Predictor (G3)</h2>
<form method="post">

<h3>Basic Info</h3>
School:
<select name="school">
<option value="GP">GP</option>
<option value="MS">MS</option>
</select><br>

Sex:
<select name="sex">
<option value="F">F</option>
<option value="M">M</option>
</select><br>

Age:
<select name="age">{% for i in range(15,23) %}<option value="{{i}}">{{i}}</option>{% endfor %}</select><br>

Address:
<select name="address">
<option value="U">U</option>
<option value="R">R</option>
</select><br>

Family Size:
<select name="famsize">
<option value="LE3">LE3</option>
<option value="GT3">GT3</option>
</select><br>

Pstatus:
<select name="Pstatus">
<option value="T">T</option>
<option value="A">A</option>
</select><br>

<h3>Education</h3>
Medu:
<select name="Medu">{% for i in range(0,5) %}<option value="{{i}}">{{i}}</option>{% endfor %}</select><br>

Fedu:
<select name="Fedu">{% for i in range(0,5) %}<option value="{{i}}">{{i}}</option>{% endfor %}</select><br>

Mjob:
<select name="Mjob">
<option value="teacher">teacher</option>
<option value="health">health</option>
<option value="services">services</option>
<option value="at_home">at_home</option>
<option value="other">other</option>
</select><br>

Fjob:
<select name="Fjob">
<option value="teacher">teacher</option>
<option value="health">health</option>
<option value="services">services</option>
<option value="at_home">at_home</option>
<option value="other">other</option>
</select><br>

Reason:
<select name="reason">
<option value="home">home</option>
<option value="reputation">reputation</option>
<option value="course">course</option>
<option value="other">other</option>
</select><br>

Guardian:
<select name="guardian">
<option value="mother">mother</option>
<option value="father">father</option>
<option value="other">other</option>
</select><br>

<h3>Study & Lifestyle</h3>
Study Time:
<select name="studytime">{% for i in range(1,5) %}<option value="{{i}}">{{i}}</option>{% endfor %}</select><br>

Failures:
<select name="failures">{% for i in range(0,5) %}<option value="{{i}}">{{i}}</option>{% endfor %}</select><br>

Absences:
<select name="absences">{% for i in range(0,94) %}<option value="{{i}}">{{i}}</option>{% endfor %}</select><br>

G1:
<select name="G1">{% for i in range(0,21) %}<option value="{{i}}">{{i}}</option>{% endfor %}</select><br>

G2:
<select name="G2">{% for i in range(0,21) %}<option value="{{i}}">{{i}}</option>{% endfor %}</select><br>

<br><input type="submit">
</form>

{% if prediction %}
<h3>Predicted G3: {{prediction}}</h3>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = dict(request.form)

        for key in ['age','Medu','Fedu','studytime','failures','absences','G1','G2']:
            user_input[key] = float(user_input[key])

        df_input = pd.DataFrame([user_input])
        df_input = pd.get_dummies(df_input)

        for col in model_cols:
            if col not in df_input:
                df_input[col] = 0

        df_input = df_input[model_cols]

        pred = model.predict(df_input)[0]

        return render_template_string(HTML, prediction=round(pred,2))

    return render_template_string(HTML)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)