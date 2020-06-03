import pandas as pd
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
model = pickle.load(open('rfc.pkl', 'rb'))
lenc = LabelEncoder()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/readmission',methods=['POST'])
def readmission():
    '''
    For rendering results on HTML GUI
    '''
    new_values = pd.read_csv(request.form['Path to .csv'])
    new_values = new_values.iloc[:, :].values
    for i in range(28):
        new_values[:,i] = lenc.fit_transform(new_values[:,i])
    prediction = model.predict(new_values)
    if prediction > 0.65:
        output = 'YES'
    else:
        output = 'NO'
    
    return render_template('index.html', final_class='Readmission: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
