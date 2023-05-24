from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

with open('NB_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('count_vectorizer.pkl', 'rb') as file:
    cv = pickle.load(file)

temp={
    0:"ham",
    1:"spam"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/find', methods=['POST'])
def find():
    data = request.get_json()
    txt = [data.get('txt')]
    vect = cv.transform(txt).toarray()
    my_prediction = model.predict(vect)
    print(my_prediction)
    return temp[my_prediction[0]]
    
if __name__ == '__main__':
    app.run(debug = True)
