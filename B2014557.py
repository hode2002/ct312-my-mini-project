from flask import Flask, request, jsonify
from flask_cors import CORS

from B2014557_train import model

app = Flask(__name__)
cors = CORS(app, resources={
    r"/api/*": {"origins": "*"}
})

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    sepal_length = data['sepal_length']
    sepal_width = data['sepal_width']
    petal_length = data['petal_length']
    petal_width = data['petal_width']
    
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    species = prediction[0]
    
    return jsonify({'species': species})

@app.route('/', methods=['GET'])

def hello():
    return jsonify({'message': 'Hello World'})

if __name__ == "__main__":
    app.run(debug=False)
