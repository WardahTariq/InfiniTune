from flask import Flask, jsonify
from ddpg import ddpgTune
app = Flask(__name__)

@app.route('/prediction')
def get_prediction():
    # Read the results from text files
    with open('./tmpLog/logFile', 'r') as file1:
        result1 = file1.read()
    with open('./tmpLog/w.log', 'r') as file2:
        result2 = file2.read()
    
    # Combine or process the results as needed
    # For example, concatenate the results into a single string
    prediction = f"Result 1: {result1}\nResult 2: {result2}"
    
    # Return the prediction as JSON
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
