from flask import Flask, jsonify, request, render_template
import trainer

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data["message"]
    response = trainer.brain(msg)
    return jsonify({'response': str(response)})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)  # starts web server
