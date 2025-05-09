from flask import Flask, jsonify, request, render_template
import trainer
import os

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
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

