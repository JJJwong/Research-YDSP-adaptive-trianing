from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "Server is awake!"

@app.route("/submit_score", methods=["POST"])
def submit_score():
    data = request.json  # expects JSON like {"score": 20, "targets_per_sec": 3.5}
    print("Received data:", data)
    # You can log it or save to a file/db here
    return jsonify({"status": "ok", "message": "Data received!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
