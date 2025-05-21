from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route("/query", methods=["POST"])
def query_model():
    data = request.get_json()
    prompt = data.get("prompt", "")
    print(f"[Flask] Received prompt: {prompt}")

    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "tinyllama",
            "prompt": prompt,
            "stream": False
        }, timeout=90)

        output = response.json().get("response", "")
        return jsonify({"response": output})

    except Exception as e:
        print(f"[Flask] Error talking to Ollama: {e}")
        return jsonify({"response": f"\u274c Error: {e}"}), 500

if __name__ == "__main__":
    app.run(port=5000)