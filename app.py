from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)

class MonsterAPIClient:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url

    def generate_text(self, prompt, max_tokens=50, temperature=0.7):
        payload = {
            "input_variables": {"prompt": prompt},
            "stream": False,
            "max_tokens": max_tokens,
            "n": 1,
            "best_of": 1,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0,
            "use_beam_search": False,
            "length_penalty": 1.0,
            "early_stopping": False
        }
        try:
            response = requests.post(f"{self.base_url}/generate", json=payload, headers={"Authorization": f"Bearer {self.api_key}"}, verify=False)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return {"error": str(e)}

api_client = MonsterAPIClient(api_key="a8fde974-0e1f-479a-a549-13a8b7cf8eea", base_url="https://aadb94e4-212b-4b8b-8341-026e2ac0a4c7.monsterapi.ai")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    print(f"Received data: {data}")
    prompt = data.get('prompt')
    max_tokens = data.get('max_tokens', 50)
    temperature = data.get('temperature', 0.7)
    result = api_client.generate_text(prompt, max_tokens, temperature)
    print(f"Generated text: {result}")
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
