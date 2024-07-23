from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = Flask(__name__)

# Load the model and tokenizer
model_name = "MartaTT/model11epochs"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define the class to tag mapping
class_to_tag = {
    0: 'Allègement des charges fiscales',
    1: 'Subvention',
    2: 'Prise en charge des coûts',
    3: 'Avance − Prêts − Garanties',
    4: 'Allègement des charges sociales',
    5: 'Avance − Prêts − Garanties − Subvention',
    6: 'Intervention en fonds propres',
    7: 'Subvention − Allègement des charges sociales',
    8: 'Avance − Prêts − Garanties − Prise en charge des coûts',
    9: 'Subvention − Prise en charge des coûts',
    10: 'Intervention en fonds propres − Allègement des charges fiscales',
    11: 'Avance − Prêts − Garanties − Allègement des charges sociales',
    12: 'Intervention en fonds propres − Subvention',
    13: 'Avance − Prêts − Garanties − Subvention − Prise en charge des coûts'
}

@app.route('/', methods=['GET'])
def home():
    return "API is running!", 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'description' not in data:
        return jsonify({"error": "No description provided"}), 400

    description = data['description']
    inputs = tokenizer(description, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        tag_label = class_to_tag.get(predicted_class, "Unknown")

    return tag_label

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
