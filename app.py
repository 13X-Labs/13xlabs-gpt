from flask import Flask, request, jsonify
from decouple import config
import openai
import tensorflow as tf

app = Flask(__name__)

openai.api_key = config('OPENAI_API_KEY')

model_path = "models/355M"

model = tf.saved_model.load(model_path)

input_text = model.signatures["serving_default"].inputs[0]
output = model.signatures["serving_default"].outputs["output_0"]

@app.route('/api/complete', methods=['POST'])
def complete():
    data = request.get_json()
    prompt = data['prompt']
    length = data.get('length', 100)
    temperature = data.get('temperature', 0.5)

    output_text = model(input_text=tf.constant([prompt]), length=tf.constant(length), temperature=tf.constant(temperature))["output_0"].numpy()[0].decode('utf-8')

    return jsonify({
        "output": output_text
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
