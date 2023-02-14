import tensorflow as tf
from transformers import pipeline
from flask import Flask, request, jsonify

# Declare a Flask application.
app = Flask(__name__)

# Models GPT-Neo
model_path = "EleutherAI/gpt-neo-2.7B"

generator = pipeline('text-generation', model=model_path)

# Define an API endpoint.
@app.route('/api/complete', methods=['POST'])
def complete():
    data = request.get_json()
    prompt = data['prompt']
    length = data.get('length', 100)

    # Use the GPT-Neo model to complete the text.
    output = generator(
		prompt,
		max_length=100,
		do_sample=True,
    )

    # Convert the output from the tokenized message back to text.
    output_text = output[0]["generated_text"]

    # Return the completed text as a JSON result.``
    return jsonify({
        "output": output_text
    })

# Launch the application.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)