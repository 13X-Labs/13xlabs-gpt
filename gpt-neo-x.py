import tensorflow as tf
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
from flask import Flask, request, jsonify

# Declare a Flask application.
app = Flask(__name__)

# Models GPT-Neo-X
model_path = "EleutherAI/gpt-neox-20b"

model = GPTNeoXForCausalLM.from_pretrained(model_path)
tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_path)

# Define an API endpoint.
@app.route('/api/complete', methods=['POST'])
def complete():
    data = request.get_json()
    prompt = data['prompt']
    length = data.get('length', 100)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
  
    # Use the GPT-Neo model to complete the text.
    output = model.generate(
		input_ids,
		max_length=100,
		temperature=0.9,
		do_sample=True,
    )

    # Convert the output from the tokenized message back to text.
    output_text = tokenizer.batch_decode(output)[0]

    # Return the completed text as a JSON result.``
    return jsonify({
        "output": output_text
    })
# Launch the application.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)