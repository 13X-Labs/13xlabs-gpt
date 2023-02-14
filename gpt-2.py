import time
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM
from flask import Flask, request, jsonify

# Declare a Flask application.
app = Flask(__name__)

# Path Models GPT-3
model_name = "gpt2-xl"

# Initialize a tokenizer and GPT-2 model.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForCausalLM.from_pretrained(model_name)

# Define an API endpoint.
@app.route('/api/complete', methods=['POST'])
def complete():
    data = request.get_json()
    prompt = data['prompt']
    length = data.get('length', 100)
    temperature = data.get('temperature', 0.7)

    # Convert the prompt to a digital tokenized message.
    input_ids = tokenizer.encode(prompt, return_tensors='tf')

    # Use the GPT-2 model to complete the text
    start_time = time.time()
    output = model.generate(
        input_ids=input_ids,
        max_length=length + len(input_ids[0]),
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        top_k=0
    )
    elapsed_time = time.time() - start_time

    # Convert the output from the tokenized message back to text.
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Return the completed text as a JSON result.
    return jsonify({
        "output": output_text,
        "elapsed_time": elapsed_time
    })

# Launch the application.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)