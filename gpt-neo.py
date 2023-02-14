import tensorflow as tf
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from flask import Flask, request, jsonify

# Declare a Flask application.
app = Flask(__name__)

# Models GPT-Neo
# model_path = "EleutherAI/gpt-neo-2.7B"
model_path = "EleutherAI/gpt-neo-125M"

# Initialize a tokenizer and GPT-Neo model.
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPTNeoForCausalLM.from_pretrained(model_path)

# Define an API endpoint.
@app.route('/api/complete', methods=['POST'])
def complete():
    data = request.get_json()
    prompt = data['prompt']
    length = data.get('length', 100)
    temperature = data.get('temperature', 0.5)

    # Convert the prompt to a digital tokenized message.
    input_ids = tokenizer.encode(prompt, return_tensors='tf')

    # Use the GPT-Neo model to complete the text.
    output = model.generate(
        input_ids=input_ids,
        max_length=length + len(input_ids[0]),
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        top_k=0
    )

    # Convert the output from the tokenized message back to text.
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Return the completed text as a JSON result.
    return jsonify({
        "output": output_text
    })

# Launch the application.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)