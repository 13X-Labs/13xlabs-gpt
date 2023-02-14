from flask import Flask, request, jsonify
from decouple import config
import openai
import tensorflow as tf

# Khai báo ứng dụng Flask
app = Flask(__name__)

# Cài đặt API key của OpenAI
openai.api_key = config('OPENAI_API_KEY')

# Đường dẫn đến mô hình GPT-2
model_path = "models/124M"

# Khởi tạo mô hình GPT-2
sess = tf.Session()
ckpt = tf.train.latest_checkpoint(model_path)
saver = tf.train.import_meta_graph(ckpt + ".meta")
saver.restore(sess, ckpt)

# Xác định biến "graph" để lấy đầu vào và đầu ra
graph = tf.get_default_graph()
input_text = graph.get_tensor_by_name("model/input_text:0")
output = graph.get_tensor_by_name("model/output:0")

# Định nghĩa địa chỉ API
@app.route('/api/complete', methods=['POST'])
def complete():
    data = request.get_json()
    prompt = data['prompt']
    length = data.get('length', 100)
    temperature = data.get('temperature', 0.5)

    # Sử dụng mô hình GPT-2 để hoàn thành văn bản
    with graph.as_default():
        output_text = sess.run(output, feed_dict={
            input_text: [prompt],
            "model/length:0": length,
            "model/temperature:0": temperature
        })[:, len(prompt):][0].decode('utf-8')

    # Trả về kết quả hoàn thành văn bản dưới dạng JSON
    return jsonify({
        "output": output_text
    })

# Khởi chạy ứng dụng
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
