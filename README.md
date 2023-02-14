# 13XLabs Research

## Introduce GPT-2
GPT-2 is a natural language processing technology developed by OpenAI and has some free applications, including:

- Talk to Transformer: a tool that allows you to generate complete text from short paragraphs.

- Hugging Face's AI: a tool that allows you to generate the next sentence for any given paragraph.

- Write with Transformer: a tool that allows you to create articles, titles, and video subtitles from short paragraphs.

Note that these free applications only allow using a small portion of the features of GPT-2.

## Native Installation

All steps can optionally be done in a virtual environment using tools such as `virtualenv` or `conda`.

Install tensorflow 1.12 (with GPU support, if you have a GPU and want everything to run faster)
```
pip3 install tensorflow==1.12.0
```
or
```
pip3 install tensorflow-gpu==1.12.0
```

Install other python packages:
```
pip3 install -r requirements.txt
```

Download the model data
```
python3 download_model.py 124M
python3 download_model.py 355M
python3 download_model.py 774M
python3 download_model.py 1558M
```