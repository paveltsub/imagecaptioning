from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import DenseNet201
from deep_translator import GoogleTranslator
import os
import sys

app = Flask(__name__, static_folder='static', template_folder='templates')

# Загрузка модели и токенизатора
model = load_model(os.path.join(os.path.dirname(__file__), 'model.keras'))
with open(os.path.join(os.path.dirname(__file__), 'tokenizer.pickle'), 'rb') as f:
    tokenizer = pickle.load(f)

# Параметры
max_length = 34
base_model = DenseNet201(weights='imagenet', include_top=False, pooling='avg')

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_caption_russian(image_path):
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    feature = base_model.predict(x, verbose=0)

    in_text = "startseq"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        y_pred = model.predict([feature, seq], verbose=0)
        y_pred = np.argmax(y_pred)
        word = idx_to_word(y_pred, tokenizer)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word

    caption_en = in_text.replace('startseq', '').replace('endseq', '').strip()
    caption_ru = GoogleTranslator(source='en', target='ru').translate(caption_en)
    return caption_ru

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    save_path = os.path.join(app.static_folder, 'uploaded.jpg')
    img_file.save(save_path)
    caption = generate_caption_russian(save_path)
    return jsonify({'caption': caption})

if __name__ == '__main__':
    # Получаем аргументы командной строки для хоста и порта
    host = sys.argv[1] if len(sys.argv) > 1 else '0.0.0.0'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

    app.run(debug=True, host=host, port=port)
