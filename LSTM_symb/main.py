import numpy as np
import random
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import LambdaCallback
import matplotlib.pyplot as plt
import os

# 1. Загрузка текстового файла
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    print(f"Загружен текст длиной {len(text)} символов")
    return text

# 2. Формирование словаря символов
def create_vocab(text):
    chars = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    print(f"Создан словарь из {len(chars)} уникальных символов")
    return chars, char_to_idx, idx_to_char

# 3. Векторизация токенов и подготовка данных
def prepare_data(text, chars, char_to_idx, seq_length=40, step=3):
    sequences = []
    next_chars = []
    
    for i in range(0, len(text) - seq_length, step):
        sequences.append(text[i:i + seq_length])
        next_chars.append(text[i + seq_length])
    
    print(f"Создано {len(sequences)} обучающих последовательностей")
    
    # Векторизация
    X = np.zeros((len(sequences), seq_length, len(chars)), dtype=np.bool)
    y = np.zeros((len(sequences), len(chars)), dtype=np.bool)
    
    for i, seq in enumerate(sequences):
        for t, char in enumerate(seq):
            X[i, t, char_to_idx[char]] = 1
        y[i, char_to_idx[next_chars[i]]] = 1
    
    return X, y

# 4. Конструирование сети
def build_model(vocab_size, lstm_units=128):
    model = Sequential([
        LSTM(lstm_units, input_shape=(SEQ_LENGTH, vocab_size)), # 1. LSTM слой - обрабатывает последовательности
        Dense(vocab_size, activation='softmax') # 2. Выходной слой - предсказывает следующее слово (Вероятностное распределение)
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print("Модель создана:")
    model.summary()
    return model

# 5. Функция генерации текста
def generate_text(model, chars, char_to_idx, idx_to_char, seed_text, temperature=1.0, length=100):
    generated = []
    for i in range(length):
        # Векторизация входных данных
        x = np.zeros((1, SEQ_LENGTH, len(chars)))
        for t, char in enumerate(seed_text):
            x[0, t, char_to_idx[char]] = 1.
        
        # Предсказание следующего символа
        preds = model.predict(x, verbose=0)[0]
        next_idx = sample(preds, temperature)
        next_char = idx_to_char[next_idx]
        
        generated.append(next_char)
        seed_text = seed_text[1:] + next_char
    
    return ''.join(generated)

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Параметры модели
SEQ_LENGTH = 40
STEP = 5
BATCH_SIZE = 128
EPOCHS = 40
MODEL_NAME = "./dist/lstm_chatbot"
TEMPERATURES = [0.2, 0.5, 1.0]  # Разные значения температуры для генерации

# Основной код
if __name__ == "__main__":
    # 1. Загрузка данных
    text = load_text("../learn_data.txt")  # Укажите путь к файлу
    
    # 2. Создание словаря
    chars, char_to_idx, idx_to_char = create_vocab(text)
    
    # 3. Подготовка данных
    X, y = prepare_data(text, chars, char_to_idx, SEQ_LENGTH, STEP)
    

    # 4. Создание модели
    model = build_model(len(chars))
    
    # Для визуализации процесса обучения
    loss_history = []
    
    # Callback для вывода примеров генерации после каждой эпохи
    def on_epoch_end(epoch, logs):
        print(f"\nЭпоха {epoch + 1}/{EPOCHS}")
        print(f"Текущая loss: {logs['loss']:.4f}")
        loss_history.append(logs['loss'])
        
        # Генерация текста с разными температурами
        seed_idx = random.randint(0, len(text) - SEQ_LENGTH - 1)
        seed_text = text[seed_idx:seed_idx + SEQ_LENGTH]
        
        for temp in TEMPERATURES:
            print(f"\nТемпература {temp}:")
            generated = generate_text(model, chars, char_to_idx, idx_to_char, 
                                   seed_text, temp, length=100)
            print(generated)
    
    # 5. Обучение модели
    print("\nНачало обучения...")
    model.fit(X, y,
             batch_size=BATCH_SIZE,
             epochs=EPOCHS,
             callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])
    
    # 6. Сохранение модели и словаря
    model.save(f"{MODEL_NAME}.keras")
    with open(f"{MODEL_NAME}_vocab.pkl", "wb") as f:
        pickle.dump({'chars': chars, 
                    'char_to_idx': char_to_idx, 
                    'idx_to_char': idx_to_char}, f)
        
    # 7. Построение графика потерь
    plt.plot(range(1, EPOCHS+1), loss_history)
    plt.title('Изменение loss во время обучения')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.savefig('loss_plot.png')
    plt.show()