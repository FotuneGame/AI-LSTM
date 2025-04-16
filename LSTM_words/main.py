import re
import os
import nltk
import random
import pickle
import pymorphy3
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences




# 0. Лемминг и стоп-слова

# Скачиваем стоп-слова для русского языка (выполняется один раз)
nltk.download('stopwords')

# Инициализация лемматизатора и стоп-слов
morph = pymorphy3.MorphAnalyzer()
russian_stopwords = stopwords.words('russian')

# Дополняем список стоп-слов (при необходимости)
custom_stopwords = {'это', 'весь', 'который'}
russian_stopwords.extend(custom_stopwords)

def preprocess_word(word):
    """Лемматизация и проверка на стоп-слово"""
    if not re.fullmatch(r'[а-яёa-z\-]+', word, re.IGNORECASE):
        return None
    
    lemma = morph.parse(word)[0].normal_form
    return lemma if lemma not in russian_stopwords else None




# 1. Загрузка и предварительная обработка текста
def load_and_preprocess_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    
    # Извлечение слов с фильтрацией
    words = []
    for word in re.findall(r'[а-яёa-z\-]+', text):
        processed = preprocess_word(word)
        if processed:
            words.append(processed)
    
    print(f"После обработки осталось {len(words)} значимых слов (изначально ~{len(text.split())})")
    return words



# 2. Формирование словаря слов
def create_word_vocab(words, num_words=5000):
    """Создает словарь только для лемм"""
    # Фильтрация не-слов уже выполнена в load_and_preprocess_text
    
    # Создание токенизатора
    tokenizer = Tokenizer(
        num_words=num_words,
        oov_token="<OOV>",
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n0123456789'
    )
    
    # Обучение токенизатора на лемматизированных словах
    tokenizer.fit_on_texts([' '.join(words)])
    
    # Создание словарей
    word_index = tokenizer.word_index
    index_word = {v: k for k, v in word_index.items()}
    
    print(f"Создан словарь из {len(word_index)} уникальных лемм")
    return tokenizer, word_index, index_word



# 3. Подготовка данных для обучения
def prepare_word_data(words, tokenizer, seq_length=20, step=5):
    sequences = []
    next_words = []
    
    # Создание последовательностей слов
    for i in range(0, len(words) - seq_length, step):
        sequences.append(' '.join(words[i:i + seq_length]))
        next_words.append(words[i + seq_length])
    
    # Преобразование текста в последовательности индексов
    X = tokenizer.texts_to_sequences(sequences)
    X = pad_sequences(X, maxlen=seq_length, padding='post')
    
    # Преобразование меток в one-hot кодирование
    y = np.zeros((len(next_words), len(tokenizer.word_index) + 1), dtype=np.bool)
    for i, word in enumerate(next_words):
        if word in tokenizer.word_index:
            y[i, tokenizer.word_index[word]] = 1
    
    print(f"Создано {len(X)} обучающих последовательностей")
    return X, y, tokenizer



# 4. Конструирование модели для работы со словами
def build_word_model(vocab_size, seq_length=40, embedding_dim=512, lstm_units=128):
    """
    Создает и компилирует модель LSTM для генерации текста на уровне слов
    
    Параметры:
        vocab_size (int): Размер словаря (количество уникальных слов)
        seq_length (int): Длина входной последовательности в словах (по умолчанию 40)
        embedding_dim (int): Размерность векторного представления слов (по умолчанию 512)
        lstm_units (int): Количество нейронов в LSTM слое (по умолчанию 128)
    
    Возвращает:
        model: Скомпилированная модель Keras
    """
    model = Sequential([
        Embedding(input_dim=vocab_size + 1, 
                 output_dim=embedding_dim, 
                 input_length=seq_length),  # 1. Слой Embedding - преобразует индексы слов в плотные векторы
        LSTM(lstm_units), # 2. LSTM слой - обрабатывает последовательности
        Dense(vocab_size + 1, activation='softmax') # 3. Выходной слой - предсказывает следующее слово (Вероятностное распределение)
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    # Принудительно строим модель, передав фиктивные данные
    dummy_input = np.zeros((1, seq_length))
    model(dummy_input)
    
    print("Модель создана:")
    model.summary()  # Печатает архитектуру и количество параметров в консоль
    return model



# 5. Функция генерации текста на основе слов
def generate_word_text(model, tokenizer, index_word, seed_text, temperature=1.0, length=20):
    generated = []
    
    for _ in range(length):
        # Преобразование seed_text в последовательность индексов
        sequence = tokenizer.texts_to_sequences([' '.join(seed_text)])[0]
        sequence = pad_sequences([sequence], maxlen=SEQ_LENGTH, padding='post')
        
        # Предсказание следующего слова
        preds = model.predict(sequence, verbose=0)[0]
        next_idx = sample(preds, temperature)
        next_word = index_word.get(next_idx, "<OOV>")
        
        generated.append(next_word)
        seed_text.append(next_word)
        seed_text = seed_text[1:]
    
    return ' '.join(generated)


# 6. Функция для генерации текста
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)




# Параметры модели
SEQ_LENGTH = 40  # Длина последовательности в словах
STEP = 5         # Шаг при создании последовательностей
BATCH_SIZE = 128 # Количесвто обработки текстов за раз перед обновлением весов
EMBIDDING_SIZE = 256 #Количесвто связей токена с другими
EPOCHS = 40
TEMPERATURES = [0.3, 0.7, 1.2] #степень дозволенности сети (её творческие способности лол)
MODEL_NAME = "./dist/lstm_chatbot"
MAX_WORDS = 20000  # Максимальный размер словаря

if __name__ == "__main__":
    # 1. Загрузка и предварительная обработка данных
    words = load_and_preprocess_text("../learn_data.txt")
    
    # 2. Создание словаря слов
    tokenizer, word_index, index_word = create_word_vocab(words, MAX_WORDS)
    
    # 3. Подготовка данных
    X, y, tokenizer = prepare_word_data(words, tokenizer, SEQ_LENGTH, STEP)
    
    # 4. Создание модели
    model = build_word_model(len(word_index), SEQ_LENGTH, EMBIDDING_SIZE)
    
    # Для визуализации процесса обучения
    loss_history = []
    
    # Callback для вывода примеров генерации после каждой эпохи
    def on_epoch_end(epoch, logs):
        print(f"\n\nЭпоха {epoch + 1}/{EPOCHS}")
        print(f"Текущая loss: {logs['loss']:.4f}")
        loss_history.append(logs['loss'])
        
        # Генерация текста с разными температурами
        seed_idx = random.randint(0, len(words) - SEQ_LENGTH - 1)
        seed_text = words[seed_idx:seed_idx + SEQ_LENGTH]
        print(f"\nВходной текст: {seed_text}")
        
        for temp in TEMPERATURES:
            print(f"\nТемпература {temp}:")
            generated = generate_word_text(model, tokenizer, index_word, 
                                        seed_text.copy(), temp, length=20)
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
        pickle.dump({
            'tokenizer': tokenizer,
            'word_index': word_index,
            'index_word': index_word
        }, f)
    
    # 7. Построение графика потерь
    plt.plot(range(1, EPOCHS+1), loss_history)
    plt.title('Изменение loss во время обучения')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.savefig('loss_plot.png')
    plt.show()