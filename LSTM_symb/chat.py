import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import pickle

class LSTMChatBot:
    def __init__(self, model_path='./dist/lstm_chatbot.keras', vocab_path='./dist/lstm_chatbot_vocab.pkl'):
        self.model = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.chars = None
        self.seq_length = 40
        self.temperature = 0.5
        
        self.load_model(model_path, vocab_path)
    
    def load_model(self, model_path, vocab_path):
        """Загрузка предобученной модели и словаря"""
        if os.path.exists(model_path) and os.path.exists(vocab_path):
            self.model = load_model(model_path)
            with open(vocab_path, 'rb') as f:
                vocab_data = pickle.load(f)
                self.char_to_idx = vocab_data['char_to_idx']
                self.idx_to_char = vocab_data['idx_to_char']
                self.chars = vocab_data['chars']
            print("Модель и словарь успешно загружены")
        else:
            raise FileNotFoundError("Файлы модели или словаря не найдены")
    
    def preprocess_input(self, text):
        """Подготовка входных данных для модели"""
        text = text.lower().strip()
        if len(text) > self.seq_length:
            text = text[-self.seq_length:]
        elif len(text) < self.seq_length:
            text = text.rjust(self.seq_length)
        return text
    
    def generate_response(self, input_text, max_length=100):
        """Генерация ответа на входное сообщение"""
        if not self.model or not self.char_to_idx:
            return "Модель не загружена"
            
        # Подготовка входного текста
        seed = self.preprocess_input(input_text)
        generated = []
        
        for i in range(max_length):
            # Векторизация входных данных
            x = np.zeros((1, self.seq_length, len(self.chars)))
            for t, char in enumerate(seed):
                if char in self.char_to_idx:
                    x[0, t, self.char_to_idx[char]] = 1.
            
            # Предсказание следующего символа
            preds = self.model.predict(x, verbose=0)[0]
            next_index = self.sample(preds, self.temperature)
            next_char = self.idx_to_char[next_index]
            
            generated.append(next_char)
            seed = seed[1:] + next_char
            
            # Остановка при завершении предложения
            if next_char in ['\n', '.', '?', '!'] and i > max_length//2:
                break
        
        # Постобработка сгенерированного текста
        response = ''.join(generated)
        response = response.replace('\n', ' ').strip()
        return response
    
    def sample(self, preds, temperature=1.0):
        """Выбор следующего символа с учетом температуры"""
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    
    def chat(self):
        """Интерактивный режим чата"""
        print("Чат-бот готов к общению! Введите 'выход' для завершения.")
        while True:
            user_input = input("Вы: ")
            if user_input.lower() in ['выход', 'exit', 'quit']:
                print("До свидания!")
                break
                
            response = self.generate_response(user_input)
            print(f"Бот: {user_input}{response}")

# Пример использования
if __name__ == "__main__":
    bot = LSTMChatBot()
    
    # Запуск интерактивного чата
    bot.chat()