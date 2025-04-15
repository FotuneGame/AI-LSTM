from dotenv import dotenv_values
import os
import re
import nltk
import pickle
import random
import pymorphy3
import numpy as np
import tensorflow as tf
from telegram import Update
from nltk.corpus import stopwords
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Загрузка конфигурации
CONFIG = dotenv_values(".env")
TOKEN = CONFIG.get('TOKEN')  # Получаем токен безопасным способом

if not TOKEN:
    raise ValueError("Не найден TOKEN в .env файле")


class NLPProcessor:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self):
        """Загрузка стоп-слов с резервным вариантом"""
        try:
            nltk.data.find('corpora/stopwords')
        except:
            try:
                nltk.download('stopwords', quiet=True)
            except:
                pass
        
        try:
            stops = set(stopwords.words('russian'))
        except:
            # Fallback-список
            stops = {
                'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а',
                'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же'
            }
        
        # Дополняем пользовательскими стоп-словами
        stops.update({'это', 'вот', 'ну', 'давай', 'ладно', 'значит'})
        return stops

    def preprocess_text(self, text):
        """Полная обработка текста: токенизация + лемматизация + фильтрация"""
        words = re.findall(r'[а-яёa-z\-]+', text.lower())
        processed_words = []
        
        for word in words:
            try:
                lemma = self.morph.parse(word)[0].normal_form
                if lemma not in self.stopwords:
                    processed_words.append(lemma)
            except:
                continue
                
        return processed_words

class TelegramLSTMBot:
    def __init__(self, model_path='./dist/lstm_chatbot.keras', vocab_path='./dist/lstm_chatbot_vocab.pkl'):
        self.model = None
        self.tokenizer = None
        self.word_index = None
        self.index_word = None
        self.seq_length = 20
        self.temperature = 0.7
        self.max_words = 5000
        self.nlp = NLPProcessor()
        
        self.load_model(model_path, vocab_path)
    
    def load_model(self, model_path, vocab_path):
        if os.path.exists(model_path) and os.path.exists(vocab_path):
            self.model = load_model(model_path)
            with open(vocab_path, 'rb') as f:
                vocab_data = pickle.load(f)
                self.tokenizer = vocab_data['tokenizer']
                self.word_index = vocab_data['word_index']
                self.index_word = vocab_data['index_word']
            print("Модель и словарь успешно загружены")
        else:
            raise FileNotFoundError("Файлы модели или словаря не найдены")
    
    def preprocess_input(self, text):
        """Обработка ввода с лемматизацией"""
        words = self.nlp.preprocess_text(text)
        
        # Корректировка длины последовательности
        if len(words) > self.seq_length:
            words = words[-self.seq_length:]
        elif len(words) < self.seq_length:
            words = [''] * (self.seq_length - len(words)) + words
            
        return words
    
    def generate_response(self, input_text, max_length=20, min_prob=0.05):
        if not self.model or not self.tokenizer:
            return "Модель не загружена"
            
        seed_words = self.preprocess_input(input_text)
        if not seed_words:
            return "Не удалось обработать ваш запрос"
            
        generated = []
        for _ in range(max_length):
            sequence = self.tokenizer.texts_to_sequences([' '.join(seed_words)])
            if not sequence or not sequence[0]:
                break
                
            sequence = pad_sequences(sequence, maxlen=self.seq_length, padding='post')
            
            # Получаем предсказания модели
            preds = self.model.predict(sequence, verbose=0)[0]
            preds = np.clip(preds, 1e-10, 1.0)  # Защита от нулевых вероятностей
            
            # Выбираем следующее слово и его вероятность
            next_idx, prob = self._sample(preds, self.temperature)
            next_word = self.index_word.get(next_idx, "")
            
            # Проверяем условие остановки по низкой вероятности
            if prob < min_prob and len(generated) > 2:  # Не останавливаемся слишком рано
                break
                
            if not next_word or next_word == "<OOV>":
                break
                
            generated.append(next_word)
            seed_words.append(next_word)
            seed_words = seed_words[1:]
            
            # Остановка по знакам препинания
            if next_word in ['.', '?', '!'] and len(generated) > 2:
                break
        
        # Постобработка ответа
        response = ' '.join(generated).capitalize()
        if response and not response.endswith(('.','!','?')):
            response += '.' if len(response) > 5 else ""
            
        return response if response else "Не получилось сгенерировать ответ"

    def _sample(self, preds, temperature):
        preds = np.asarray(preds).astype('float64')
        
        # Применяем температурное масштабирование
        if temperature > 0:
            preds = np.log(preds + 1e-10) / temperature  # Защита от log(0)
            exp_preds = np.exp(preds - np.max(preds))  # Численно устойчивый вариант
            preds = exp_preds / np.sum(exp_preds)
        
        # Выбираем слово и возвращаем его вероятность
        next_idx = np.argmax(preds)
        return next_idx, preds[next_idx]

# Telegram Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """
📚 Я - LSTM-бот с NLP-обработкой:
• Автоматическая лемматизация
• Фильтрация стоп-слов
• Настройка температуры генерации

Команды:
/temp - изменить температуру (0.1-2.0)
/status - показать текущие настройки
"""
    await update.message.reply_text(help_text)

async def set_temperature(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not context.args:
            await update.message.reply_text(f"🌡 Текущая температура: {bot.temperature}")
            return
            
        new_temp = float(context.args[0])
        if 0.1 <= new_temp <= 2.0:
            bot.temperature = new_temp
            await update.message.reply_text(f"✅ Установлена температура: {bot.temperature}")
        else:
            await update.message.reply_text("❌ Температура должна быть между 0.1 и 2.0")
    except:
        await update.message.reply_text("⚠️ Использование: /temp 0.7")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    response = bot.generate_response(user_input)
    await update.message.reply_text(response)

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_msg = f"""
⚙️ Текущие настройки:
• Температура: {bot.temperature}
• NLP: {"Активен" if bot.nlp.morph else "Ограниченный режим"}
• Стоп-слов загружено: {len(bot.nlp.stopwords)}
"""
    await update.message.reply_text(status_msg)

if __name__ == '__main__':
    # Инициализация бота
    bot = TelegramLSTMBot()
    
    # Настройка Telegram бота
    app = Application.builder().token(TOKEN).build()
    
    # Регистрация обработчиков
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("temp", set_temperature))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("Бот запущен с NLP-обработкой...")
    app.run_polling()