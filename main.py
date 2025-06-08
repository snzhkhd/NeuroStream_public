from __future__ import annotations
import asyncio
import mysql.connector
from mysql.connector import Error
import json
import time
import pytchat
import requests
import requests.exceptions
from datetime import datetime, timedelta
import ffmpeg
import subprocess 
import os
from pydub import AudioSegment
import shutil
import uuid
import random
from typing import List, Optional
from pathlib import Path
import re
from openai import OpenAI
from tqdm import tqdm
from urllib.parse import urlencode
import threading
import socketio
import uuid
import logging
import aiomysql
import websockets
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception,before_sleep_log
from tenacity.retry import retry_if_exception_type  # Явный импорт
from openai import AsyncOpenAI, OpenAIError 
from aiohttp import ClientError 
from googleapiclient.discovery import build
import httpx
import aiofiles
from tts_manager import TTSManager
import traceback
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)


#donationalerts
CLIENT_ID = ""
CLIENT_SECRET = "" 
ACCESS_TOKEN = ""
USER_ID = ""
REFRESH_TOKEN = ""
TOKEN_EXPIRY = datetime.now()

YOUTUBE_API_KEY = ""
CHANNEL_ID = "UC-"

LOG_DIR = "log"
log_file = None

VIDEO_DIR = "output_videos/"
os.makedirs(VIDEO_DIR, exist_ok=True)

AUDIO_DIR = "temp_audio/"
STREAM_ID = "qcUzq40Rv6I" 
API_DOMAIN = "neurostream.local"


TTS_SERVERS = [8000, 8100] # порты для fish-speech

FFMPEG_PATH = "./ffmpeg.exe"
# Инициализируем менеджер после создания логгера
tts_manager = TTSManager(TTS_SERVERS)

# Флаг для отслеживания состояния соединения
connected = False


async_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="YOU_KEY",
)

# Конфигурация
mysql_config = {
    "host": "mysql-8.0", 
    "user": "root",    
    "password": "",
    "db": "my_database",  
    "port": 3306,                 # Порт MySQL (стандартный 3306)
    "charset": "utf8mb4"
}

async def save_tokens(access_token, refresh_token, expiry_time):
    """Сохранение токенов в БД"""
    try:
        async with aiomysql.connect(**mysql_config) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO auth_tokens (access_token, refresh_token, expiry_time)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE 
                        access_token = %s,
                        refresh_token = %s,
                        expiry_time = %s
                    """,
                    (access_token, refresh_token, expiry_time,
                     access_token, refresh_token, expiry_time)
                )
                await conn.commit()
                console_msg("Токены сохранены в БД", "green")
    except Exception as e:
        console_msg(f"Ошибка сохранения токенов: {str(e)}", "red")

async def get_tokens():
    """Получение токенов из БД"""
    try:
        async with aiomysql.connect(**mysql_config) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT * FROM auth_tokens ORDER BY id DESC LIMIT 1")
                row = await cursor.fetchone()
                if row:
                    return {
                        "access_token": row[1],
                        "refresh_token": row[2],
                        "expiry_time": row[3]
                    }
                return None
    except Exception as e:
        console_msg(f"Ошибка получения токенов из БД: {str(e)}", "red")
        return None

async def refresh_access_token():
    # Обновляем глобальные переменные
    global ACCESS_TOKEN, REFRESH_TOKEN, TOKEN_EXPIRY
    """Обновление токенов через Refresh Token"""
    url = "https://www.donationalerts.com/oauth/token"
    data = {
        "grant_type": "refresh_token",
        "refresh_token": REFRESH_TOKEN,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope": "oauth-donation-subscribe oauth-user-show"
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        tokens = response.json()
        access_token = tokens["access_token"]
        refresh_token = tokens.get("refresh_token", REFRESH_TOKEN)
        expires_in = tokens["expires_in"]
        expiry_time = datetime.now() + timedelta(seconds=expires_in)
        
        
        ACCESS_TOKEN = access_token
        REFRESH_TOKEN = refresh_token
        TOKEN_EXPIRY = expiry_time
        
        # Сохраняем в БД
        await save_tokens(ACCESS_TOKEN, REFRESH_TOKEN, TOKEN_EXPIRY)
        console_msg("Токены успешно обновлены", "green")
    else:
        console_msg(f"Ошибка обновления токенов: {response.text}", "red")
        raise Exception("Refresh Token недействителен")
        
async def get_socket_token():
    """Получение socket_connection_token для Centrifugo"""
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    response = requests.get("https://www.donationalerts.com/api/v1/user/oauth", headers=headers)
    if response.status_code == 200:
        return response.json()["data"]["socket_connection_token"]
    else:
        raise Exception("Не удалось получить socket_connection_token")

async def connect_to_centrifugo():
    """Асинхронное подключение к Centrifugo"""
    os.system("ipconfig /flushdns") 
    uri = "wss://centrifugo.donationalerts.com/connection/websocket"
    while True:
        try:
            async with websockets.connect(uri,ping_interval=20, ping_timeout=60,close_timeout=1) as ws:
                # Получаем socket_connection_token
                socket_token = await get_socket_token()  # Пример асинхронной обертки
                await ws.send(json.dumps({
                    "params": {"token": socket_token},
                    "id": 1
                }))
                auth_response = await ws.recv()
                client_id = json.loads(auth_response)['result']['client']

                # Подписка на канал
                subscribe_url = "https://www.donationalerts.com/api/v1/centrifuge/subscribe"
                data = {
                    "channels": [f"$alerts:donation_{USER_ID}"],
                    "client": client_id
                }
                headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
                res = requests.post(subscribe_url, json=data, headers=headers)
                subscription_token = res.json()['channels'][0]['token']

                # Подписываемся на канал
                await ws.send(json.dumps({
                    "params": {"channel": f"$alerts:donation_{USER_ID}", "token": subscription_token},
                    "method": 1,
                    "id": 2
                }))

                # Слушаем уведомления
                while True:
                    try:
                        message = await ws.recv()
                        data = json.loads(message)
                        if (
                            'result' in data and
                            'data' in data['result'] and
                            'data' in data['result']['data'] and  # Вложенность данных
                            'username' in data['result']['data']['data']  # Проверка ключа username
                        ):
                            await on_donation(data)  # Используем существующий обработчик
                    except websockets.exceptions.ConnectionClosed:
                        console_msg("Соединение с Centrifugo разорвано", "yellow")
                        break
        except Exception as e:
            console_msg(f"Ошибка WebSocket: {str(e)}", "red")
            logger.error(f"WebSocket error: {e}")
            await asyncio.sleep(5)

async def token_refresher():
    """Автоматическое обновление токенов каждые 60 минут"""
    while True:
        await asyncio.sleep(3600)  # Каждый час
        if datetime.now() > (TOKEN_EXPIRY - timedelta(minutes=10)):
            await refresh_access_token()

async def get_initial_tokens():
    """Получение токенов из БД при старте"""
    global ACCESS_TOKEN, REFRESH_TOKEN, TOKEN_EXPIRY
    tokens = await get_tokens()
    if tokens:
        ACCESS_TOKEN = tokens["access_token"]
        REFRESH_TOKEN = tokens["refresh_token"]
        TOKEN_EXPIRY = tokens["expiry_time"]
        console_msg("Токены загружены из БД", "green")
    else:
        # Если токенов нет, выполнить OAuth-авторизацию
        await perform_initial_oauth()

async def perform_initial_oauth():
    """Инициализация OAuth при первом запуске"""
    global ACCESS_TOKEN, REFRESH_TOKEN, TOKEN_EXPIRY
    
    # Формируем URL для авторизации
    auth_url = f"https://www.donationalerts.com/oauth/authorize?" \
               f"client_id={CLIENT_ID}&" \
               f"redirect_uri=http://localhost/callback&" \
               f"response_type=code&" \
               f"scope=oauth-donation-subscribe%20oauth-user-show"
    
    console_msg("Откройте ссылку в браузере и авторизуйтесь:", "cyan")
    print(auth_url)
    code = input("Вставьте код из URL: ")
    
    # Обмен кода на токены
    token_url = "https://www.donationalerts.com/oauth/token"
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": "http://localhost/callback"
    }
    response = requests.post(token_url, data=data)
    if response.status_code == 200:
        tokens = response.json()
        access_token = tokens["access_token"]
        refresh_token = tokens["refresh_token"]
        expires_in = tokens["expires_in"]
        expiry_time = datetime.now() + timedelta(seconds=expires_in)
        
        # Обновляем глобальные переменные
        ACCESS_TOKEN = access_token
        REFRESH_TOKEN = refresh_token
        TOKEN_EXPIRY = expiry_time
        
        # Сохраняем в БД
        await save_tokens(ACCESS_TOKEN, REFRESH_TOKEN, expiry_time)
        console_msg("Токены успешно инициализированы", "green")
    else:
        console_msg(f"Ошибка инициализации токенов: {response.text}", "red")
        raise Exception("Не удалось получить токены")

def init_log():
    """Создает папку для логов и открывает файл для записи"""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_filename = datetime.now().strftime("log_%Y%m%d_%H%M%S.txt")
    log_path = os.path.join(LOG_DIR, log_filename)
    log_file = open(log_path, "a", encoding="utf-8")
    return log_file
# Функция для логгирования
def console_msg(message, color='white'):
    global log_file
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    colors = {
        'white': '\x1b[37m',
        'yellow': '\x1b[33m',
        'red': '\x1b[31m',
        'green': '\x1b[32m',
        'cyan': '\x1b[36m'
    }
    
    # Вывод в консоль
    console_str = f"{colors[color]}{now} {message}\x1b[0m"
    print(console_str)
    
    # Запись в файл
    if log_file:
        log_str = f"{now} {message}"
        log_file.write(log_str + "\n")
        log_file.flush()  # Принудительная запись на диск [[1]]
    else:
        print("Ошибка: log_file не инициализирован")
        
def test_mysql_connection():
    try:
        conn = mysql.connector.connect(**mysql_config)
        if conn.is_connected():
            console_msg("УСПЕШНОЕ ПОДКЛЮЧЕНИЕ К MySQL!")
            conn.close()
        else:
            console_msg("НЕ УДАЛОСЬ ПОДКЛЮЧИТЬСЯ К MySQL!")
    except Error as e:
        console_msg(f"ОШИБКА ПОДКЛЮЧЕНИЯ: {e}")
        
# Подключение к БД
def init_db():
    try:
        conn = mysql.connector.connect(**mysql_config)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id INT AUTO_INCREMENT PRIMARY KEY,
                author VARCHAR(255) NOT NULL,
                character_name VARCHAR(255) NOT NULL,
                topic TEXT NOT NULL,
                status VARCHAR(50) DEFAULT 'queued',
                video_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP,
                priority INT DEFAULT 0,
                donation_level INT DEFAULT 0
            )
        ''')
        # Новая таблица для истории
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS played_videos (
                id INT PRIMARY KEY,
                author VARCHAR(255) NOT NULL,
                character_name VARCHAR(255) NOT NULL,
                topic TEXT NOT NULL,
                video_path TEXT,
                played_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        console_msg("Таблица tasks создана/проверена")
    except Error as e:
        console_msg(f"Ошибка инициализации БД: {e}",'red')
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

async def get_queue_status():
    async with aiomysql.connect(**mysql_config) as conn:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN donation_level = 2 THEN 1 ELSE 0 END) as super_donors,
                    SUM(CASE WHEN donation_level = 1 THEN 1 ELSE 0 END) as donors,
                    SUM(CASE WHEN donation_level = 0 THEN 1 ELSE 0 END) as regular
                FROM tasks 
                WHERE status = 'queued'
            """)
            return await cursor.fetchone()

async def cleanup_processed_donations():
    while True:
        try:
            conn = mysql.connector.connect(**mysql_config)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM processed_donations WHERE processed_at < NOW() - INTERVAL 1 DAY")
            conn.commit()
        except Error as e:
            console_msg(f"Ошибка очистки: {e}", "red")
        await asyncio.sleep(86400)  # Каждые 24 часа

async def is_donation_processed(donation_id):
    async with aiomysql.connect(**mysql_config) as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1 FROM processed_donations WHERE donation_id = %s", (donation_id,))
            return await cur.fetchone() is not None

async def mark_donation_as_processed(donation_id):
    async with aiomysql.connect(**mysql_config) as conn:
        async with conn.cursor() as cur:
            await cur.execute("INSERT IGNORE INTO processed_donations (donation_id) VALUES (%s)", (donation_id,))
            await conn.commit()

# Добавление задачи в БД
async def add_task_to_db(author, character, topic, donation_level):
    try:
        # Создаем асинхронное подключение
        async with aiomysql.connect(
            host=mysql_config['host'],
            port=mysql_config['port'],
            user=mysql_config['user'],
            password=mysql_config['password'],
            db=mysql_config['db']
        ) as conn:
            
            async with conn.cursor() as cursor:
                # Проверяем лимит очереди
                queue_status = await get_queue_status()
                if queue_status['total'] >= 20 and donation_level == 0:
                    console_msg("Очередь переполнена, задача не добавлена", "yellow")
                    return False

                # Вставляем новую задачу
                query = """
                    INSERT INTO tasks 
                    (author, character_name, topic, donation_level, priority) 
                    VALUES (%s, %s, %s, %s, %s)
                """
                priority = donation_level * 1000
                await cursor.execute(query, (
                    author, 
                    character, 
                    topic, 
                    donation_level, 
                    priority
                ))
                
                await conn.commit()
                console_msg(f"Задача добавлена: {topic}")
                return True

    except Exception as e:
        console_msg(f"Ошибка добавления задачи: {str(e)}", "red")
        return False

# Функция проверки статуса задачи
async def get_task_status(task_id: int) -> str:
    """Получает статус задачи из БД"""
    async with aiomysql.connect(**mysql_config) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                "SELECT status FROM tasks WHERE id = %s",
                (task_id,)
            )
            result = await cursor.fetchone()
            return result[0] if result else None

async def get_next_task():
    async with aiomysql.connect(**mysql_config) as conn:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            # Используем транзакцию с блокировкой
            await conn.begin()
            
            try:
                await cursor.execute('''
                    SELECT * FROM tasks 
                    WHERE status = 'queued' 
                    ORDER BY priority DESC, id ASC 
                    LIMIT 1 FOR UPDATE SKIP LOCKED
                ''')
                task = await cursor.fetchone()
                
                # if task:
                    # await cursor.execute('''
                        # UPDATE tasks 
                        # SET status = 'processing' 
                        # WHERE id = %s
                    # ''', (task['id'],))
                    # await conn.commit()
                    
                return task
            except Exception as e:
                await conn.rollback()
                raise


# Обновление статуса задачи
async def update_task_status(task_id, status, video_path=None):
    async with aiomysql.connect(**mysql_config) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute('''
                UPDATE tasks 
                SET status = %s, 
                    video_path = %s, 
                    processed_at = NOW() 
                WHERE id = %s
            ''', (status, video_path, task_id))
            console_msg(f"Обновление задачи {task_id}: status={status}, video_path={video_path}")  # Лог 5
            await conn.commit()

def get_video_duration(video_path: Path) -> float:
    """Получаем длительность видео файла"""
    try:
        result = subprocess.run(
            [
                'ffprobe', 
                '-v', 'error', 
                '-show_entries', 'format=duration', 
                '-of', 'default=noprint_wrappers=1:nokey=1', 
                str(video_path)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        console_msg(f"Ошибка получения длительности: {str(e)}", "red")
        return 0

async def process_video_clip_async(input_video, audio_path, output_path, duration, ffmpeg_path, subtitle_text=None, author=None, topic=None):
    try:
        await asyncio.to_thread(
            process_video_clip, 
            input_video, 
            audio_path, 
            output_path, 
            duration, 
            ffmpeg_path, 
            subtitle_text,
            author,
            topic
        )
    except Exception as e:
        console_msg(f"Асинхронная ошибка обработки: {str(e)}", "red")
        return False
    return True
    
async def reset_task_to_queued(task_id: int):
    """Возвращает задачу в очередь и очищает диалог"""
    try:
        async with aiomysql.connect(**mysql_config) as conn:
            async with conn.cursor() as cursor:
                # Обновляем статус и очищаем диалог
                await cursor.execute(
                    "UPDATE tasks SET status = 'queued', dialogue = NULL WHERE id = %s",
                    (task_id,)
                )
                await conn.commit()
                console_msg(f"Задача {task_id} сброшена в очередь и очищена", "green")
    except Exception as e:
        console_msg(f"Ошибка сброса задачи {task_id}: {str(e)}", "red")
        await conn.rollback() if "conn" in locals() else None    
    
async def reset_all_errors():
    """Сброс всех задач с ошибкой в очередь"""
    try:
        async with aiomysql.connect(**mysql_config) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "UPDATE tasks SET status = 'queued', dialogue = NULL WHERE status = 'error'"
                )
                await conn.commit()
                console_msg(f"Сброшено {cursor.rowcount} задач", "green")
    except Exception as e:
        console_msg(f"Ошибка сброса всех задач: {str(e)}", "red")
        
def process_video_clip(input_video, audio_path, output_path, duration, ffmpeg_path, subtitle_text=None, author=None, topic=None):
    """Обработка одного видео-клипа с синхронизацией аудио и субтитрами"""
    try:
        # 1. Обрезаем видео до нужной длительности
        temp_video = output_path.with_suffix('.tmp.mp4')
        font_path = './arial.ttf'

        if subtitle_text:
            if not os.path.exists('./arial.ttf'):
                raise Exception("Шрифт arial.ttf не найден")
                
        # Проверка поддержки NVENC
        nvenc_params = {
            'vcodec': 'h264_nvenc',
            'preset': 'p6',
            'cq': '23',
            'acodec': 'aac',
            'b:a': '192k',      # Битрейт аудио
            'ar': '44100',      # Частота дискретизации
            'ac': '2',          # Каналы (стерео)
            'movflags': '+faststart',
            'vsync': 'vfr'
        }    
        
         # Получаем входной видео поток
        video_stream = ffmpeg.input(str(input_video))
        video_stream = (
            video_stream
            .filter('trim', duration=duration)
            # Стандартизация разрешения
            .filter('scale', 1280, 720)
            # Стандартизация FPS
            .filter('fps', fps=30)
        )
        
        # Добавляем субтитры, если текст передан
        if subtitle_text:
            subtitle_text = (
                subtitle_text
                .replace('\\', '\\\\')  # Экранирование обратных слешей
                .replace(':', '\\:')    # Двоеточия
                .replace("'", "\\'")    # Апострофы
                .replace('%', '%%')     # Проценты
            )
            video_stream = video_stream.filter(
                'drawtext',
                text=subtitle_text,
                fontfile=font_path,
                fontsize=24,
                fontcolor='white',
                x='(w-text_w)/2',
                y='h-th-40'
            )
        # author=None
        # topic=None
        if author or topic:
            if author:
                video_stream = video_stream.filter(
                    'drawtext',
                    text=f"Автор: {author}",
                    fontfile=font_path,
                    fontsize=20,
                    fontcolor='white',
                    x='20',  # Правый верхний угол
                    y='20'
                )
            
            if topic:
                topic_short = topic[:50]  # Обрезаем до 50 символов
                video_stream = video_stream.filter(
                    'drawtext',
                    text=f"Тема: {topic_short}",
                    fontfile=font_path,
                    fontsize=20,
                    fontcolor='white',
                    x='20',  # Правый верхний угол, ниже автора
                    y='50'
                )
        # Обрезаем видео
        # Первый шаг: сохраняем видео с субтитрами
        try:
            (
                ffmpeg
                .output(
                    video_stream.filter('trim', duration=duration),
                    ffmpeg.input(audio_path),
                    str(output_path),
                    **nvenc_params
                )
                .overwrite_output()
                .run(cmd=ffmpeg_path, capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            console_msg(f"FFmpeg Error (video processing): {e.stderr.decode()}", "red")
            return False
        finally:
            if temp_video.exists():
                temp_video.unlink()

        return True

    except Exception as e:
        console_msg(f"Общая ошибка обработки: {str(e)}", "red")
        return False
 

def normalize_name(name: str, config: dict) -> str:
    # Приводим к нижнему регистру и удаляем лишние символы
    cleaned = re.sub(r'[^а-яёa-z]', '', name.lower(), flags=re.IGNORECASE)
    
    # Создаем словарь {нормализованное_имя: оригинальное_имя}
    valid_names = {re.sub(r'[^а-яё]', '', s["name"].lower()): s["name"] 
                   for s in config["speakers"]}
    
    # Ищем ближайшее совпадение
    for norm, original in valid_names.items():
        if norm == cleaned:
            return original
        
    # Если точного совпадения нет - используем нечеткий поиск
    from thefuzz import fuzz  # pip install thefuzz
    matches = [(original, fuzz.ratio(cleaned, norm)) 
              for norm, original in valid_names.items()]
    best_match = max(matches, key=lambda x: x[1])
    
    if best_match[1] > 50:  # Порог схожести
        console_msg(f"Исправлено имя: '{name}' -> '{best_match[0]}'", "yellow")
        return best_match[0]
    
    raise ValueError(f"Персонаж '{name}' не найден. Доступные: {list(valid_names.values())}")

# Асинхронная функция для запросов к LLM с повторами
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type(
        (asyncio.TimeoutError, httpx.ConnectError, httpx.ReadTimeout, OpenAIError)
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
async def get_llm_response_async(prompt: str, timeout: int = 45) -> str:
    # Инициализация счетчика попыток
    if not hasattr(get_llm_response_async, "attempt"):
        get_llm_response_async.attempt = 0
    get_llm_response_async.attempt += 1

    try:
        if not hasattr(async_client, 'chat'):
            raise RuntimeError("Неверная конфигурация клиента LLM")
        
        # Для Python <3.11 используем asyncio.wait_for
        try:
            response = await asyncio.wait_for(
                async_client.chat.completions.create(
                    model="deepseek/deepseek-chat-v3-0324:free:nitro",  #"deepseek/deepseek-chat:free:nitro",       "deepseek/deepseek-chat-v3-0324:free"
                    messages=[{"role": "user", "content": prompt}],
                    timeout=httpx.Timeout(10.0, connect=15.0, read=timeout)
                ),
                timeout=timeout + 5  # Общий таймаут для всей операции
            )
            if not response or not response.choices:
                logger.error(f"Пустой ответ от LLM. Полный ответ: {response}")
                raise ValueError("Пустой ответ от LLM")
            first_choice = response.choices[0]
            if not hasattr(first_choice, 'message') or not hasattr(first_choice.message, 'content'):
                logger.error(f"Некорректная структура ответа: {response}")
                raise ValueError("Некорректная структура ответа LLM")
            content = first_choice.message.content.strip()
            if not content:
                raise ValueError("Пустое содержимое ответа")
                
            return content
        except asyncio.TimeoutError as e:
            raise asyncio.TimeoutError("Превышен общий таймаут 15с") from e

        return response.choices[0].message.content
    except Exception as e:
        console_msg(f"Попытка {get_llm_response_async.attempt} провалена: {str(e)}", "yellow")
        raise
    finally:
        # Сброс счетчика после завершения всех попыток
        if get_llm_response_async.attempt >= 5:
            get_llm_response_async.attempt = 0

async def add_cover_to_video(input_path: str, output_path: str, author: str, topic: str, font_path: str):
    """Добавляет текстовую обложку к видео через FFmpeg"""
    temp_thumb = Path(output_path).with_suffix(".tmp.png")
    text = f"Автор: {author}\nТема: {topic[:80]}" if topic else f"Автор: {author}"

    # Генерация статического изображения с текстом
    (
        ffmpeg
        .input('color=black', format='lavfi')
        .filter('scale', 1280, 720)
        .filter('drawtext',
                text=text,
                fontfile=font_path,
                fontsize=74,
                fontcolor='white',
                box=1,
                boxcolor='black@0.7',
                x='(w-text_w)/2',
                y='(h-text_h)/2')
        .output(str(temp_thumb), vframes=1, format='image2')
        .run(quiet=True, capture_stdout=True, capture_stderr=True)
    )

    # Входные файлы
    input_video = ffmpeg.input(input_path)
    input_thumb = ffmpeg.input(temp_thumb)

    # Добавление обложки в метаданные (без изменения видео)
    (
        ffmpeg
        .output(
            input_video,
            input_thumb,
            str(output_path),
            vcodec="copy",
            acodec="copy",
            **{"map": "0", "map": "1", "disposition:v:1": "attached_pic"}  # Исправленный `map`
        )
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )

    # Удаление временного изображения
    temp_thumb.unlink()

async def process_video_segment(speaker, text, config, audio_path, output_path, author=None, topic=None):
    try:
        speaker_config = next(
            (s for s in config["speakers"] if s["name"] == speaker),  # Исправлено
            None
        )
        
        if not speaker_config:
            raise ValueError(f"Спикер {speaker_name} не найден в конфиге")

        # Выбираем случайное видео из доступных для этого спикера
        video_clip = Path(random.choice(speaker_config["video_clips"])).resolve()
        
        success = await process_video_clip_async(
            input_video=video_clip,
            audio_path=audio_path,
            output_path=output_path,
            duration=await get_audio_duration(audio_path),
            ffmpeg_path=FFMPEG_PATH,
            subtitle_text=f"{speaker}: {text}",
            author=author,
            topic=topic
        )
        return output_path if success else None
    except Exception as e:
        console_msg(f"Ошибка обработки видео: {str(e)}", "red")
        return None

async def get_audio_duration(path: Path) -> float:
    audio = await asyncio.to_thread(AudioSegment.from_wav, path)
    return len(audio) / 1000    
# Основная функция обработки задачи
async def process_task(task):
    task_id = None
    tts_port = None
    temp_dir = None
    config = None
    try:
        if not task:
            console_msg("Получена пустая задача", "yellow")
            return

        task_id = task['id']
        author = task['author'] 
        character = task['character_name']
        topic = task['topic']
        temp_dir = Path(f"temp_task_{task_id}")
    
        # Атомарная проверка статуса и блокировка задачи
        async with aiomysql.connect(**mysql_config) as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await conn.begin()
                try:
                    await cursor.execute(
                        "SELECT status FROM tasks WHERE id = %s FOR UPDATE",
                        (task_id,)
                    )
                    result = await cursor.fetchone()
                    
                    if not result or result['status'] != 'queued':
                        console_msg(f"Задача {task_id} уже обрабатывается", "yellow")
                        await conn.rollback()
                        return

                    # Обновление статуса в транзакции
                    await cursor.execute(
                        "UPDATE tasks SET status = 'processing' WHERE id = %s",
                        (task_id,)
                    )
                    await conn.commit()
                except Exception as e:
                    await conn.rollback()
                    raise

        console_msg(f"Начата обработка задачи {task_id}: {topic}", "cyan")
        

        # Выделение TTS-сервера для всей задачи
        tts_port = await tts_manager.acquire(task_id)
        console_msg(f"Сервер {tts_port} выделен для задачи {task_id}", "green")
        
        config_path = f"characters/{character}.json"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Конфиг {character} не найден")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Получаем текущий диалог из БД
        async with aiomysql.connect(**mysql_config) as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    "SELECT dialogue FROM tasks WHERE id = %s",
                    (task_id,)
                )
                result = await cursor.fetchone()
                existing_dialogue = result['dialogue'] if result else None

        # Если диалог уже существует - используем его
        if existing_dialogue:
            console_msg(f"Используем существующий диалог для задачи {task_id}", "green")
            dialogue = existing_dialogue
        else:
            # Генерация нового диалога
            try:
                prompt = config["prompt"].format(topic=topic)
                dialogue = await get_llm_response_async(prompt, timeout=15)
                console_msg("Диалог сгенерирован успешно", "green")
                
                # Сохраняем новый диалог в БД
                async with aiomysql.connect(**mysql_config) as conn:
                    async with conn.cursor() as cursor:
                        await cursor.execute(
                            "UPDATE tasks SET dialogue = %s WHERE id = %s",
                            (dialogue, task_id)
                        )
                        await conn.commit()
                        
            except Exception as e:
                console_msg(f"Ошибка LLM: {str(e)}", "red")
                dialogue = await generate_fallback_dialogue(author, e)


        # Обработка и очистка диалога
        dialogue = sanitize_dialogue(dialogue)
        dialogue_lines = parse_dialogue_lines(dialogue,config)

        # Создание рабочей директории
        temp_dir.mkdir(exist_ok=True)
        video_clips = []

        # Проверка FFmpeg
        await verify_ffmpeg()
        log_file = temp_dir / "ffmpeg.log"    
        # Обработка каждого фрагмента
        for idx, (speaker, text) in tqdm(enumerate(dialogue_lines), total=len(dialogue_lines)):
            try:
                # Генерация аудио
                audio_path = await generate_audio(
                    speaker,
                    text,
                    config,
                    tts_port,
                    temp_dir / f"audio_{idx}.wav"
                )

                # Обработка видео
                video_path = await process_video_segment(
                    speaker,
                    text,
                    config,
                    audio_path,
                    temp_dir / f"segment_{idx}.mp4",
                    author=author,
                    topic=topic
                )

                video_clips.append(video_path)

            except Exception as e:
                console_msg(f"Ошибка обработки сегмента {idx}: {str(e)}", "red")
                continue

        # Получаем настройки музыки (если есть)
        music_config = config.get("background_music", {})
        add_music = music_config.get("enabled", False)
        music_path = music_config.get("path", "./music/prokopenko.mp3")  # путь по умолчанию
        music_volume = music_config.get("volume", 0.3)  # громкость по умолчанию
        # Финальный монтаж
        if video_clips:
            final_path = await render_final_video(
                task_id=task_id,
                video_clips=video_clips,
                temp_dir=temp_dir,
                music_config=music_config if add_music else None
            )
            await update_task_status(task_id, "completed", final_path)
            console_msg(f"Видео готово: {final_path}", "green")
            await tts_manager.release(task_id)
        else:
            raise Exception("Не удалось создать ни одного видео-сегмента")
            # await reset_task_to_queued(task_id)

    except Exception as e:
        console_msg(f"Критическая ошибка: {str(e)}", "red")
        await update_task_status(task_id, "error", str(e))
        # await reset_task_to_queued(task_id)
    finally:
        # Гарантированная очистка ресурсов
        if tts_port:
            await tts_manager.release(task_id)
        if temp_dir.exists():
            await cleanup_temp_dir(temp_dir)

# Вспомогательные функции
async def generate_fallback_dialogue(author: str, error: Exception) -> str:
    try:
        with open("characters/sanych2049.json") as f:
            config = json.load(f)
        return f"Саныч: {author}, ошибка генерации: {str(error)}"
    except Exception:
        return "Оператор: Критическая системная ошибка"

def sanitize_dialogue(text: str) -> str:
    return re.sub(r'[*\\/"]', '', text)

def parse_dialogue_lines(text: str, config: dict) -> list:
    dialogue_lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line or ": " not in line:
            continue
            
        speaker, content = line.split(': ', 1)
        try:
            normalized_speaker = normalize_name(speaker, config)
            dialogue_lines.append((normalized_speaker, content.strip()))
        except ValueError as e:
            console_msg(f"Пропущена реплика: {str(e)}", "yellow")
            continue
            
    return dialogue_lines

async def verify_ffmpeg():
    result = await asyncio.to_thread(
        subprocess.run,
        [FFMPEG_PATH, "-version"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError("FFmpeg не работает")

async def generate_audio(speaker: str, text: str, config: dict, port: int, path: Path) -> Path:
    try:
        # Получаем конфигурацию спикера
        speaker_config = next(
            (s for s in config["speakers"] if s["name"] == speaker),
            None
        )
        if not speaker_config:
            raise ValueError(f"Персонаж {speaker} не найден в конфигурации")
        # Базовые настройки TTS
        tts_settings = {
            "max_new_tokens": 1024,
            "chunk_length": 250,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "temperature": 0.7,
            "seed": 0
        }
        
        # Обновляем настройки из конфига персонажа
        if "tts" in config:
            tts_settings.update(config["tts"])

        # Формируем payload
        payload = {
            "text": f"[{speaker_config['audio_ref']}] {text}",
            "references": speaker_config["audio_ref"],
            "output_format": "wav",
            **tts_settings  # Используем только TTS-параметры
        }

        # Отправляем запрос
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post( f"http://localhost:{port}/tts/simple",data=payload )
            response.raise_for_status()

        # Сохраняем аудио
        async with aiofiles.open(path, "wb") as f:
            await f.write(response.content)
            
        return path

    except Exception as e:
        console_msg(f"Ошибка генерации аудио для {speaker}: {str(e)}", "red")
        raise

async def cleanup_temp_dir(path: Path):
    try:
        await asyncio.to_thread(shutil.rmtree, path)
    except Exception as e:
        console_msg(f"Ошибка очистки временных файлов: {str(e)}", "yellow")

async def render_final_video(
    task_id: int,
    video_clips: List[Path],
    temp_dir: Path,
    music_config: Optional[dict] = None
) -> Path:
    """Объединяет видео-клипы в финальный ролик."""
    try:
        # Создаем список для конкатенации
        concat_list = temp_dir / "concat_list.txt"
        with open(concat_list, "w", encoding="utf-8") as f:
            for clip in video_clips:
                f.write(f"file '{clip.absolute()}'\n")
        
        
        video_duration = sum(get_video_duration(clip) for clip in video_clips)
        # Финальный путь
        final_path = Path(VIDEO_DIR) / f"{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

        concat_params = {
            'c:v': 'h264_nvenc',
            'preset': 'slow',          # Лучшее сжатие
            'crf': '26',               # Баланс качества/размера
            'b:v': '2500k',            # Битрейт видео
            'maxrate': '3000k',
            'bufsize': '6000k',
            'c:a': 'aac',
            'b:a': '96k',              # Немного увеличили аудио для 720p
            'ar': '44100',
            'ac': '2',                 # Стерео звук
            'f': 'mp4',
            'vf': 'scale=1280:-2',     # 720p (16:9)
            'fps_mode': 'cfr',         # Фиксированный FPS
            'r': 30,                   # 30 кадров/сек
            'movflags': '+faststart',
            'pix_fmt': 'yuv420p',
            'y': None
        }

        # Создаем базовый FFmpeg процесс
        input_video = ffmpeg.input(str(concat_list), format='concat', safe=0)
        audio_stream = input_video.audio

         # Обработка музыки
        if music_config:
            music_path = Path(music_config.get("path", "./music/prokopenko.mp3"))
            volume = music_config.get("volume", 0.3)
            
            if music_path.exists():
                try:
                    music = (
                        ffmpeg.input(str(music_path))
                        .filter('atrim', duration=video_duration)
                        .filter('asetpts', 'PTS-STARTPTS')
                        .filter('volume', volume)
                    )
                    audio_stream = ffmpeg.filter([audio_stream, music], 'amix', duration='first')
                except ffmpeg.Error as e:
                    console_msg(f"Ошибка обработки музыки: {e.stderr.decode()}", "yellow")
                    audio_stream = input_video.audio
            else:
                console_msg(f"Файл музыки {music_path} не найден", "yellow")

        # Собираем финальный процесс
        log_file = "./ffmpeg.log"
        try:
            (
                ffmpeg
                .output(input_video.video, audio_stream, str(final_path), **concat_params)
                .overwrite_output()
                .run(
                    cmd=FFMPEG_PATH,
                    capture_stdout=True,
                    capture_stderr=True
                )
            )
        except ffmpeg.Error as e:
            error_msg = f"FFmpeg error: {e.stderr.decode().strip()}"
            console_msg(f"Детали в логе: {log_file}", "red")
            raise RuntimeError(error_msg)

        if not final_path.exists():
            raise FileNotFoundError(f"Файл результата {final_path} не создан")

        # console_msg(f"Видео сохранено: {final_path}", "green")
        return final_path

    except Exception as e:
        console_msg(f"Полная ошибка: {traceback.format_exc()}", "red")
        raise RuntimeError(f"Ошибка рендеринга: {str(e)}")

async def run_ffmpeg(inputs: dict, outputs: dict):
    """Универсальный запуск FFmpeg."""
    command = ['ffmpeg']
    
    # Добавляем входные файлы
    for input_file, options in inputs.items():
        command += ['-i', input_file]
        if options:
            command += options.split()
    
    # Добавляем выходные параметры
    for output_file, params in outputs.items():
        command += ['-map', '0']  # Используем первый входной поток
        for key, value in params.items():
            if value is None:
                command += [f'-{key}']
            else:
                command += [f'-{key}', str(value)]
        command.append(output_file)
    
    # Запускаем процесс
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    # Ждем завершения
    stdout, stderr = await proc.communicate()
    
    if proc.returncode != 0:
        error_msg = stderr.decode(errors='replace')
        raise RuntimeError(f"FFmpeg error: {error_msg}")
        

# Асинхронный цикл обработки очереди
async def task_processor():
    spinner = ["|", "/", "-", "\\"]
    spin_index = 0
    while True:
        # Анимация в одной строке
        print(f"\rПроверяю очередь задач... {spinner[spin_index]}", end="")
        spin_index = (spin_index + 1) % 4
        await asyncio.sleep(0.1)  # Скорость анимации
        
        task = await get_next_task()
        if task:
            print("\r" + " " * 80, end="")  # Очистка строки
            console_msg(f"\nНайдена задача: {task}")  # Лог задачи
            await asyncio.create_task(process_task(task))
        else:
            await asyncio.sleep(0.1)  # Чтобы общий цикл был 1 секунда


async def get_active_stream_id():
    try:
        youtube = build('youtube', 'v3', 
                      developerKey=YOUTUBE_API_KEY,
                      cache_discovery=False)
        
        # Объединяем запросы для экономии квоты
        response = youtube.search().list(
            part="id",
            channelId=CHANNEL_ID,
            type="video",
            order="date",
            maxResults=5  # Проверяем сразу несколько видео
        ).execute()
        
        # Вручную ищем live-трансляции среди результатов
        for item in response.get('items', []):
            video_id = item['id']['videoId']
            video_response = youtube.videos().list(
                part="liveStreamingDetails",
                id=video_id
            ).execute()
            
            if video_response.get('items', [{}])[0].get('liveStreamingDetails'):
                return video_id
                
        return None
        
    except Exception as e:
        print(f"\nAPI Error: {str(e)}")
        return None

# Чтение комментариев и добавление в очередь
async def chat_reader():
    spinner = ["|", "/", "-", "\\"]
    spin_index = 0
    while True:
        try:
            current_stream_id = STREAM_ID#input("current_stream_id")#await get_active_stream_id()
            if not current_stream_id:
                # print("\rНе найдено активных трансляций", end="")
                await asyncio.sleep(1)
                continue
            chat = pytchat.create(video_id=current_stream_id)
            while chat.is_alive():
                if not current_stream_id:
                    print(f"\rНе найдено активных трансляций..ждём стрим..{spinner[spin_index]}", end="")
                else:
                    print(f"\rОжидание сообщений... {spinner[spin_index]}", end="")
                    spin_index = (spin_index + 1) % 4
                    await asyncio.sleep(0.3)
                
                async for c in chat.get().async_items():
                    print("\r" + " " * 80, end="")  # Очистка строки
                    # console_msg(f"[{c.author.name}]: {c.message}")
                    
                    if c.message.startswith("!") and ' ' in c.message:
                        command_part, topic_text = c.message.split(' ', 1)
                        character = command_part[1:].lower().strip()
                        config_path = f"characters/{character}.json"
                        if not os.path.exists(config_path):
                            console_msg("Ошибка: Нет конфига персонажа", "yellow")
                            continue
                        topic = topic_text.strip()
                        if not character or not topic:
                            console_msg("Некорректная команда", "yellow")
                            continue
                        await add_task_to_db(c.author.name, character, topic, 0)
        except Exception as e:
            console_msg(f"Ошибка чтения чата: {e}", "red")
            await asyncio.sleep(1)


async def on_donation(data):
    try:
        # Проверка актуальности токена
        if datetime.now() > (TOKEN_EXPIRY - timedelta(minutes=5)):
            await refresh_access_token()
        
        # alert = data.get('data', data)  # Адаптация под структуру Centrifugo
        # donation_id = alert.get('id', str(uuid.uuid4()))
        
        #Проверка на дубликаты
        # if await is_donation_processed(donation_id):
            # return
        print("ПОЛНЫЕ ДАННЫЕ СООБЩЕНИЯ:", data)
        alert_data = data['result']['data']['data']
        username = alert_data.get('username', 'Не указано')
        message_text = alert_data.get('message', '')  # Правильное имя переменной
        amount = alert_data.get('amount', 0)
        currency = alert_data.get('currency', 'USD')
        
        # Определение уровня доната
        donation_level = 0
        if amount >= 100:
            donation_level = 1 if amount <= 500 else 2
        
        # Обработка команд
        if message_text.startswith("!"):  # Ошибка здесь: message не определена
            parts = message_text[1:].split(' ', 1)
            command = parts[0].lower()
            topic = parts[1].strip() if len(parts) > 1 else ""
            if command and topic:
                await add_task_to_db(username, command, topic, donation_level)
                # await mark_donation_as_processed(donation_id)
        else:
            console_msg(f"Получен донат: {username} - {amount} RUB", "green")
    except Exception as e:
        console_msg(f"Ошибка обработки доната: {str(e)}", "red")



async def main():
    global log_file
    try:
        # Инициализация лога
        log_file = init_log()  # Теперь log_file доступна глобально
        if not log_file:
            raise Exception("Не удалось инициализировать лог")
        # Проверка подключения к MySQL
        test_mysql_connection()
        
        init_db()
        await reset_all_errors()
        
        
        await get_initial_tokens()
        
        if datetime.now() > TOKEN_EXPIRY:
            await refresh_access_token()
        workers = [task_processor() for _ in range(2)]
        # Запуск асинхронных задач
        await asyncio.gather(
            # chat_reader(),
            connect_to_centrifugo(),
            token_refresher(),
            # task_processor(),  # Ваша задача (например, обработка заданий)
            *workers,
            cleanup_processed_donations()
        )
    except Exception as e:
        logger.error(f"Ошибка в основном цикле: {str(e)}")
    finally:
        # Закрытие файла лога
        if log_file:
            log_file.close()


if __name__ == "__main__":
    asyncio.run(main())
    