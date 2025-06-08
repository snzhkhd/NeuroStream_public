from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector
from mysql.connector import Error
from fastapi import Response  # Для возврата статуса 204
import glob
import random
import os

mysql_config = {
    "host": "mysql-8.0",          # Например: "localhost" или "127.0.0.1"
    "user": "root",    # Например: "root"
    "password": "",
    "database": "my_database",    # Имя вашей базы из URL
    "port": 3306,                 # Порт MySQL (стандартный 3306)
    "charset": "utf8mb4"
}

app = FastAPI()

app.mount("/templates", StaticFiles(directory="templates"), name="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/output_videos", StaticFiles(directory="output_videos"), name="output_videos")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def get_queue_status():
    try:
        conn = mysql.connector.connect(**mysql_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN donation_level = 2 THEN 1 ELSE 0 END) as super_donors,
                SUM(CASE WHEN donation_level = 1 THEN 1 ELSE 0 END) as donors,
                SUM(CASE WHEN donation_level = 0 THEN 1 ELSE 0 END) as regular
            FROM tasks 
            WHERE status = 'queued'
        ''')
        return cursor.fetchone()
    except Error as e:
        console_msg(f"Ошибка получения статуса очереди: {e}")
        return None
    finally:
        cursor.close()
        conn.close()
        
@app.get("/api/queue")
async def get_queue():
    try:
        conn = mysql.connector.connect(**mysql_config)
        cursor = conn.cursor(dictionary=True)
        
        # Получаем все статусы
        cursor.execute('''
            SELECT 
                SUM(status = 'queued') as total,
                SUM(status = 'processing') as processing,
                SUM(status = 'completed') as completed
            FROM tasks
        ''')
        status_counts = cursor.fetchone()
        
       # Получаем задачи в очереди
        cursor.execute('''
            SELECT id, author, 
                   SUBSTRING(topic, 1, 20) as short_topic,
                   donation_level,
                   status
            FROM tasks 
            ORDER BY 
                CASE status
                    WHEN 'processing' THEN 1
                    WHEN 'queued' THEN 2
                    ELSE 3
                END,
                priority DESC, 
                id ASC
            LIMIT 20
        ''')
        queue = cursor.fetchall()
        
        return {
            "status": status_counts,
            "queue": queue
        }
    except Error as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/next_video")
async def get_next_video():
    try:
        conn = mysql.connector.connect(**mysql_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute('''
            SELECT * FROM tasks 
            WHERE status = 'completed'
            ORDER BY processed_at ASC
            LIMIT 1
        ''')
        video = cursor.fetchone()
        if video:
            # Помечаем как воспроизводимое
            cursor.execute('''
                UPDATE tasks 
                SET status = 'playing' 
                WHERE id = %s
            ''', (video['id'],))
            conn.commit()
        # 2. Если нет - берём случайное фоновое видео
        if not video:
            fallback_videos = glob.glob("static/fallback_videos/*.mp4")
            if not fallback_videos:
                return Response(status_code=204)
            
            return {
                "id": -1,  # Специальный ID для фоновых видео
                "author": "NeuroStream",
                "topic": "Развлекательный контент",
                "video_path": "/" + random.choice(fallback_videos)
            }
        print(video['video_path'])
        video['video_path'] = f"/output_videos/{os.path.basename(video['video_path'])}"
        if not os.path.exists(video['video_path'].lstrip('/')):
                raise HTTPException(404, detail="Video file not found")
        return video
    except Error as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/video_played/{video_id}")
async def video_played(video_id: int):
    try:
        conn = mysql.connector.connect(**mysql_config)
        cursor = conn.cursor(dictionary=True)  # ← ДОБАВЬТЕ ЭТО!
        
        # Проверяем существование задачи
        cursor.execute("SELECT * FROM tasks WHERE id = %s", (video_id,))
        task = cursor.fetchone()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Переносим в историю (без указания id)   task['id'], id, 
        cursor.execute('''
            INSERT INTO played_videos 
                (author, character_name, topic, video_path)
            VALUES (%s, %s, %s, %s)
        ''', (
            task['author'],
            task['character_name'],
            task['topic'],
            task['video_path']
        ))

        # Удаляем из задач
        cursor.execute('DELETE FROM tasks WHERE id = %s', (video_id,))
        
        conn.commit()
        return {"status": "success"}
        
    except mysql.connector.Error as e:
        print(f"MySQL Error: {e}")
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        print(f"General Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)