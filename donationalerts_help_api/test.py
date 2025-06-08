import asyncio
import websockets
import json
import requests

# Твои данные
ACCESS_TOKEN = ""
USER_ID = ""

async def main():
    # Получаем socket_connection_token
    user_info_url = "https://www.donationalerts.com/api/v1/user/oauth"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    user_response = requests.get(user_info_url, headers=headers)
    
    if user_response.status_code != 200:
        print("Ошибка:", user_response.json())
        return
    
    user_data = user_response.json()["data"]
    socket_token = user_data["socket_connection_token"]

    # Подключаемся к Centrifugo
    uri = "wss://centrifugo.donationalerts.com/connection/websocket"
    async with websockets.connect(uri) as ws:
        # Авторизация в Centrifugo
        auth_msg = {
            "params": {"token": socket_token},
            "id": 1
        }
        await ws.send(json.dumps(auth_msg))
        auth_response = await ws.recv()
        auth_data = json.loads(auth_response)
        client_id = auth_data['result']['client']
        
        # Подписка на канал
        channel = f"$alerts:donation_{USER_ID}"
        subscribe_url = "https://www.donationalerts.com/api/v1/centrifuge/subscribe"
        data = {
            "channels": [channel],
            "client": client_id
        }
        res = requests.post(subscribe_url, json=data, headers=headers)
        
        if res.status_code != 200:
            print("Ошибка подписки:", res.json())
            return

        subscription_token = res.json()['channels'][0]['token']

        # Подключаемся к каналу
        subscribe_msg = {
            "params": {
                "channel": channel,
                "token": subscription_token
            },
            "method": 1,
            "id": 2
        }
        await ws.send(json.dumps(subscribe_msg))

        # Получаем уведомления
        while True:
            try:
                message = await ws.recv()
                data = json.loads(message)
                # print("ПОЛНЫЕ ДАННЫЕ СООБЩЕНИЯ:", data)  # Выводим все данные
                
                # Проверяем, что это реальный донат
                if (
                    'result' in data and
                    'data' in data['result'] and
                    'data' in data['result']['data'] and  # Вложенность данных
                    'username' in data['result']['data']['data']  # Проверка ключа username
                ):
                    alert_data = data['result']['data']['data']  # Данные доната
                    
                    # Извлекаем необходимые поля
                    username = alert_data.get('username', 'Не указано')
                    message_text = alert_data.get('message', '')
                    amount = alert_data.get('amount', 0)
                    currency = alert_data.get('currency', 'USD')
                    
                    print(f"Имя: {username}")
                    print(f"Сообщение: {message_text}")
                    print(f"Сумма: {amount} {currency}")
                    # print("-" * 30)
                
            except Exception as e:
                print(f"Ошибка: {e}")

if __name__ == "__main__":
    asyncio.run(main())