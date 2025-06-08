import requests
from datetime import datetime, timedelta
# Твои данные
CLIENT_ID = ""
CLIENT_SECRET = ""
REDIRECT_URI = "http://localhost/callback"
SCOPE = "oauth-donation-subscribe oauth-user-show"  # Скобок нет, пробел между значениями

# 1. Формируем URL для авторизации
auth_url = f"https://www.donationalerts.com/oauth/authorize?" \
           f"client_id={CLIENT_ID}&" \
           f"redirect_uri={REDIRECT_URI}&" \
           f"response_type=code&" \
           f"scope={SCOPE}"

print("Открой эту ссылку в браузере и авторизуйся:")
print(auth_url)

# 2. После авторизации перенаправит на REDIRECT_URI с параметром `code`
code = input("Скопируй код из URL и вставь сюда: ")

# 3. Обмен кода на access_token
token_url = "https://www.donationalerts.com/oauth/token"
data = {
    "grant_type": "authorization_code",
    "code": code,
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "redirect_uri": REDIRECT_URI
}

response = requests.post(token_url, data=data)
tokens = response.json()
ACCESS_TOKEN = tokens["access_token"]
REFRESH_TOKEN = tokens["refresh_token"]
expires = tokens["expires_in"]
print("Access Token:", ACCESS_TOKEN)
print("Refresh Token:", REFRESH_TOKEN)

expires_in = int(tokens["expires_in"])  # Преобразуем в целое число
expiry_time = datetime.now() + timedelta(seconds=expires_in)
print("expiry_time:", expiry_time)

# Используем access_token из предыдущего шага
user_info_url = "https://www.donationalerts.com/api/v1/user/oauth"
headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

response = requests.get(user_info_url, headers=headers)
if response.status_code == 200:
    user_data = response.json()
    print("Ответ API:", user_data) 
else:
    print(f"Ошибка {response.status_code}: {response.text}")