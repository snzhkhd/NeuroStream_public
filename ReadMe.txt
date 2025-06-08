устанавливаем fish-speech - https://t.me/neuroportchat/2010/10099
из папки __FSTSS_API - api.py переместить в корень fish_speech

Нужно создать локальный веб-сервер, я использовал - open server panel (https://ospanel.io/)
установленные модули - MySQL 8, PHP 8.1, Nginx-1.26
импортировать базы из папки __DB в свой локальный сервер

Для работы видео монтажа нужно скачать и распаковать в корневую папку ffmpeg - https://drive.google.com/file/d/1NpjRyzXJxgsA9ElYj_abnRwuQNFIULP2/view
если не хотите качать файлы от сюда то можно сказать целиком(вместе с ffmpeg) с гугла - https://drive.google.com/file/d/16XpBgqgJAZDki0RB24BuiFrJ8HAtDnDt/view?usp=sharing
(fish-speech не включён в архивы)
Мои медиа файлы (видео и аудио для персонажей) - https://drive.google.com/file/d/1DgKXN9q3XU-mJacIsTB3FEzHSDD82Mrm/view?usp=sharing
запуск 
0_run tts api	- API fish-speech, 0_0_run api 	для запуска второго API. в данном случае я просто скопировал api.py, поменял порт и переименовал в api2.py. можно дописать код для автоматизации этого процесса при желании
1_run player	- плеер для видео, берёт из базы и выводит в html.  нужно отредактировать и настроить mysql_config под себя
2_run_html	- откой блокнотом и поменяй путь на свой
3_run		- основной файл обрабьотки видео\аудио\донаты и тд. нужна указать свои настройки для donationalerts(опционально), нужно указать индификатор стрима - STREAM_ID. (либо дописать код для YOUTUBE_API_KEY и использовать его). указать порты TTS_SERVERS или отавить как есть. продублировать настройки mysql_config из player 


настройка OBS - смотрите OBS.png - просто указать путь к файлу player.html.
можно запускать без стрима

папки
characters - конфиги персонажей, сделал как смог, работает). можно указывать музыку фона, клипы, аудио референс для fish-speech и промпт
static - через каждые 3 видео воспроизведится disclaimer из этой папки.  папка fallback_videos - тут лежат видео которые будет возпроизводится когда нет очереди. можно ложить туда что угодно

donationalerts_help_api - я использовал эти скрипты для получения ключей с donationalerts

все ключи хранятся в базе



main.py 
функция get_llm_response_async
тут можно указать ллм для запросов   - model="deepseek/deepseek-chat-v3-0324:free:nitro",  #"deepseek/deepseek-chat:free:nitro",       "deepseek/deepseek-chat-v3-0324:free"
так же нужно указать свой ключ в async_client от openrouter
