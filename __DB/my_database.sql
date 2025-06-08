-- phpMyAdmin SQL Dump
-- version 5.2.2
-- https://www.phpmyadmin.net/
--
-- Хост: mysql-8.0
-- Время создания: Июн 08 2025 г., 15:34
-- Версия сервера: 8.0.35
-- Версия PHP: 8.1.28

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- База данных: `my_database`
--

-- --------------------------------------------------------

--
-- Структура таблицы `auth_tokens`
--

CREATE TABLE `auth_tokens` (
  `id` int NOT NULL,
  `access_token` text,
  `refresh_token` text,
  `expiry_time` datetime NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Дамп данных таблицы `auth_tokens`
--

INSERT INTO `auth_tokens` (`id`, `access_token`, `refresh_token`, `expiry_time`) VALUES
(1, NULL, NULL, '2026-04-01 02:14:55');

-- --------------------------------------------------------

--
-- Структура таблицы `played_videos`
--

CREATE TABLE `played_videos` (
  `id` int NOT NULL,
  `author` varchar(255) NOT NULL,
  `character_name` varchar(255) NOT NULL,
  `topic` text NOT NULL,
  `video_path` text,
  `played_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Структура таблицы `processed_donations`
--

CREATE TABLE `processed_donations` (
  `donation_id` varchar(255) NOT NULL,
  `processed_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Структура таблицы `tasks`
--

CREATE TABLE `tasks` (
  `id` int NOT NULL,
  `author` varchar(255) NOT NULL,
  `character_name` varchar(255) NOT NULL,
  `topic` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `status` enum('queued','processing','completed','error','playing') CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT 'queued',
  `video_path` text,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `processed_at` timestamp NULL DEFAULT NULL,
  `priority` int NOT NULL,
  `donation_level` int NOT NULL,
  `dialogue` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Дамп данных таблицы `tasks`
--

INSERT INTO `tasks` (`id`, `author`, `character_name`, `topic`, `status`, `video_path`, `created_at`, `processed_at`, `priority`, `donation_level`, `dialogue`) VALUES
(777, 'snzhkhd', 'cap', 'Лекция о сервер-клиент приложениях на примере создания мода на кооператив для игры Gothic ', 'completed', 'output_videos\\777_20250415_134004.mp4', '2025-04-15 10:33:53', '2025-04-15 10:40:13', 0, 0, 'Капитан: Так… ну я тебе щас лекцию прочитаю.  \nПахом: Хорошо.  \nКапитан: Сиди вон туда, блядь, садись туда, сука, вот сюда вон, блядь.  \nКапитан: Молчи, блядь, пасть свою заткни…  \nПахом: А?  \nКапитан: Говорю, пасть свою заткни!  \nКапитан: Так… ну я тебе щас лекцию прочитаю.  \nКапитан: Значит, сервер-клиент приложения, перед тем как ты, блядь, сдохнешь от своей тупости, задумали расхуячить кооператив в Gothic, то, что потом вошло в историю, как мод на четверых.  \nКапитан: Слушай и запоминай.  \nКапитан: Командующим сервером был движок, блядь, старый, но исполнительный… исполнительный, безусловно, профессионал.  \nКапитан: Но без фантазии, у разработчиков вообще людей с фантазиями было немного.  \nКапитан: Дерьмо на палочке, ничего, блядь, не знаешь, ничего не можешь.  \nКапитан: Чё ты вообще, блядь, в моддинге делаешь?  \nПахом: Я?  \nКапитан: МОЛЧАТЬ! Какие пакеты данных летят между клиентами?  \nПахом: А?  \nКапитан: Не «А». Какие пакеты, сука?  \nКапитан: Какой самый известный протокол для синхронизации?  \nПахом: TCP!  \nКапитан: Идиот, блядь. СКОЛЬКО БАЙТ, СУКА?  \nКапитан: СКОЛЬКО, блядь, БАЙТ, СКОТИНА, блядь?  \nКапитан: 7 декабря 2001 года движок Gothic в составе 4 клиентов: первый, второй, третий и четвертый — появились на траверсе у Хориниса.  \nКапитан: Первое ударное соединение насчитывало 50 пакетов в секунду, 40 запросов и 81 ответ.  \nКапитан: В итоге этого дерьма 4 игрока получили лаги. Какие лаги? КАКИЕ ЛАГИ?  \nКапитан: Десинхрон, вылет, зависание и отвал.  \nКапитан: ЭТО ЗНАТЬ НАДО, если ты учился в шестом училище.  \nКапитан: ЭТО КЛАССИКА, БЛЯДЬ! СКОЛЬКО БАЙТ, СУКА? СКОЛЬКО, блядь, БАЙТ, СКОТИНА, блядь?  \nКапитан: Сейчас наши моддеры ориентируются именно на этих долбоебов.  \nКапитан: По крайней мере эти те немногие, кто выебли лаги в жопу.  \nКапитан: Это знать надо, дерьмо собачье.  \nКапитан: А, блядь, НАХУЙ!  \nКапитан: НУ ИДИ СЮДА, СУКА, блядь, ДЕРЬМО СОБАЧЬЕ, блядь, а, блядь.  \nКапитан: Так, ну щас чай принесут, мы с тобой продолжим, продолжим, продолжим…  \nКапитан: Ты хоть и полный идиот, но… Я думаю, что тебе эта информация будет полезна, по крайней мере в ближайший час.');

--
-- Индексы сохранённых таблиц
--

--
-- Индексы таблицы `auth_tokens`
--
ALTER TABLE `auth_tokens`
  ADD PRIMARY KEY (`id`);

--
-- Индексы таблицы `played_videos`
--
ALTER TABLE `played_videos`
  ADD PRIMARY KEY (`id`);

--
-- Индексы таблицы `processed_donations`
--
ALTER TABLE `processed_donations`
  ADD PRIMARY KEY (`donation_id`);

--
-- Индексы таблицы `tasks`
--
ALTER TABLE `tasks`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT для сохранённых таблиц
--

--
-- AUTO_INCREMENT для таблицы `auth_tokens`
--
ALTER TABLE `auth_tokens`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT для таблицы `played_videos`
--
ALTER TABLE `played_videos`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=196;

--
-- AUTO_INCREMENT для таблицы `tasks`
--
ALTER TABLE `tasks`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=778;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
