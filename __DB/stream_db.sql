-- phpMyAdmin SQL Dump
-- version 5.2.2
-- https://www.phpmyadmin.net/
--
-- Хост: mysql-8.0
-- Время создания: Июн 08 2025 г., 15:35
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
-- База данных: `stream_db`
--

-- --------------------------------------------------------

--
-- Структура таблицы `topics_current`
--

CREATE TABLE `topics_current` (
  `id` int NOT NULL,
  `date` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `type` varchar(64) COLLATE utf8mb4_general_ci NOT NULL,
  `speaker` varchar(64) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `priority` int NOT NULL DEFAULT '0',
  `source` varchar(128) COLLATE utf8mb4_general_ci NOT NULL,
  `requestor_id` varchar(512) COLLATE utf8mb4_general_ci NOT NULL,
  `user_id` varchar(1024) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `topic` varchar(2048) COLLATE utf8mb4_general_ci NOT NULL,
  `topic_original` varchar(2048) COLLATE utf8mb4_general_ci NOT NULL,
  `characters` text COLLATE utf8mb4_general_ci NOT NULL,
  `scenario` text COLLATE utf8mb4_general_ci NOT NULL,
  `npc` text COLLATE utf8mb4_general_ci
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Структура таблицы `topics_generated`
--

CREATE TABLE `topics_generated` (
  `id` bigint NOT NULL,
  `date` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `type` varchar(64) COLLATE utf8mb4_general_ci NOT NULL,
  `speaker` varchar(64) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `priority` int NOT NULL DEFAULT '0',
  `source` varchar(128) COLLATE utf8mb4_general_ci NOT NULL,
  `requestor_id` varchar(512) COLLATE utf8mb4_general_ci NOT NULL,
  `user_id` varchar(1024) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `topic` varchar(2048) COLLATE utf8mb4_general_ci NOT NULL,
  `topic_original` varchar(2048) COLLATE utf8mb4_general_ci NOT NULL,
  `characters` text COLLATE utf8mb4_general_ci NOT NULL,
  `scenario` text COLLATE utf8mb4_general_ci NOT NULL,
  `npc` text COLLATE utf8mb4_general_ci
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Структура таблицы `topics_suggested`
--

CREATE TABLE `topics_suggested` (
  `id` bigint NOT NULL,
  `date` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `type` varchar(64) COLLATE utf8mb4_general_ci NOT NULL,
  `speaker` varchar(64) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `url` varchar(2048) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `priority` int NOT NULL DEFAULT '0',
  `source` varchar(128) COLLATE utf8mb4_general_ci NOT NULL,
  `requestor_id` varchar(512) COLLATE utf8mb4_general_ci NOT NULL,
  `character` varchar(50) COLLATE utf8mb4_general_ci NOT NULL DEFAULT 'default',
  `user_id` varchar(1024) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `topic` varchar(2048) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `topic_original` varchar(2048) COLLATE utf8mb4_general_ci DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Индексы сохранённых таблиц
--

--
-- Индексы таблицы `topics_current`
--
ALTER TABLE `topics_current`
  ADD PRIMARY KEY (`id`);

--
-- Индексы таблицы `topics_generated`
--
ALTER TABLE `topics_generated`
  ADD PRIMARY KEY (`id`);

--
-- Индексы таблицы `topics_suggested`
--
ALTER TABLE `topics_suggested`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT для сохранённых таблиц
--

--
-- AUTO_INCREMENT для таблицы `topics_current`
--
ALTER TABLE `topics_current`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT для таблицы `topics_generated`
--
ALTER TABLE `topics_generated`
  MODIFY `id` bigint NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=85862;

--
-- AUTO_INCREMENT для таблицы `topics_suggested`
--
ALTER TABLE `topics_suggested`
  MODIFY `id` bigint NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=45;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
