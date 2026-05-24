# Лабораторная работа N4

## Тема

Интеграция Apache Kafka сервиса с сервисом модели BBC News Classifier.

## Цель

Получить навыки реализации Kafka Producer и Kafka Consumer и интегрировать их с сервисом модели, контейнерной инфраструктурой, защищённым хранилищем секретов и CI/CD pipeline.

## Архитектура

Реализован поток обработки:

```text
HTTP client -> bbc-news-api -> Kafka Producer -> prediction-results topic -> Kafka Consumer -> ClickHouse
```

- `bbc-news-api` принимает `/predict`, выполняет инференс и публикует результат в Kafka.
- `kafka` работает как брокер сообщений.
- `kafka-consumer` читает topic `prediction-results` и сохраняет записи в ClickHouse.
- `clickhouse` хранит историю предсказаний и загруженные train/test данные.
- `vault-bootstrap` расшифровывает Ansible Vault и готовит пользователя ClickHouse.

## Kafka Producer

Producer реализован в `src/bbc_news/kafka.py`.

Формат сообщения:

```json
{
  "request_id": "uuid",
  "row_position": 0,
  "input_text": "Stock market gained today",
  "predicted_label": "business",
  "created_at": "2026-05-24T12:00:00+00:00",
  "model_version": "lab4"
}
```

Producer включается переменной `KAFKA_ENABLED=true`. Если Kafka выключена, API сохраняет результат напрямую в ClickHouse, что сохраняет совместимость локальных unit-тестов.

## Kafka Consumer

Consumer реализован в `src/bbc_news/consumer.py` и запускается отдельным контейнером:

```bash
python -m bbc_news.consumer
```

Consumer:

- подключается к `KAFKA_BOOTSTRAP_SERVERS`;
- читает topic `prediction-results`;
- валидирует JSON payload;
- преобразует сообщение в `PredictionLogRecord`;
- сохраняет запись в ClickHouse.

## Интеграция с секретами

ClickHouse credentials не хранятся в коде. Consumer использует существующий механизм `Ansible Vault`:

- зашифрованный файл: `secrets/clickhouse.vault.yml`;
- пароль: `secrets/.vault_pass.txt`, в контейнерах монтируется как `/run/secrets/.vault_pass.txt`;
- чтение секретов: `src/bbc_news/secrets.py`;
- подключение к ClickHouse: `src/bbc_news/storage.py`.

## Docker Compose

Обязательное использование `docker-compose` выполнено в `docker-compose.yml`.

Сервисы:

- `vault-bootstrap`;
- `clickhouse`;
- `kafka`;
- `kafka-init`;
- `bbc-news-api`;
- `kafka-consumer`.

Команды проверки:

```bash
docker compose up -d --build
docker compose exec -T bbc-news-api python scripts/load_clickhouse_data.py --config /app/config.ini
docker compose exec -T bbc-news-api python scripts/run_scenario.py --scenario /app/scenario.json --base-url http://127.0.0.1:8000
docker compose down
```

## CI/CD

CI pipeline: `.github/workflows/ci.yml`.

CI выполняет:

- установку зависимостей;
- обучение модели;
- запуск тестов с coverage;
- сборку Docker image;
- push в DockerHub при наличии `DOCKERHUB_USERNAME` и `DOCKERHUB_TOKEN`;
- подпись image через `cosign`;
- генерацию `dev_sec_ops.yml`.

CD pipeline: `.github/workflows/cd.yml`.

CD выполняет:

- запуск по `workflow_dispatch`, расписанию или после успешного CI;
- подготовку Vault password file из GitHub Secret `ANSIBLE_VAULT_PASSWORD`;
- запуск `docker compose`;
- загрузку train/test данных в ClickHouse;
- функциональный сценарий `scenario.json`.

## Функциональное тестирование

Сценарий `scenario.json` проверяет:

1. `/health` возвращает работоспособный API и ClickHouse.
2. `/predict` возвращает два предсказания.
3. `/predictions?limit=2` возвращает две записи, сохранённые consumer-ом после получения Kafka-сообщений.

Скрипт `scripts/run_scenario.py` поддерживает повторные проверки, чтобы дождаться асинхронной записи consumer-а в ClickHouse.

## Результаты

- GitHub: https://github.com/KrasilnikovAV/big_data_lab4
- DockerHub: https://hub.docker.com/r/kabeton2/bbc-news-classifier_v4
- Актуальный дистрибутив: `dist/bbc-news-classifier-lab4.zip`
