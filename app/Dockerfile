# Этап 1: Установка зависимостей
FROM python:3.11 AS builder
ENV PYTHONPATH=/usr/src
WORKDIR /usr/src/app
COPY requirements.txt /usr/src/app
RUN pip install -r requirements.txt

# Этап 2: Копирование остальных файлов и запуск приложения
FROM python:3.11
ENV PYTHONPATH=/usr/src
WORKDIR /usr/src/app
COPY --from=builder /usr/src/app /usr/src/app
COPY . /usr/src/app
#CMD python3 start_server.py
CMD uvicorn app.start_server:app --host 0.0.0.0 --port 80 --workers 1
