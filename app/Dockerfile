FROM python:3.11
ENV PYTHONPATH=/usr/src
EXPOSE 80
WORKDIR /usr/src/app
COPY . /usr/src/app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt
CMD uvicorn app.start_server:app --host 0.0.0.0 --port 80 --workers 1