FROM python:3.7-slim

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN pip install Flask gunicorn flask_wtf transformers sentence_transformers learn google.cloud.logging

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
