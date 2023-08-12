# Dockerfile
FROM python:3.9
ENV PYTHONUNBUFFERED True
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-sdk -y
RUN gcloud auth activate-service-account --key-file=gft-bike-insurance-329926313918.json

ENTRYPOINT ["python"]
CMD ["app.py"]
