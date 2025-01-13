FROM python:3.10-alpine

WORKDIR /app

COPY . /app/

RUN apk add libsndfile libsndfile-dev
RUN apk add espeak-ng
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["uvicorn", "api:app","--port","8080","--host","0.0.0.0"]