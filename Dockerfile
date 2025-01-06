FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt && apt-get update && apt-get install -y fonts-freefont-ttf fonts-dejavu fonts-liberation

COPY . .

CMD ["/bin/bash", "-c", "while true; do sleep 1; done"]