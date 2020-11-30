FROM python:3.7-slim

RUN apt-get update && apt-get install -y curl 

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "backend.py"]
