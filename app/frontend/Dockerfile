FROM python:3.7-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["wait-for-it", "--service", "backend:8080", "--", "streamlit", "run", "frontend.py"]
