FROM python:3.12.11-slim-bookworm

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "app.py"]
