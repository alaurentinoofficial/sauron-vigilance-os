FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn

COPY . .

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 
