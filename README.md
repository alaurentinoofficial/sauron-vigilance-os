# Sauron Vigilance OS

![screeshot](assets/screenshot.png)

## Docker Setup

### Run Everything with Docker Compose
To run both frontend and backend services:
```
docker-compose up
```

## Standard Setup

### Run Backend
```
cd back/

python3 -m pip install -r requirements.txt

uvicorn main:app --reload
```

### Run Frontend
```
cd front/

npm install

npm run dev
```
