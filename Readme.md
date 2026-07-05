# AI-Based ID Recognition System

An AI-powered identity document recognition system that extracts structured information from ID cards, performs face verification, and supports liveness detection to improve identity validation accuracy.

---

## Features

- Identity document processing
- OCR-based data extraction
- Face detection and face matching
- Liveness detection
- Structured data extraction
- FastAPI backend
- Database integration
- REST API support

---

## Tech Stack

- Python
- FastAPI
- OpenCV
- OCR
- Face Recognition
- SQLAlchemy
- PostgreSQL / SQLite
- Mistral AI
- Supabase

---

## Project Structure

```text
.
├── backend/
│   ├── app.py
│   ├── extractor.py
│   ├── face_utils.py
│   ├── face_matcher_improved.py
│   ├── liveness.py
│   ├── webcam_capturer.py
│   ├── database.py
│   ├── schemas.py
│   ├── worker.py
│   ├── requirements.txt
│   └── .env
│
├── templates/
│   └── landing.html
│
├── .gitignore
└── README.md
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/VanshRa-pixel/ID-Recognition.git
cd ID-Recognition
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows**

```bash
.venv\Scripts\activate
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r backend/requirements.txt
```

### Configure Environment Variables

Create a `.env` file inside the `backend` folder.

```env
MISTRAL_API_KEY=YOUR_API_KEY
SUPABASE_URL=YOUR_SUPABASE_URL
SUPABASE_KEY=YOUR_SUPABASE_KEY
```

### Run the application

```bash
uvicorn backend.app:app --reload
```

---

## API Workflow

1. Upload an identity document.
2. Extract text using OCR.
3. Detect and crop the face.
4. Perform face matching.
5. Run liveness detection.
6. Store extracted information in the database.
7. Return structured JSON response.

---

## Future Improvements

- Aadhaar QR code parsing
- Passport support
- Driving License support
- Multi-language OCR
- Docker deployment
- Cloud deployment
- Authentication and authorization

---

## Author

**Vansh Rajput**

GitHub: https://github.com/VanshRa-pixel
