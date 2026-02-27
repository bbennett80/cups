# fastcups

FastAPI + Socket.IO classroom feedback app.

Teachers create unique class URLs, students select a status color (green/yellow/red), and teacher views update live.

## Current functionality

- Teacher authentication:
  - Sign up, sign in, and sign out
  - Session cookie-based auth
- Teacher dashboard:
  - Create class URLs
  - Class URLs are globally unique across all teachers
  - List your classes with quick links
- Teacher classroom view:
  - Real-time updates (no polling reload) for:
    - Active student count
    - Connected student count
    - Green/yellow/red distribution
- Student view:
  - One-click green/yellow/red status update
  - Single active-tab behavior per student cookie
- Sharing:
  - Dashboard `Student Link` opens a modal with a QR code for the class URL
- UI:
  - Global light/dark mode toggle (persisted in browser local storage)

## Tech stack

- FastAPI
- python-socketio (ASGI server mode)
- Jinja2 templates
- SQLite (teacher accounts, sessions, classes)

## Requirements

- Python 3.12+ recommended

Dependencies are pinned in `requirements.txt`.

## Setup

### Option A (recommended): `uv`

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Run:

```bash
uv run --python .venv/bin/python uvicorn main:app --reload
```

### Option B: standard venv + pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## App routes

Public:

- `/` home / usage overview
- `/{class_id}` student class page

Teacher auth:

- `/teacher/signup` (GET/POST)
- `/teacher/signin` (GET/POST)
- `/teacher/logout` (POST)

Teacher app:

- `/teacher/dashboard` teacher dashboard
- `/teacher/classes` create class (POST)
- `/{class_id}/teacher` teacher live view (owner only)

## Notes

- The app creates `teachers.db` in the project root on first run.
- Teacher pages are protected and class ownership is enforced.
- Existing in-memory student state resets when the process restarts.
