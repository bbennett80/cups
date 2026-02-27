from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import socketio

from http.cookies import SimpleCookie
from urllib.parse import quote
from datetime import datetime, timedelta, timezone
import random, string, collections, time, secrets, hashlib, hmac, sqlite3, re

fastapi_app = FastAPI()
socket_server = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = socketio.ASGIApp(socket_server, fastapi_app)
fastapi_app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

sid2student = dict()
sid2class = dict()
sid2teacher = dict()
teacher_sid2class = dict()
student2color = dict()
class2students = collections.defaultdict(lambda: set())

DB_PATH = "teachers.db"
TEACHER_SESSION_COOKIE = "teacher_session"
SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 30
PBKDF2_ITERATIONS = 200_000
DASHBOARD_PATH = "/teacher/dashboard"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS teachers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS teacher_sessions (
                token TEXT PRIMARY KEY,
                teacher_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (teacher_id) REFERENCES teachers(id)
            );

            CREATE TABLE IF NOT EXISTS classes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                teacher_id INTEGER NOT NULL,
                class_name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                FOREIGN KEY (teacher_id) REFERENCES teachers(id)
            );
            """
        )


def normalize_email(email):
    return email.strip().lower()


def hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password, password_hash):
    try:
        algorithm, iterations, salt_hex, digest_hex = password_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
    except ValueError:
        return False

    computed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        int(iterations),
    )
    return hmac.compare_digest(computed.hex(), digest_hex)


def safe_next_path(next_path):
    if isinstance(next_path, str) and next_path.startswith("/") and not next_path.startswith("//"):
        return next_path
    return DASHBOARD_PATH


def create_teacher_session(teacher_id):
    token = secrets.token_urlsafe(32)
    created_at = datetime.now(timezone.utc)
    expires_at = created_at + timedelta(seconds=SESSION_MAX_AGE_SECONDS)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO teacher_sessions (token, teacher_id, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (token, teacher_id, created_at.isoformat(), expires_at.isoformat()),
        )
    return token


def get_teacher_by_session_token(token):
    if not token:
        return None
    with get_db() as conn:
        return conn.execute(
            """
            SELECT t.id, t.email
            FROM teacher_sessions ts
            JOIN teachers t ON t.id = ts.teacher_id
            WHERE ts.token = ? AND ts.expires_at > ?
            """,
            (token, datetime.now(timezone.utc).isoformat()),
        ).fetchone()


def current_teacher(request: Request):
    return get_teacher_by_session_token(request.cookies.get(TEACHER_SESSION_COOKIE))


def template_context(request: Request, extra=None):
    context = {"request": request, "current_teacher": current_teacher(request)}
    if extra:
        context.update(extra)
    return context


def set_teacher_cookie(response: Response, token):
    response.set_cookie(
        key=TEACHER_SESSION_COOKIE,
        value=token,
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
    )


def clear_teacher_session(token):
    if not token:
        return
    with get_db() as conn:
        conn.execute("DELETE FROM teacher_sessions WHERE token = ?", (token,))


def normalize_class_name(value):
    return value.strip().lower()


def is_valid_class_name(value):
    return re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{1,38}[a-z0-9])?", value) is not None


def get_teacher_classes(teacher_id):
    with get_db() as conn:
        return conn.execute(
            """
            SELECT class_name, created_at
            FROM classes
            WHERE teacher_id = ?
            ORDER BY class_name ASC
            """,
            (teacher_id,),
        ).fetchall()


def teacher_owns_class(teacher_id, class_id):
    with get_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM classes WHERE teacher_id = ? AND class_name = ?",
            (teacher_id, class_id),
        ).fetchone()
    return row is not None


init_db()

@fastapi_app.get("/", response_class=HTMLResponse)
def root(request: Request):
    url = f"{request.base_url}"
    return templates.TemplateResponse('howto.html', template_context(request, {"url": url}))


@fastapi_app.get("/teacher/signup", response_class=HTMLResponse)
def teacher_signup_page(request: Request, next_path: str = DASHBOARD_PATH):
    next_path = safe_next_path(next_path)
    teacher = current_teacher(request)
    if teacher:
        return RedirectResponse(next_path, status_code=303)
    return templates.TemplateResponse(
        "teacher_signup.html",
        template_context(request, {"next_path": next_path}),
    )


@fastapi_app.post("/teacher/signup", response_class=HTMLResponse)
def teacher_signup_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(...),
    next_path: str = Form(DASHBOARD_PATH),
):
    next_path = safe_next_path(next_path)
    email = normalize_email(email)

    if not email or "@" not in email:
        return templates.TemplateResponse(
            "teacher_signup.html",
            template_context(request, {"error": "Enter a valid email.", "next_path": next_path}),
            status_code=400,
        )
    if len(password) < 8:
        return templates.TemplateResponse(
            "teacher_signup.html",
            template_context(request, {"error": "Password must be at least 8 characters.", "next_path": next_path}),
            status_code=400,
        )
    if password != password_confirm:
        return templates.TemplateResponse(
            "teacher_signup.html",
            template_context(request, {"error": "Passwords do not match.", "next_path": next_path}),
            status_code=400,
        )

    try:
        with get_db() as conn:
            cursor = conn.execute(
                """
                INSERT INTO teachers (email, password_hash, created_at)
                VALUES (?, ?, ?)
                """,
                (email, hash_password(password), datetime.now(timezone.utc).isoformat()),
            )
            teacher_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        return templates.TemplateResponse(
            "teacher_signup.html",
            template_context(request, {"error": "A teacher account with this email already exists.", "next_path": next_path}),
            status_code=400,
        )

    token = create_teacher_session(teacher_id)
    response = RedirectResponse(next_path, status_code=303)
    set_teacher_cookie(response, token)
    return response


@fastapi_app.get("/teacher/signin", response_class=HTMLResponse)
def teacher_signin_page(request: Request, next_path: str = DASHBOARD_PATH):
    next_path = safe_next_path(next_path)
    teacher = current_teacher(request)
    if teacher:
        return RedirectResponse(next_path, status_code=303)
    return templates.TemplateResponse(
        "teacher_signin.html",
        template_context(request, {"next_path": next_path}),
    )


@fastapi_app.post("/teacher/signin", response_class=HTMLResponse)
def teacher_signin_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next_path: str = Form(DASHBOARD_PATH),
):
    next_path = safe_next_path(next_path)
    email = normalize_email(email)

    with get_db() as conn:
        teacher = conn.execute(
            "SELECT id, email, password_hash FROM teachers WHERE email = ?",
            (email,),
        ).fetchone()

    if not teacher or not verify_password(password, teacher["password_hash"]):
        return templates.TemplateResponse(
            "teacher_signin.html",
            template_context(request, {"error": "Invalid email or password.", "next_path": next_path}),
            status_code=401,
        )

    token = create_teacher_session(teacher["id"])
    response = RedirectResponse(next_path, status_code=303)
    set_teacher_cookie(response, token)
    return response


@fastapi_app.post("/teacher/logout")
def teacher_logout(request: Request):
    session_token = request.cookies.get(TEACHER_SESSION_COOKIE)
    clear_teacher_session(session_token)
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie(TEACHER_SESSION_COOKIE)
    return response


@fastapi_app.get("/teacher/dashboard", response_class=HTMLResponse)
def teacher_dashboard(request: Request, error: str | None = None):
    teacher = current_teacher(request)
    if not teacher:
        return RedirectResponse(f"/teacher/signin?next_path={quote(DASHBOARD_PATH, safe='')}", status_code=303)

    classes = get_teacher_classes(teacher["id"])
    return templates.TemplateResponse(
        "teacher_dashboard.html",
        template_context(request, {"classes": classes, "error": error}),
    )


@fastapi_app.post("/teacher/classes")
def create_teacher_class(request: Request, class_name: str = Form(...)):
    teacher = current_teacher(request)
    if not teacher:
        return RedirectResponse(f"/teacher/signin?next_path={quote(DASHBOARD_PATH, safe='')}", status_code=303)

    class_name = normalize_class_name(class_name)
    if not is_valid_class_name(class_name):
        classes = get_teacher_classes(teacher["id"])
        return templates.TemplateResponse(
            "teacher_dashboard.html",
            template_context(
                request,
                {
                    "classes": classes,
                    "error": "Class URL must be 3-40 chars, lowercase letters/numbers/hyphens, and cannot start/end with a hyphen.",
                },
            ),
            status_code=400,
        )

    try:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO classes (teacher_id, class_name, created_at)
                VALUES (?, ?, ?)
                """,
                (teacher["id"], class_name, datetime.now(timezone.utc).isoformat()),
            )
    except sqlite3.IntegrityError:
        classes = get_teacher_classes(teacher["id"])
        return templates.TemplateResponse(
            "teacher_dashboard.html",
            template_context(
                request,
                {
                    "classes": classes,
                    "error": "That class URL is already taken. Choose another name.",
                },
            ),
            status_code=409,
        )

    return RedirectResponse(DASHBOARD_PATH, status_code=303)


@fastapi_app.get('/{class_id}', response_class=HTMLResponse)
def student_interface(request: Request, response: Response, class_id: str):
    student_id = request.cookies.get('student_id') or ''.join(random.choices(string.ascii_letters, k=12))
    class2students[class_id].add(student_id)
    response = templates.TemplateResponse(
        'student.html',
        template_context(request, {"timestamp": time.time(), "class_id": class_id}),
    )
    response.set_cookie(key='student_id', value=student_id)
    # print("Student class ID: ", class_id) 
    return response


@socket_server.event
async def connect(sid, environ):
    cookie_header = dict(environ.get("asgi.scope", {}).get("headers", [])).get(b"cookie", b"")
    cookie = SimpleCookie()
    cookie.load(cookie_header.decode("utf-8"))
    student_cookie = cookie.get("student_id")
    if student_cookie:
        sid2student[sid] = student_cookie.value
    teacher_cookie = cookie.get(TEACHER_SESSION_COOKIE)
    if teacher_cookie:
        teacher = get_teacher_by_session_token(teacher_cookie.value)
        if teacher:
            sid2teacher[sid] = teacher["id"]


@socket_server.on('register_student')
async def register_student(sid, timestamp: str, class_id: str):
    student_id = sid2student.get(sid)
    if not student_id:
        return

    await socket_server.emit('deactivate_old_tabs', # a student can only have a single tab active
            {'student_id':  student_id, 'timestamp': timestamp})
    student2color[student_id] = 'inactive' # upon connecting / opening a new tab a student is in an inactive state
    sid2student[sid] = student_id # used for keeping track of connected students
    sid2class[sid] = class_id
    previous_classes = {cls for cls, students in class2students.items() if student_id in students}
    for cls in class2students: # a student can only be in a single class
        class2students[cls].discard(student_id)
    class2students[class_id].add(student_id)
    for affected_class in (previous_classes | {class_id}):
        await emit_teacher_stats(affected_class)


def student_count(class_id): 
    class_students = class2students[class_id]
    return sum(1 for student in sid2student.values() if student in class_students)


def connected_student2color(class_id):
    return {k: v for k, v in student2color.items() if (k in class2students[class_id]) and (k in sid2student.values())}


def active_student_count(class_id): # active student == one who is connected and color != 'inactive'
    return sum(1 for color in connected_student2color(class_id).values() if color != 'inactive')


def color_fraction(class_id):
    connected_colors = list(connected_student2color(class_id).values())
    active_count = sum(1 for color in connected_colors if color != 'inactive') or 1
    return {
        color: sum(1 for student_color in connected_colors if student_color == color) / active_count
        for color in ['green', 'yellow', 'red']
    }


def teacher_stats(class_id):
    return {
        "class_id": class_id,
        "student_count": student_count(class_id),
        "active_student_count": active_student_count(class_id),
        "color2frac": color_fraction(class_id),
    }


async def emit_teacher_stats(class_id):
    payload = teacher_stats(class_id)
    for teacher_sid, teacher_class_id in teacher_sid2class.items():
        if teacher_class_id == class_id:
            await socket_server.emit("teacher_stats", payload, to=teacher_sid)


@fastapi_app.get('/{class_id}/teacher')
def teacher_interface(request: Request, class_id: str):
    teacher = current_teacher(request)
    if not teacher:
        return RedirectResponse(f"/teacher/signin?next_path={quote(f'/{class_id}/teacher', safe='')}", status_code=303)
    if not teacher_owns_class(teacher["id"], class_id):
        return RedirectResponse(DASHBOARD_PATH, status_code=303)
    return templates.TemplateResponse(
        'teacher.html',
        template_context(
            request,
            {
                "class_id": class_id,
                "student_count": student_count(class_id),
                "active_student_count": active_student_count(class_id),
                "color2frac": color_fraction(class_id),
            },
        ),
    )


@socket_server.on('join_teacher_class')
async def join_teacher_class(sid, class_id):
    teacher_id = sid2teacher.get(sid)
    if not teacher_id:
        return
    if not teacher_owns_class(teacher_id, class_id):
        return
    teacher_sid2class[sid] = class_id
    await socket_server.emit("teacher_stats", teacher_stats(class_id), to=sid)


@socket_server.on('color_change')
async def handle_color_change(sid, new_color):
    student_id = sid2student.get(sid)
    if student_id:
        student2color[student_id] = new_color
        class_id = sid2class.get(sid)
        if class_id:
            await emit_teacher_stats(class_id)


@socket_server.event
async def disconnect(sid):
    class_id = sid2class.pop(sid, None)
    sid2student.pop(sid, None)
    sid2teacher.pop(sid, None)
    teacher_sid2class.pop(sid, None)
    if class_id:
        await emit_teacher_stats(class_id)
