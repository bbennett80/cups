# from flask import Flask, render_template, request, make_response
from fastapi import FastAPI, Request, Cookie
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# from flask_socketio import SocketIO, emit
from fastapi_socketio import SocketManager, emit


import random, string, collections, time
from urllib.parse import urlparse

app = FastAPI()
socketio = SocketManager(app)

templates = Jinja2Templates(directory="templates")

sid2student = dict()
student2color = dict()
class2students = collections.defaultdict(lambda: set())

@app.get("/", response_class=HTMLResponse)
def root(request: Request): 
    return templates.TemplateResponse('howto.html', {"request": request})


@app.route('/{class_id}', response_class=HTMLResponse)
def student_interface(request: Request, class_id: collections.Union[str, None] = Cookie(default=None)):
    student_id = Cookie.get('student_id') or ''.join(random.choices(string.ascii_letters, k=12))
    class2students[class_id].add(student_id)
    response = templates.TemplateResponse('student.html', timestamp=time.time(), class_id=class_id))
    response.set_cookie('student_id', student_id)
    return {"request": request, "class_id": class_id}


@socketio.on('register_student')
def register_student(request: Request, timestamp: str, class_id: str):
    student_id = Cookie.get('student_id')
    emit('deactivate_old_tabs', # a student can only have a single tab active
            {'student_id':  student_id, 'timestamp': timestamp}, broadcast=True, namespace='/')
    student2color[student_id] = 'inactive' # upon connecting / opening a new tab a student is in an inactive state
    sid2student[request.sid] = student_id # used for keeping track of connected students
    for cls in class2students: # a student can only be in a single class
        class2students[cls].discard(student_id)
    class2students[class_id].add(student_id)


def student_count(class_id): 
    return list(sid2student.values()).filter(lambda s: s in class2students[class_id]).count()


def connected_student2color(class_id):
    return {k: v for k, v in student2color.items() if (k in class2students[class_id]) and (k in sid2student.values())}


def active_student_count(class_id): # active student == one who is connected and color != 'inactive'
    return L(connected_student2color(class_id).values()).filter(lambda c: c != 'inactive').count()


def color_fraction(class_id):
    return {color: list(connected_student2color(class_id).values()).map(eq(color)).sum()/(active_student_count(class_id) or 1)
            for color in ['green', 'yellow', 'red']}


@app.get('/{class_id}/teacher')
def teacher_interface(request: Request, class_id: str):
    return templates.TemplateResponse('teacher.html', 
                                      {
                                      "request": request, 
                                      "class_id": class_id,
                                      "student_count": student_count(class_id),
                                      "active_student_count": active_student_count(class_id), 
                                      "color2frac": color_fraction(class_id)
                                      })


@socketio.on('color_change')
def handle_color_change(new_color): student2color[Request.cookies['student_id']] = new_color


@socketio.on('disconnect')
def handle_disconnect():
    student = sid2student.pop(Request.sid, None)


# def count(self: list()): return len(self)
