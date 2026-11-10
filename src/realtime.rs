use std::collections::HashMap;

use axum_extra::extract::cookie::CookieJar;
use serde::{Deserialize, Serialize};
use serde_json::json;
use socketioxide::extract::{Data, SocketRef};

use crate::{AppState, auth, db};

#[derive(Clone, Copy, Default, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Color {
    Green,
    Yellow,
    Red,
    #[default]
    Inactive,
}

struct Student {
    student_id: String,
    class_id: String,
    timestamp: f64,
    color: Color,
    socket: SocketRef,
}

struct Teacher {
    class_id: String,
    session_token: String,
    socket: SocketRef,
}

#[derive(Default)]
pub(crate) struct LiveState {
    students: HashMap<String, Student>,
    teachers: HashMap<String, Teacher>,
}

#[derive(Debug, Serialize)]
pub(crate) struct Stats {
    class_id: String,
    student_count: usize,
    active_student_count: usize,
    color2frac: Fractions,
}

#[derive(Debug, Serialize)]
struct Fractions {
    green: f64,
    yellow: f64,
    red: f64,
}

impl LiveState {
    pub(crate) fn stats(&self, class_id: &str) -> Stats {
        let mut student_count = 0;
        let (mut green, mut yellow, mut red) = (0, 0, 0);
        for student in self.students.values().filter(|s| s.class_id == class_id) {
            student_count += 1;
            match student.color {
                Color::Green => green += 1,
                Color::Yellow => yellow += 1,
                Color::Red => red += 1,
                Color::Inactive => {}
            }
        }
        let active_student_count = green + yellow + red;
        let denominator = active_student_count.max(1) as f64;
        Stats {
            class_id: class_id.into(),
            student_count,
            active_student_count,
            color2frac: Fractions {
                green: green as f64 / denominator,
                yellow: yellow as f64 / denominator,
                red: red as f64 / denominator,
            },
        }
    }

    fn emit_stats(&self, class_id: &str) {
        let stats = self.stats(class_id);
        for teacher in self.teachers.values().filter(|t| t.class_id == class_id) {
            if let Err(error) = teacher.socket.emit("teacher_stats", &stats) {
                tracing::debug!(%error, "teacher socket unavailable");
            }
        }
    }

    fn register(
        &mut self,
        socket: SocketRef,
        student_id: String,
        timestamp: f64,
        class_id: String,
    ) {
        if !timestamp.is_finite() {
            return;
        }
        // Keep one active connection per student cookie, even while old tabs disconnect.
        if let Some(newer) = self
            .students
            .values()
            .find(|s| s.student_id == student_id && s.timestamp > timestamp)
        {
            let _ = socket.emit(
                "deactivate_old_tabs",
                &json!({"student_id": student_id, "timestamp": newer.timestamp}),
            );
            return;
        }
        let current_sid = socket.id.to_string();
        let old_sids: Vec<_> = self
            .students
            .iter()
            .filter(|(sid, student)| student.student_id == student_id || **sid == current_sid)
            .map(|(sid, _)| sid.clone())
            .collect();
        let mut affected = vec![class_id.clone()];
        for sid in old_sids {
            if let Some(old) = self.students.remove(&sid) {
                if old.socket.id != socket.id {
                    let _ = old.socket.emit(
                        "deactivate_old_tabs",
                        &json!({"student_id": student_id, "timestamp": timestamp}),
                    );
                }
                affected.push(old.class_id);
            }
        }
        self.students.insert(
            current_sid,
            Student {
                student_id,
                class_id,
                timestamp,
                color: Color::Inactive,
                socket,
            },
        );
        affected.sort();
        affected.dedup();
        for class in affected {
            self.emit_stats(&class);
        }
    }

    pub(crate) fn remove_teacher_session(&mut self, token: &str) {
        self.teachers
            .retain(|_, teacher| teacher.session_token != token);
    }
}

pub(crate) async fn connect(socket: SocketRef, state: AppState) {
    let register_state = state.clone();
    socket.on(
        "register_student",
        move |socket: SocketRef, Data((timestamp, class_id)): Data<(f64, String)>| {
            let state = register_state.clone();
            async move {
                let jar = CookieJar::from_headers(&socket.req_parts().headers);
                if let Some(cookie) = jar.get("student_id") {
                    state.live.lock().await.register(
                        socket,
                        cookie.value().to_owned(),
                        timestamp,
                        class_id,
                    );
                }
            }
        },
    );

    let color_state = state.clone();
    socket.on(
        "color_change",
        move |socket: SocketRef, Data(color): Data<Color>| {
            let state = color_state.clone();
            async move {
                let mut live = state.live.lock().await;
                if let Some(student) = live.students.get_mut(&socket.id.to_string()) {
                    student.color = color;
                    let class_id = student.class_id.clone();
                    live.emit_stats(&class_id);
                }
            }
        },
    );

    let teacher_state = state.clone();
    socket.on(
        "join_teacher_class",
        move |socket: SocketRef, Data(class_id): Data<String>| {
            let state = teacher_state.clone();
            async move {
                if let Err(error) = join_teacher(socket, state, class_id).await {
                    tracing::error!(%error, "joining teacher class failed");
                }
            }
        },
    );

    socket.on_disconnect(move |socket: SocketRef| {
        let state = state.clone();
        async move {
            let mut live = state.live.lock().await;
            let sid = socket.id.to_string();
            live.teachers.remove(&sid);
            if let Some(student) = live.students.remove(&sid) {
                live.emit_stats(&student.class_id);
            }
        }
    });
}

async fn join_teacher(socket: SocketRef, state: AppState, class_id: String) -> anyhow::Result<()> {
    let jar = CookieJar::from_headers(&socket.req_parts().headers);
    let Some(cookie) = jar.get(auth::COOKIE) else {
        return Ok(());
    };
    // Recheck the session at subscription time, including expiry and class ownership.
    let Some(teacher) = db::teacher_by_token(&state.db, cookie.value()).await? else {
        return Ok(());
    };
    if !db::owns_class(&state.db, teacher.id, &class_id).await? {
        return Ok(());
    }
    let mut live = state.live.lock().await;
    socket.emit("teacher_stats", &live.stats(&class_id))?;
    live.teachers.insert(
        socket.id.to_string(),
        Teacher {
            class_id,
            session_token: cookie.value().to_owned(),
            socket,
        },
    );
    Ok(())
}
