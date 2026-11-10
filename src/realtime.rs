//! Socket.IO adapter: deserialize commands, authorize, transition, execute effects.
use std::{collections::HashMap, future::Future, sync::Arc, time::Duration};

use axum_extra::extract::cookie::CookieJar;
use chrono::Utc;
use serde_json::json;
use socketioxide::extract::{AckSender, Data, SocketRef};
use tokio::sync::Mutex;

use crate::{
    AppState, auth,
    classroom::{Classroom, Effect, Event, Feedback, Outcome, Rejection, Stats, Subscription},
};

#[derive(Default)]
pub(crate) struct LiveState {
    classroom: Classroom,
    sockets: HashMap<String, SocketRef>,
}

impl LiveState {
    pub(crate) fn stats(&self, class_id: &str) -> Stats {
        self.classroom.stats(class_id)
    }

    fn attach(&mut self, socket: &SocketRef) {
        let sid = socket.id.to_string();
        if !socket.connected() {
            // A database result can finish before the disconnect callback runs.
            // Observe the transport state now, so that result cannot subscribe.
            self.sockets.remove(&sid);
            self.apply(Event::Disconnected { sid });
            return;
        }
        if !self.sockets.contains_key(&sid) {
            let jar = CookieJar::from_headers(&socket.req_parts().headers);
            self.sockets.insert(sid.clone(), socket.clone());
            self.apply(Event::Connected {
                sid,
                student_id: jar.get("student_id").map(|c| c.value().into()),
            });
        }
    }

    pub(crate) fn apply(&mut self, event: Event) -> Outcome {
        let transition = self.classroom.transition(event, Utc::now());
        for effect in transition.effects {
            let (sid, result) = match effect {
                Effect::Stats { sid, stats } => {
                    let result = self
                        .sockets
                        .get(&sid)
                        .map(|s| s.emit("teacher_stats", &stats));
                    (sid, result)
                }
                Effect::Deactivate {
                    sid,
                    student_id,
                    timestamp,
                } => {
                    let payload = json!({
                        "student_id": student_id, "timestamp": timestamp, "superseded": true,
                    });
                    let result = self
                        .sockets
                        .get(&sid)
                        .map(|s| s.emit("deactivate_old_tabs", &payload));
                    (sid, result)
                }
                Effect::AccessEnded { sid, reason } => {
                    let result = self
                        .sockets
                        .get(&sid)
                        .map(|s| s.emit("teacher_access_ended", &json!({"reason": reason})));
                    (sid, result)
                }
            };
            if let Some(Err(error)) = result {
                tracing::debug!(%sid, %error, "state effect could not be delivered");
            }
        }
        match &transition.outcome {
            Outcome::Ok { state } => tracing::trace!(?state, "classroom transition"),
            Outcome::Error { reason } => tracing::debug!(?reason, "classroom command rejected"),
        }
        transition.outcome
    }
}

/// Expiry remains observable even when no students are sending events.
pub(crate) fn start_expiry_clock(live: &Arc<Mutex<LiveState>>) {
    let live = Arc::downgrade(live);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        loop {
            interval.tick().await;
            let Some(live) = live.upgrade() else {
                break;
            };
            live.lock().await.apply(Event::Tick);
        }
    });
}

// Install handlers synchronously when Socketioxide invokes the connect closure,
// before returning a future. A first command can safely attach its own connection.
pub(crate) fn connect(socket: SocketRef, state: AppState) -> impl Future<Output = ()> + Send {
    let register_state = state.clone();
    socket.on(
        "register_student",
        move |socket: SocketRef,
              Data((timestamp, class_id)): Data<(f64, String)>,
              ack: AckSender| {
            let state = register_state.clone();
            async move {
                let mut live = state.live.lock().await;
                live.attach(&socket);
                let outcome = live.apply(Event::Register {
                    sid: socket.id.to_string(),
                    timestamp,
                    class_id,
                });
                let _ = ack.send(&outcome);
            }
        },
    );

    let color_state = state.clone();
    socket.on(
        "color_change",
        move |socket: SocketRef, Data(feedback): Data<Feedback>, ack: AckSender| {
            let state = color_state.clone();
            async move {
                let mut live = state.live.lock().await;
                live.attach(&socket);
                let outcome = live.apply(Event::ChangeFeedback {
                    sid: socket.id.to_string(),
                    feedback,
                });
                let _ = ack.send(&outcome);
            }
        },
    );

    let teacher_state = state.clone();
    socket.on(
        "join_teacher_class",
        move |socket: SocketRef, Data(class_id): Data<String>, ack: AckSender| {
            let state = teacher_state.clone();
            async move {
                let outcome = match join_teacher(socket.clone(), state.clone(), class_id).await {
                    Ok(outcome) => outcome,
                    Err(error) => {
                        tracing::error!(%error, "joining teacher class failed");
                        state.live.lock().await.apply(Event::Deny {
                            sid: socket.id.to_string(),
                            reason: Rejection::Unavailable,
                        })
                    }
                };
                let _ = ack.send(&outcome);
            }
        },
    );

    let disconnect_state = state.clone();
    socket.on_disconnect(move |socket: SocketRef| {
        let state = disconnect_state.clone();
        async move {
            let mut live = state.live.lock().await;
            let sid = socket.id.to_string();
            live.sockets.remove(&sid);
            live.apply(Event::Disconnected { sid });
        }
    });

    async move {
        state.live.lock().await.attach(&socket);
    }
}

async fn join_teacher(
    socket: SocketRef,
    state: AppState,
    class_id: String,
) -> anyhow::Result<Outcome> {
    // Serialize authorization + subscribe with logout + revoke. Database work
    // holds only this auth gate, never the classroom mutex.
    let _auth_guard = state.auth_gate.lock().await;
    let jar = CookieJar::from_headers(&socket.req_parts().headers);
    let access = auth::class_access(&state.db, &jar, &class_id).await?;
    let mut live = state.live.lock().await;
    live.attach(&socket);
    let sid = socket.id.to_string();
    let event = match access {
        auth::ClassAccess::Allowed(session) => Event::Watch {
            sid,
            class_id,
            session: Subscription {
                token: session.token,
                expires_at: session.expires_at,
            },
        },
        auth::ClassAccess::SignInRequired(auth::SessionFailure::Expired) => Event::Deny {
            sid,
            reason: Rejection::Expired,
        },
        auth::ClassAccess::SignInRequired(_) => Event::Deny {
            sid,
            reason: Rejection::Unauthenticated,
        },
        auth::ClassAccess::Forbidden => Event::Deny {
            sid,
            reason: Rejection::Forbidden,
        },
    };
    Ok(live.apply(event))
}
