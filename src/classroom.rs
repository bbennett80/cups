//! Pure classroom state machine. No sockets, database, clocks, or background tasks.
//! All mutations enter through `transition`; the adapter executes returned effects.

use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Feedback {
    #[default]
    Inactive,
    Green,
    Yellow,
    Red,
}

#[derive(Clone)]
pub(crate) struct Subscription {
    pub token: String,
    pub expires_at: DateTime<Utc>,
}

struct Participation {
    student_id: String,
    class_id: String,
    timestamp: f64,
    feedback: Feedback,
}

enum StudentState {
    Unregistered,
    Registered(Participation),
    Superseded { timestamp: f64 },
}

enum TeacherState {
    NotWatching,
    Watching {
        class_id: String,
        session: Subscription,
    },
    Denied,
}

struct Connection {
    student_id: Option<String>,
    student: StudentState,
    teacher: TeacherState,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Rejection {
    Disconnected,
    StudentCookieRequired,
    InvalidRegistration,
    Superseded,
    NotRegistered,
    Unauthenticated,
    Expired,
    Revoked,
    Forbidden,
    Unavailable,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CommandState {
    Connected,
    Registered,
    Updated,
    Watching,
    Disconnected,
    Revoked,
    Checked,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum Outcome {
    Ok { state: CommandState },
    Error { reason: Rejection },
}

impl Outcome {
    fn ok(state: CommandState) -> Self {
        Self::Ok { state }
    }
    fn error(reason: Rejection) -> Self {
        Self::Error { reason }
    }
}

pub(crate) enum Event {
    Connected {
        sid: String,
        student_id: Option<String>,
    },
    Register {
        sid: String,
        timestamp: f64,
        class_id: String,
    },
    ChangeFeedback {
        sid: String,
        feedback: Feedback,
    },
    Watch {
        sid: String,
        class_id: String,
        session: Subscription,
    },
    Deny {
        sid: String,
        reason: Rejection,
    },
    Disconnected {
        sid: String,
    },
    RevokeSession {
        token: String,
    },
    Tick,
}

#[derive(Debug, PartialEq)]
pub(crate) enum Effect {
    Stats {
        sid: String,
        stats: Stats,
    },
    Deactivate {
        sid: String,
        student_id: String,
        timestamp: f64,
    },
    AccessEnded {
        sid: String,
        reason: Rejection,
    },
}

pub(crate) struct Transition {
    pub outcome: Outcome,
    pub effects: Vec<Effect>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub(crate) struct Stats {
    pub class_id: String,
    pub student_count: usize,
    pub active_student_count: usize,
    pub color2frac: Fractions,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub(crate) struct Fractions {
    pub green: f64,
    pub yellow: f64,
    pub red: f64,
}

#[derive(Default)]
pub(crate) struct Classroom {
    connections: BTreeMap<String, Connection>,
}

impl Classroom {
    pub(crate) fn stats(&self, class_id: &str) -> Stats {
        let (mut count, mut green, mut yellow, mut red) = (0, 0, 0, 0);
        for connection in self.connections.values() {
            if let StudentState::Registered(student) = &connection.student {
                if student.class_id != class_id {
                    continue;
                }
                count += 1;
                match student.feedback {
                    Feedback::Inactive => {}
                    Feedback::Green => green += 1,
                    Feedback::Yellow => yellow += 1,
                    Feedback::Red => red += 1,
                }
            }
        }
        let active = green + yellow + red;
        let denominator = active.max(1) as f64;
        Stats {
            class_id: class_id.into(),
            student_count: count,
            active_student_count: active,
            color2frac: Fractions {
                green: green as f64 / denominator,
                yellow: yellow as f64 / denominator,
                red: red as f64 / denominator,
            },
        }
    }

    /// `now` is supplied by the caller, so expiry and races can be tested without sleeps.
    pub(crate) fn transition(&mut self, event: Event, now: DateTime<Utc>) -> Transition {
        let mut effects = Vec::new();
        // Every event checks expiry before generating any classroom data.
        for (sid, connection) in &mut self.connections {
            if matches!(&connection.teacher, TeacherState::Watching { session, .. } if session.expires_at <= now)
            {
                connection.teacher = TeacherState::Denied;
                effects.push(Effect::AccessEnded {
                    sid: sid.clone(),
                    reason: Rejection::Expired,
                });
            }
        }
        let mut changed = BTreeSet::new();
        let outcome = match event {
            Event::Connected { sid, student_id } => {
                self.connections.entry(sid).or_insert(Connection {
                    student_id,
                    student: StudentState::Unregistered,
                    teacher: TeacherState::NotWatching,
                });
                Outcome::ok(CommandState::Connected)
            }
            Event::Register {
                sid,
                timestamp,
                class_id,
            } => self.register(&sid, timestamp, class_id, &mut changed, &mut effects),
            Event::ChangeFeedback { sid, feedback } => match self.connections.get_mut(&sid) {
                Some(Connection {
                    student: StudentState::Registered(student),
                    ..
                }) => {
                    if student.feedback != feedback {
                        student.feedback = feedback;
                        changed.insert(student.class_id.clone());
                    }
                    Outcome::ok(CommandState::Updated)
                }
                Some(Connection {
                    student: StudentState::Superseded { .. },
                    ..
                }) => Outcome::error(Rejection::Superseded),
                Some(_) => Outcome::error(Rejection::NotRegistered),
                None => Outcome::error(Rejection::Disconnected),
            },
            Event::Watch {
                sid,
                class_id,
                session,
            } => match self.connections.get_mut(&sid) {
                Some(connection) if session.expires_at > now => {
                    connection.teacher = TeacherState::Watching {
                        class_id: class_id.clone(),
                        session,
                    };
                    effects.push(Effect::Stats {
                        sid,
                        stats: self.stats(&class_id),
                    });
                    Outcome::ok(CommandState::Watching)
                }
                Some(connection) => {
                    connection.teacher = TeacherState::Denied;
                    Outcome::error(Rejection::Expired)
                }
                None => Outcome::error(Rejection::Disconnected),
            },
            Event::Deny { sid, reason } => match self.connections.get_mut(&sid) {
                Some(connection) => {
                    connection.teacher = TeacherState::Denied;
                    Outcome::error(reason)
                }
                None => Outcome::error(Rejection::Disconnected),
            },
            Event::Disconnected { sid } => {
                if let Some(Connection {
                    student: StudentState::Registered(student),
                    ..
                }) = self.connections.remove(&sid)
                {
                    changed.insert(student.class_id);
                }
                Outcome::ok(CommandState::Disconnected)
            }
            Event::RevokeSession { token } => {
                for (sid, connection) in &mut self.connections {
                    if matches!(&connection.teacher, TeacherState::Watching { session, .. } if session.token == token)
                    {
                        connection.teacher = TeacherState::Denied;
                        effects.push(Effect::AccessEnded {
                            sid: sid.clone(),
                            reason: Rejection::Revoked,
                        });
                    }
                }
                Outcome::ok(CommandState::Revoked)
            }
            Event::Tick => Outcome::ok(CommandState::Checked),
        };
        for class_id in changed {
            let stats = self.stats(&class_id);
            for (sid, connection) in &self.connections {
                if matches!(&connection.teacher, TeacherState::Watching { class_id: watched, .. } if *watched == class_id)
                {
                    effects.push(Effect::Stats {
                        sid: sid.clone(),
                        stats: stats.clone(),
                    });
                }
            }
        }
        Transition { outcome, effects }
    }

    fn register(
        &mut self,
        sid: &str,
        timestamp: f64,
        class_id: String,
        changed: &mut BTreeSet<String>,
        effects: &mut Vec<Effect>,
    ) -> Outcome {
        let Some(connection) = self.connections.get(sid) else {
            return Outcome::error(Rejection::Disconnected);
        };
        let Some(student_id) = connection.student_id.clone() else {
            return Outcome::error(Rejection::StudentCookieRequired);
        };
        if !timestamp.is_finite() || class_id.is_empty() {
            return Outcome::error(Rejection::InvalidRegistration);
        }
        if let StudentState::Superseded { timestamp } = connection.student {
            effects.push(Effect::Deactivate {
                sid: sid.into(),
                student_id,
                timestamp,
            });
            return Outcome::error(Rejection::Superseded);
        }
        if matches!(&connection.student, StudentState::Registered(s) if s.timestamp == timestamp && s.class_id == class_id)
        {
            // Registration retries must not erase feedback or publish duplicate stats.
            return Outcome::ok(CommandState::Registered);
        }
        if matches!(&connection.student, StudentState::Registered(s) if timestamp < s.timestamp) {
            return Outcome::error(Rejection::InvalidRegistration);
        }
        let newest = self
            .connections
            .values()
            .filter_map(|c| match &c.student {
                StudentState::Registered(s) if s.student_id == student_id => Some(s.timestamp),
                _ => None,
            })
            .max_by(f64::total_cmp);
        if let Some(newer) = newest.filter(|newer| *newer > timestamp) {
            self.connections.get_mut(sid).unwrap().student =
                StudentState::Superseded { timestamp: newer };
            effects.push(Effect::Deactivate {
                sid: sid.into(),
                student_id,
                timestamp: newer,
            });
            return Outcome::error(Rejection::Superseded);
        }
        for (old_sid, connection) in &mut self.connections {
            if let StudentState::Registered(old) = &connection.student
                && old.student_id == student_id
            {
                changed.insert(old.class_id.clone());
                if old_sid != sid {
                    effects.push(Effect::Deactivate {
                        sid: old_sid.clone(),
                        student_id: student_id.clone(),
                        timestamp,
                    });
                }
                connection.student = StudentState::Superseded { timestamp };
            }
        }
        changed.insert(class_id.clone());
        self.connections.get_mut(sid).unwrap().student = StudentState::Registered(Participation {
            student_id,
            class_id,
            timestamp,
            feedback: Feedback::Inactive,
        });
        Outcome::ok(CommandState::Registered)
    }
}

#[cfg(test)]
mod tests;
