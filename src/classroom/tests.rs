use super::*;
use chrono::Duration;

fn now() -> DateTime<Utc> {
    DateTime::parse_from_rfc3339("2026-11-09T12:00:00+00:00")
        .unwrap()
        .with_timezone(&Utc)
}
fn connect(model: &mut Classroom, sid: &str, student: Option<&str>) {
    model.transition(
        Event::Connected {
            sid: sid.into(),
            student_id: student.map(str::to_owned),
        },
        now(),
    );
}
fn register(model: &mut Classroom, sid: &str, timestamp: f64, class: &str) -> Transition {
    model.transition(
        Event::Register {
            sid: sid.into(),
            timestamp,
            class_id: class.into(),
        },
        now(),
    )
}
fn color(model: &mut Classroom, sid: &str, feedback: Feedback) -> Transition {
    model.transition(
        Event::ChangeFeedback {
            sid: sid.into(),
            feedback,
        },
        now(),
    )
}
fn watch(model: &mut Classroom, sid: &str, class: &str, token: &str) -> Transition {
    model.transition(
        Event::Watch {
            sid: sid.into(),
            class_id: class.into(),
            session: Subscription {
                token: token.into(),
                expires_at: now() + Duration::seconds(10),
            },
        },
        now(),
    )
}
fn disconnect(model: &mut Classroom, sid: &str) -> Transition {
    model.transition(Event::Disconnected { sid: sid.into() }, now())
}

#[test]
fn registration_and_feedback_retries_are_idempotent() {
    let mut model = Classroom::default();
    connect(&mut model, "s", Some("student"));
    register(&mut model, "s", 1.0, "biology");
    color(&mut model, "s", Feedback::Green);
    connect(&mut model, "s", Some("student"));
    assert!(register(&mut model, "s", 1.0, "biology").effects.is_empty());
    assert!(color(&mut model, "s", Feedback::Green).effects.is_empty());
    let stats = model.stats("biology");
    assert_eq!(
        (
            stats.student_count,
            stats.active_student_count,
            stats.color2frac.green
        ),
        (1, 1, 1.0)
    );
}

#[test]
fn commands_cannot_skip_connection_and_registration_guards() {
    let mut model = Classroom::default();
    assert_eq!(
        register(&mut model, "missing", 1.0, "biology").outcome,
        Outcome::error(Rejection::Disconnected)
    );
    connect(&mut model, "anonymous", None);
    assert_eq!(
        register(&mut model, "anonymous", 1.0, "biology").outcome,
        Outcome::error(Rejection::StudentCookieRequired)
    );
    connect(&mut model, "s", Some("student"));
    assert_eq!(
        color(&mut model, "s", Feedback::Red).outcome,
        Outcome::error(Rejection::NotRegistered)
    );
    for timestamp in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            register(&mut model, "s", timestamp, "biology").outcome,
            Outcome::error(Rejection::InvalidRegistration)
        );
    }
    assert_eq!(
        register(&mut model, "s", 1.0, "").outcome,
        Outcome::error(Rejection::InvalidRegistration)
    );
    assert_eq!(model.stats("biology").student_count, 0);
}

#[test]
fn superseded_tabs_are_terminal_even_after_the_new_tab_disconnects() {
    let mut model = Classroom::default();
    connect(&mut model, "old", Some("student"));
    register(&mut model, "old", 1.0, "biology");
    connect(&mut model, "new", Some("student"));
    let effects = register(&mut model, "new", 2.0, "biology").effects;
    assert!(effects.contains(&Effect::Deactivate {
        sid: "old".into(),
        student_id: "student".into(),
        timestamp: 2.0
    }));
    disconnect(&mut model, "new");
    assert_eq!(
        color(&mut model, "old", Feedback::Green).outcome,
        Outcome::error(Rejection::Superseded)
    );
    assert_eq!(
        register(&mut model, "old", 99.0, "biology").outcome,
        Outcome::error(Rejection::Superseded)
    );
    assert_eq!(model.stats("biology").student_count, 0);
    // Reload means a new connection and a newly rendered page, so it can register.
    connect(&mut model, "reload", Some("student"));
    assert_eq!(
        register(&mut model, "reload", 100.0, "biology").outcome,
        Outcome::ok(CommandState::Registered)
    );
}

#[test]
fn equal_timestamps_still_have_one_owner_and_old_disconnect_is_harmless() {
    let mut model = Classroom::default();
    connect(&mut model, "a", Some("student"));
    connect(&mut model, "b", Some("student"));
    register(&mut model, "a", 1.0, "biology");
    register(&mut model, "b", 1.0, "biology");
    color(&mut model, "b", Feedback::Yellow);
    disconnect(&mut model, "a");
    disconnect(&mut model, "a");
    assert_eq!(model.stats("biology").student_count, 1);
    assert_eq!(model.stats("biology").color2frac.yellow, 1.0);
}

#[test]
fn older_replays_do_not_invalidate_the_current_registration() {
    let mut model = Classroom::default();
    connect(&mut model, "current", Some("student"));
    register(&mut model, "current", 3.0, "biology");
    color(&mut model, "current", Feedback::Red);
    assert_eq!(
        register(&mut model, "current", 2.0, "chemistry").outcome,
        Outcome::error(Rejection::InvalidRegistration)
    );
    assert_eq!(model.stats("biology").color2frac.red, 1.0);
    connect(&mut model, "stale-reconnect", Some("student"));
    assert_eq!(
        register(&mut model, "stale-reconnect", 1.0, "chemistry").outcome,
        Outcome::error(Rejection::Superseded)
    );
    assert_eq!(model.stats("biology").student_count, 1);
    assert_eq!(model.stats("chemistry").student_count, 0);
}

#[test]
fn moving_classes_resets_feedback_and_notifies_both_classes_only() {
    let mut model = Classroom::default();
    for (sid, class) in [("t1", "biology"), ("t2", "chemistry"), ("t3", "physics")] {
        connect(&mut model, sid, None);
        watch(&mut model, sid, class, "session");
    }
    connect(&mut model, "s", Some("student"));
    register(&mut model, "s", 1.0, "biology");
    color(&mut model, "s", Feedback::Green);
    let moved = register(&mut model, "s", 2.0, "chemistry");
    let recipients: Vec<_> = moved
        .effects
        .iter()
        .filter_map(|e| {
            if let Effect::Stats { sid, .. } = e {
                Some(sid.as_str())
            } else {
                None
            }
        })
        .collect();
    assert_eq!(recipients, ["t1", "t2"]);
    assert_eq!(model.stats("biology").student_count, 0);
    assert_eq!(model.stats("chemistry").student_count, 1);
    assert_eq!(model.stats("chemistry").active_student_count, 0);
}

#[test]
fn authorization_finishing_after_disconnect_cannot_resurrect_a_subscription() {
    let mut model = Classroom::default();
    connect(&mut model, "t", None);
    disconnect(&mut model, "t");
    let late = watch(&mut model, "t", "biology", "session");
    assert_eq!(late.outcome, Outcome::error(Rejection::Disconnected));
    assert!(late.effects.is_empty());
}

#[test]
fn denial_unsubscribes_a_previously_watching_teacher() {
    let mut model = Classroom::default();
    connect(&mut model, "t", None);
    watch(&mut model, "t", "biology", "session");
    model.transition(
        Event::Deny {
            sid: "t".into(),
            reason: Rejection::Forbidden,
        },
        now(),
    );
    connect(&mut model, "s", Some("student"));
    assert!(register(&mut model, "s", 1.0, "biology").effects.is_empty());
}

#[test]
fn expiry_is_checked_before_data_and_notified_once_at_the_boundary() {
    let mut model = Classroom::default();
    connect(&mut model, "t", None);
    watch(&mut model, "t", "biology", "session");
    let expired = model.transition(
        Event::Register {
            sid: "missing".into(),
            timestamp: 1.0,
            class_id: "biology".into(),
        },
        now() + Duration::seconds(10),
    );
    assert_eq!(
        expired.effects,
        [Effect::AccessEnded {
            sid: "t".into(),
            reason: Rejection::Expired
        }]
    );
    assert!(
        model
            .transition(Event::Tick, now() + Duration::seconds(11))
            .effects
            .is_empty()
    );
    let rejected = model.transition(
        Event::Watch {
            sid: "t".into(),
            class_id: "biology".into(),
            session: Subscription {
                token: "expired".into(),
                expires_at: now(),
            },
        },
        now(),
    );
    assert_eq!(rejected.outcome, Outcome::error(Rejection::Expired));
}

#[test]
fn logout_revokes_only_its_session_and_feedback_remains_available_to_others() {
    let mut model = Classroom::default();
    for (sid, token) in [("t1", "first"), ("t2", "second")] {
        connect(&mut model, sid, None);
        watch(&mut model, sid, "biology", token);
    }
    let revoke = model.transition(
        Event::RevokeSession {
            token: "first".into(),
        },
        now(),
    );
    assert_eq!(
        revoke.effects,
        [Effect::AccessEnded {
            sid: "t1".into(),
            reason: Rejection::Revoked
        }]
    );
    connect(&mut model, "s", Some("student"));
    let registered = register(&mut model, "s", 1.0, "biology");
    assert!(matches!(&registered.effects[..], [Effect::Stats { sid, .. }] if sid == "t2"));
}

#[test]
fn mixed_event_sequences_keep_counts_and_fractions_consistent() {
    let mut model = Classroom::default();
    // Deterministic sequence exercises repeated joins, supersession, moves, colors, and disconnects.
    let mut seed = 7_u64;
    for _ in 0..500 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let index = (seed >> 32) % 6;
        let sid = format!("socket-{index}");
        let class = if index.is_multiple_of(2) {
            "biology"
        } else {
            "chemistry"
        };
        let event = match seed % 4 {
            0 => Event::Connected {
                sid,
                student_id: Some(format!("student-{}", index % 3)),
            },
            1 => Event::Register {
                sid,
                timestamp: (seed >> 32) as f64,
                class_id: class.into(),
            },
            2 => Event::ChangeFeedback {
                sid,
                feedback: [
                    Feedback::Inactive,
                    Feedback::Red,
                    Feedback::Yellow,
                    Feedback::Green,
                ][index as usize % 4],
            },
            _ => Event::Disconnected { sid },
        };
        model.transition(event, now());
        let registered: Vec<_> = model
            .connections
            .values()
            .filter_map(|c| match &c.student {
                StudentState::Registered(s) => Some(s.student_id.as_str()),
                _ => None,
            })
            .collect();
        let unique: BTreeSet<_> = registered.iter().collect();
        assert_eq!(registered.len(), unique.len());
        for class in ["biology", "chemistry"] {
            let stats = model.stats(class);
            assert!(stats.active_student_count <= stats.student_count);
            let sum = stats.color2frac.green + stats.color2frac.yellow + stats.color2frac.red;
            assert!(
                (sum - if stats.active_student_count == 0 {
                    0.0
                } else {
                    1.0
                })
                .abs()
                    < 0.000001
            );
        }
    }
}
