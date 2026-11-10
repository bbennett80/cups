use axum_extra::extract::cookie::{Cookie, CookieJar, SameSite};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use chrono::{DateTime, Duration, SecondsFormat, Utc};
use pbkdf2::pbkdf2_hmac;
use rand::{RngCore, rngs::OsRng};
use sha2::Sha256;
use sqlx::{SqliteConnection, SqlitePool};
use subtle::ConstantTimeEq;

use crate::db::{self, Teacher};

pub(crate) const COOKIE: &str = "teacher_session";
pub(crate) const DASHBOARD: &str = "/teacher/dashboard";
const SESSION_DAYS: i64 = 30;
const ITERATIONS: u32 = 200_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SessionFailure {
    Missing,
    Invalid,
    Expired,
}

pub(crate) struct AuthenticatedSession {
    pub teacher: Teacher,
    pub token: String,
    pub expires_at: DateTime<Utc>,
}

pub(crate) enum SessionState {
    Anonymous(SessionFailure),
    Authenticated(AuthenticatedSession),
}

pub(crate) enum ClassAccess {
    Allowed(AuthenticatedSession),
    SignInRequired(SessionFailure),
    Forbidden,
}

impl SessionState {
    fn resolve(token: Option<&str>, record: Option<db::SessionRecord>, now: DateTime<Utc>) -> Self {
        let Some(token) = token else {
            return Self::Anonymous(SessionFailure::Missing);
        };
        let Some(record) = record else {
            return Self::Anonymous(SessionFailure::Invalid);
        };
        let Ok(expires_at) = DateTime::parse_from_rfc3339(&record.expires_at) else {
            return Self::Anonymous(SessionFailure::Invalid);
        };
        let expires_at = expires_at.with_timezone(&Utc);
        if expires_at <= now {
            return Self::Anonymous(SessionFailure::Expired);
        }
        Self::Authenticated(AuthenticatedSession {
            teacher: record.teacher,
            token: token.into(),
            expires_at,
        })
    }
}

pub(crate) async fn session_state(
    db: &SqlitePool,
    jar: &CookieJar,
) -> Result<SessionState, sqlx::Error> {
    let token = jar.get(COOKIE).map(|c| c.value());
    let record = match token {
        Some(token) => db::session_by_token(db, token).await?,
        None => None,
    };
    Ok(SessionState::resolve(token, record, Utc::now()))
}

pub(crate) async fn class_access(
    db: &SqlitePool,
    jar: &CookieJar,
    class_id: &str,
) -> Result<ClassAccess, sqlx::Error> {
    match session_state(db, jar).await? {
        SessionState::Anonymous(reason) => Ok(ClassAccess::SignInRequired(reason)),
        SessionState::Authenticated(session) => {
            if db::owns_class(db, session.teacher.id, class_id).await? {
                Ok(ClassAccess::Allowed(session))
            } else {
                Ok(ClassAccess::Forbidden)
            }
        }
    }
}

pub(crate) fn random_token() -> String {
    let mut bytes = [0; 32];
    OsRng.fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

pub(crate) fn hash_password(password: &str) -> String {
    let mut salt = [0; 16];
    OsRng.fill_bytes(&mut salt);
    let mut digest = [0; 32];
    pbkdf2_hmac::<Sha256>(password.as_bytes(), &salt, ITERATIONS, &mut digest);
    format!(
        "pbkdf2_sha256${ITERATIONS}${}${}",
        hex::encode(salt),
        hex::encode(digest)
    )
}

pub(crate) fn verify_password(password: &str, hash: &str) -> bool {
    let parts: Vec<_> = hash.split('$').collect();
    if parts.len() != 4 || parts[0] != "pbkdf2_sha256" {
        return false;
    }
    let Ok(iterations) = parts[1].parse::<u32>() else {
        return false;
    };
    if !(1..=2_000_000).contains(&iterations) {
        return false;
    }
    let (Ok(salt), Ok(expected)) = (hex::decode(parts[2]), hex::decode(parts[3])) else {
        return false;
    };
    if expected.len() != 32 {
        return false;
    }
    let mut computed = [0; 32];
    pbkdf2_hmac::<Sha256>(password.as_bytes(), &salt, iterations, &mut computed);
    bool::from(computed.as_slice().ct_eq(&expected))
}

pub(crate) fn safe_next_path(path: Option<&str>) -> &str {
    // Browsers normalize backslashes to slashes; never accept them in redirects.
    path.filter(|p| {
        p.starts_with('/')
            && !p.starts_with("//")
            && !p.contains('\\')
            && !p.chars().any(char::is_control)
    })
    .unwrap_or(DASHBOARD)
}

pub(crate) async fn current_teacher(
    db: &SqlitePool,
    jar: &CookieJar,
) -> Result<Option<Teacher>, sqlx::Error> {
    Ok(match session_state(db, jar).await? {
        SessionState::Authenticated(session) => Some(session.teacher),
        SessionState::Anonymous(_) => None,
    })
}

/// Prepared → persisted → committed → cookie issued. Never issue before commit.
pub(crate) struct NewSession {
    token: String,
    created_at: DateTime<Utc>,
}

impl NewSession {
    pub(crate) fn prepare() -> Self {
        Self {
            token: random_token(),
            created_at: Utc::now(),
        }
    }

    pub(crate) async fn persist(
        &self,
        conn: &mut SqliteConnection,
        teacher_id: i64,
    ) -> Result<(), sqlx::Error> {
        sqlx::query("INSERT INTO teacher_sessions (token, teacher_id, created_at, expires_at) VALUES (?, ?, ?, ?)")
            .bind(&self.token).bind(teacher_id)
            .bind(self.created_at.to_rfc3339_opts(SecondsFormat::Micros, false))
            .bind((self.created_at + Duration::days(SESSION_DAYS)).to_rfc3339_opts(SecondsFormat::Micros, false))
            .execute(conn).await?;
        Ok(())
    }

    pub(crate) fn issue_cookie(self, jar: CookieJar) -> CookieJar {
        jar.add(
            Cookie::build((COOKIE, self.token))
                .path("/")
                .http_only(true)
                .same_site(SameSite::Lax)
                .max_age(time::Duration::days(SESSION_DAYS))
                .build(),
        )
    }
}

pub(crate) async fn session(
    db: &SqlitePool,
    teacher_id: i64,
    jar: CookieJar,
) -> Result<CookieJar, sqlx::Error> {
    let session = NewSession::prepare();
    let mut transaction = db.begin().await?;
    session.persist(&mut transaction, teacher_id).await?;
    transaction.commit().await?;
    Ok(session.issue_cookie(jar))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(expires: &str) -> db::SessionRecord {
        db::SessionRecord {
            teacher: Teacher {
                id: 1,
                email: "teacher@test.com".into(),
            },
            expires_at: expires.into(),
        }
    }

    #[test]
    fn sessions_distinguish_missing_invalid_expired_and_authenticated() {
        let now = DateTime::parse_from_rfc3339("2026-11-09T12:00:00+00:00")
            .unwrap()
            .with_timezone(&Utc);
        for (token, row, reason) in [
            (None, None, SessionFailure::Missing),
            (Some("invalid"), None, SessionFailure::Invalid),
            (
                Some("bad-date"),
                Some(record("invalid")),
                SessionFailure::Invalid,
            ),
            (
                Some("expired"),
                Some(record("2026-11-09T12:00:00+00:00")),
                SessionFailure::Expired,
            ),
        ] {
            assert!(
                matches!(SessionState::resolve(token, row, now), SessionState::Anonymous(actual) if actual == reason)
            );
        }
        // Python-compatible RFC3339 timestamps and non-UTC offsets represent the same instant.
        let session = SessionState::resolve(
            Some("valid"),
            Some(record("2026-11-09T05:00:01-07:00")),
            now,
        );
        assert!(
            matches!(session, SessionState::Authenticated(s) if s.teacher.id == 1 && s.token == "valid")
        );
    }
}
