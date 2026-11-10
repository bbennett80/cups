use axum_extra::extract::cookie::{Cookie, CookieJar, SameSite};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use chrono::{Duration, SecondsFormat, Utc};
use pbkdf2::pbkdf2_hmac;
use rand::{RngCore, rngs::OsRng};
use sha2::Sha256;
use sqlx::SqlitePool;
use subtle::ConstantTimeEq;

use crate::db::{self, Teacher};

pub(crate) const COOKIE: &str = "teacher_session";
pub(crate) const DASHBOARD: &str = "/teacher/dashboard";
const SESSION_DAYS: i64 = 30;
const ITERATIONS: u32 = 200_000;

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
    match jar.get(COOKIE) {
        Some(cookie) => db::teacher_by_token(db, cookie.value()).await,
        None => Ok(None),
    }
}

pub(crate) async fn session(
    db: &SqlitePool,
    teacher_id: i64,
    jar: CookieJar,
) -> Result<CookieJar, sqlx::Error> {
    let token = random_token();
    let now = Utc::now();
    let expires = now + Duration::days(SESSION_DAYS);
    sqlx::query("INSERT INTO teacher_sessions (token, teacher_id, created_at, expires_at) VALUES (?, ?, ?, ?)")
        .bind(&token).bind(teacher_id)
        .bind(now.to_rfc3339_opts(SecondsFormat::Micros, false))
        .bind(expires.to_rfc3339_opts(SecondsFormat::Micros, false))
        .execute(db).await?;
    Ok(jar.add(
        Cookie::build((COOKIE, token))
            .path("/")
            .http_only(true)
            .same_site(SameSite::Lax)
            .max_age(time::Duration::days(SESSION_DAYS))
            .build(),
    ))
}
