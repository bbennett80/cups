use axum::{
    Form,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{Html, IntoResponse, Redirect, Response},
};
use axum_extra::extract::cookie::{Cookie, CookieJar};
use rand::{Rng, distributions::Alphanumeric};
use serde::Deserialize;
use serde_json::{Value, json};
use sqlx::Row;

use crate::{AppState, auth, db};

pub(crate) struct AppError(anyhow::Error);

impl<E: std::error::Error + Send + Sync + 'static> From<E> for AppError {
    fn from(error: E) -> Self {
        Self(error.into())
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        tracing::error!(error = ?self.0, "request failed");
        (StatusCode::INTERNAL_SERVER_ERROR, "Internal server error").into_response()
    }
}

type Result = std::result::Result<Response, AppError>;

fn base_url(headers: &HeaderMap) -> String {
    let host = headers
        .get("host")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("localhost");
    let scheme = headers
        .get("x-forwarded-proto")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.split(',').next())
        .map(str::trim)
        .filter(|s| *s == "http" || *s == "https")
        .unwrap_or("http");
    format!("{scheme}://{host}/")
}

async fn render(
    state: &AppState,
    jar: &CookieJar,
    headers: &HeaderMap,
    template: &str,
    mut context: Value,
    status: StatusCode,
) -> Result {
    context["current_teacher"] = json!(auth::current_teacher(&state.db, jar).await?);
    context["style_version"] = json!(state.style_version);
    context["request"] = json!({"base_url": base_url(headers)});
    let html = state.templates.get_template(template)?.render(context)?;
    Ok((status, Html(html)).into_response())
}

fn signin_redirect(next: &str) -> Response {
    Redirect::to(&format!(
        "/teacher/signin?next_path={}",
        urlencoding::encode(next)
    ))
    .into_response()
}

#[derive(Default, Deserialize)]
pub(crate) struct AuthQuery {
    next_path: Option<String>,
}

pub(crate) async fn home(
    State(state): State<AppState>,
    jar: CookieJar,
    headers: HeaderMap,
) -> Result {
    render(
        &state,
        &jar,
        &headers,
        "howto.html",
        json!({"url": base_url(&headers)}),
        StatusCode::OK,
    )
    .await
}

async fn auth_page(
    state: &AppState,
    jar: &CookieJar,
    headers: &HeaderMap,
    query: AuthQuery,
    template: &str,
) -> Result {
    let next = auth::safe_next_path(query.next_path.as_deref());
    if auth::current_teacher(&state.db, jar).await?.is_some() {
        return Ok(Redirect::to(next).into_response());
    }
    render(
        state,
        jar,
        headers,
        template,
        json!({"next_path": next}),
        StatusCode::OK,
    )
    .await
}

pub(crate) async fn signup_page(
    State(state): State<AppState>,
    jar: CookieJar,
    headers: HeaderMap,
    Query(query): Query<AuthQuery>,
) -> Result {
    auth_page(&state, &jar, &headers, query, "teacher_signup.html").await
}

pub(crate) async fn signin_page(
    State(state): State<AppState>,
    jar: CookieJar,
    headers: HeaderMap,
    Query(query): Query<AuthQuery>,
) -> Result {
    auth_page(&state, &jar, &headers, query, "teacher_signin.html").await
}

#[derive(Deserialize)]
pub(crate) struct Signup {
    email: String,
    password: String,
    password_confirm: String,
    next_path: Option<String>,
}

pub(crate) async fn signup(
    State(state): State<AppState>,
    jar: CookieJar,
    headers: HeaderMap,
    Form(form): Form<Signup>,
) -> Result {
    let next = auth::safe_next_path(form.next_path.as_deref());
    let email = form.email.trim().to_lowercase();
    let error = if email.is_empty() || !email.contains('@') {
        Some("Enter a valid email.")
    } else if form.password.chars().count() < 8 {
        Some("Password must be at least 8 characters.")
    } else if form.password != form.password_confirm {
        Some("Passwords do not match.")
    } else {
        None
    };
    if let Some(error) = error {
        return render(
            &state,
            &jar,
            &headers,
            "teacher_signup.html",
            json!({"next_path": next, "error": error}),
            StatusCode::BAD_REQUEST,
        )
        .await;
    }
    // PBKDF2 is CPU-intensive: keep it off Tokio's request worker threads.
    let hash = tokio::task::spawn_blocking(move || auth::hash_password(&form.password)).await?;
    let insert =
        sqlx::query("INSERT INTO teachers (email, password_hash, created_at) VALUES (?, ?, ?)")
            .bind(email)
            .bind(hash)
            .bind(db::now())
            .execute(&state.db)
            .await;
    let teacher_id =
        match insert {
            Ok(result) => result.last_insert_rowid(),
            Err(sqlx::Error::Database(error)) if error.is_unique_violation() => {
                return render(&state, &jar, &headers, "teacher_signup.html", json!({
                "next_path": next, "error": "A teacher account with this email already exists."
            }), StatusCode::BAD_REQUEST).await;
            }
            Err(error) => return Err(error.into()),
        };
    let jar = auth::session(&state.db, teacher_id, jar).await?;
    Ok((jar, Redirect::to(next)).into_response())
}

#[derive(Deserialize)]
pub(crate) struct Signin {
    email: String,
    password: String,
    next_path: Option<String>,
}

pub(crate) async fn signin(
    State(state): State<AppState>,
    jar: CookieJar,
    headers: HeaderMap,
    Form(form): Form<Signin>,
) -> Result {
    let next = auth::safe_next_path(form.next_path.as_deref());
    let row = sqlx::query("SELECT id, password_hash FROM teachers WHERE email = ?")
        .bind(form.email.trim().to_lowercase())
        .fetch_optional(&state.db)
        .await?;
    if let Some(row) = row {
        let id: i64 = row.get("id");
        let hash: String = row.get("password_hash");
        if tokio::task::spawn_blocking(move || auth::verify_password(&form.password, &hash)).await?
        {
            let jar = auth::session(&state.db, id, jar).await?;
            return Ok((jar, Redirect::to(next)).into_response());
        }
    }
    render(
        &state,
        &jar,
        &headers,
        "teacher_signin.html",
        json!({
            "next_path": next, "error": "Invalid email or password."
        }),
        StatusCode::UNAUTHORIZED,
    )
    .await
}

pub(crate) async fn logout(State(state): State<AppState>, jar: CookieJar) -> Result {
    if let Some(cookie) = jar.get(auth::COOKIE) {
        sqlx::query("DELETE FROM teacher_sessions WHERE token = ?")
            .bind(cookie.value())
            .execute(&state.db)
            .await?;
        state
            .live
            .lock()
            .await
            .remove_teacher_session(cookie.value());
    }
    let jar = jar.remove(Cookie::build(auth::COOKIE).path("/").build());
    Ok((jar, Redirect::to("/")).into_response())
}

#[derive(Default, Deserialize)]
pub(crate) struct DashboardQuery {
    error: Option<String>,
}

pub(crate) async fn dashboard(
    State(state): State<AppState>,
    jar: CookieJar,
    headers: HeaderMap,
    Query(query): Query<DashboardQuery>,
) -> Result {
    let Some(teacher) = auth::current_teacher(&state.db, &jar).await? else {
        return Ok(signin_redirect(auth::DASHBOARD));
    };
    render(
        &state,
        &jar,
        &headers,
        "teacher_dashboard.html",
        json!({
            "classes": db::classes(&state.db, teacher.id).await?, "error": query.error
        }),
        StatusCode::OK,
    )
    .await
}

#[derive(Deserialize)]
pub(crate) struct CreateClass {
    class_name: String,
}

fn valid_class_name(name: &str) -> bool {
    (3..=40).contains(&name.len())
        && name
            .bytes()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-')
        && !name.starts_with('-')
        && !name.ends_with('-')
}

pub(crate) async fn create_class(
    State(state): State<AppState>,
    jar: CookieJar,
    headers: HeaderMap,
    Form(form): Form<CreateClass>,
) -> Result {
    let Some(teacher) = auth::current_teacher(&state.db, &jar).await? else {
        return Ok(signin_redirect(auth::DASHBOARD));
    };
    let name = form.class_name.trim().to_lowercase();
    let (status, error) = if !valid_class_name(&name) {
        (
            StatusCode::BAD_REQUEST,
            "Class URL must be 3-40 chars, lowercase letters/numbers/hyphens, and cannot start/end with a hyphen.",
        )
    } else {
        match sqlx::query(
            "INSERT INTO classes (teacher_id, class_name, created_at) VALUES (?, ?, ?)",
        )
        .bind(teacher.id)
        .bind(&name)
        .bind(db::now())
        .execute(&state.db)
        .await
        {
            Ok(_) => return Ok(Redirect::to(auth::DASHBOARD).into_response()),
            Err(sqlx::Error::Database(error)) if error.is_unique_violation() => (
                StatusCode::CONFLICT,
                "That class URL is already taken. Choose another name.",
            ),
            Err(error) => return Err(error.into()),
        }
    };
    render(
        &state,
        &jar,
        &headers,
        "teacher_dashboard.html",
        json!({
            "classes": db::classes(&state.db, teacher.id).await?, "error": error
        }),
        status,
    )
    .await
}

pub(crate) async fn student(
    State(state): State<AppState>,
    jar: CookieJar,
    headers: HeaderMap,
    Path(class_id): Path<String>,
) -> Result {
    let student_id = jar
        .get("student_id")
        .map(|c| c.value().to_owned())
        .unwrap_or_else(|| {
            rand::thread_rng()
                .sample_iter(&Alphanumeric)
                .take(12)
                .map(char::from)
                .collect()
        });
    let response = render(
        &state,
        &jar,
        &headers,
        "student.html",
        json!({
            "timestamp": chrono::Utc::now().timestamp_micros() as f64 / 1_000_000.0,
            "class_id": class_id
        }),
        StatusCode::OK,
    )
    .await?;
    let jar = jar.add(Cookie::build(("student_id", student_id)).path("/").build());
    Ok((jar, response).into_response())
}

pub(crate) async fn teacher(
    State(state): State<AppState>,
    jar: CookieJar,
    headers: HeaderMap,
    Path(class_id): Path<String>,
) -> Result {
    let Some(teacher) = auth::current_teacher(&state.db, &jar).await? else {
        return Ok(signin_redirect(&format!("/{class_id}/teacher")));
    };
    if !db::owns_class(&state.db, teacher.id, &class_id).await? {
        return Ok(Redirect::to(auth::DASHBOARD).into_response());
    }
    let stats = state.live.lock().await.stats(&class_id);
    render(
        &state,
        &jar,
        &headers,
        "teacher.html",
        json!(stats),
        StatusCode::OK,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::valid_class_name;

    #[test]
    fn class_names_match_the_form_contract() {
        for name in ["biology-101", "abc", "a-1", &"a".repeat(40)] {
            assert!(valid_class_name(name), "{name}");
        }
        for name in [
            "a",
            "ab",
            "-abc",
            "abc-",
            "ABC",
            "a b",
            "a/b",
            "éab",
            &"a".repeat(41),
        ] {
            assert!(!valid_class_name(name), "{name}");
        }
    }
}
