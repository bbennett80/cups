mod auth;
mod classroom;
mod db;
mod forms;
mod http;
mod realtime;

use std::{path::Path, sync::Arc, time::UNIX_EPOCH};

use anyhow::Context;
use axum::{
    Router,
    routing::{get, post},
};
use minijinja::Environment;
use socketioxide::{SocketIo, extract::SocketRef};
use sqlx::SqlitePool;
use tokio::sync::Mutex;
use tower_http::{services::ServeDir, trace::TraceLayer};

#[derive(Clone)]
pub(crate) struct AppState {
    db: SqlitePool,
    templates: Arc<Environment<'static>>,
    live: Arc<Mutex<realtime::LiveState>>,
    auth_gate: Arc<Mutex<()>>,
    style_version: u64,
}

/// Build the application with the existing Python-compatible SQLite database.
pub async fn app(db_path: &Path, asset_root: &Path) -> anyhow::Result<Router> {
    let db = db::open(db_path).await?;
    let mut templates = Environment::new();
    templates.add_filter("urlencode", |value: String| {
        urlencoding::encode(&value).into_owned()
    });
    templates.set_loader(minijinja::path_loader(asset_root.join("templates")));
    // Fail at startup if a required template is missing or invalid.
    for name in [
        "layout.html",
        "howto.html",
        "student.html",
        "teacher.html",
        "teacher_dashboard.html",
        "teacher_signin.html",
        "teacher_signup.html",
    ] {
        templates
            .get_template(name)
            .with_context(|| format!("loading {name}"))?;
    }
    let style_version = asset_root
        .join("static/styles.css")
        .metadata()
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    let state = AppState {
        db,
        templates: Arc::new(templates),
        live: Arc::default(),
        auth_gate: Arc::default(),
        style_version,
    };
    realtime::start_expiry_clock(&state.live);
    let (layer, io) = SocketIo::new_layer();
    let socket_state = state.clone();
    io.ns("/", move |socket: SocketRef| {
        realtime::connect(socket, socket_state.clone())
    });

    Ok(Router::new()
        .route("/", get(http::home))
        .route("/teacher/signup", get(http::signup_page).post(http::signup))
        .route("/teacher/signin", get(http::signin_page).post(http::signin))
        .route("/teacher/logout", post(http::logout))
        .route("/teacher/dashboard", get(http::dashboard))
        .route("/teacher/classes", post(http::create_class))
        .route("/{class_id}", get(http::student))
        .route("/{class_id}/teacher", get(http::teacher))
        .nest_service("/static", ServeDir::new(asset_root.join("static")))
        .with_state(state)
        .layer(layer)
        .layer(TraceLayer::new_for_http()))
}
