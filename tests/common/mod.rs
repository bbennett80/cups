#![allow(dead_code)]

use std::path::Path;

use axum::{
    Router,
    body::Body,
    http::{HeaderMap, Request, StatusCode},
};
use http_body_util::BodyExt;
use sqlx::{SqlitePool, sqlite::SqliteConnectOptions};
use tempfile::TempDir;
use tower::ServiceExt;

pub struct TestApp {
    pub router: Router,
    pub db: SqlitePool,
    _dir: TempDir,
}

pub struct Reply {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: String,
}

impl Reply {
    pub fn cookie(&self) -> String {
        self.headers["set-cookie"]
            .to_str()
            .unwrap()
            .split(';')
            .next()
            .unwrap()
            .to_owned()
    }

    pub fn location(&self) -> &str {
        self.headers["location"].to_str().unwrap()
    }
}

pub fn form(fields: &[(&str, &str)]) -> String {
    fields
        .iter()
        .map(|(key, value)| {
            format!(
                "{}={}",
                urlencoding::encode(key),
                urlencoding::encode(value)
            )
        })
        .collect::<Vec<_>>()
        .join("&")
}

impl TestApp {
    pub async fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("teachers.db");
        let router = cups::app(&path, Path::new(env!("CARGO_MANIFEST_DIR")))
            .await
            .unwrap();
        let db = SqlitePool::connect_with(SqliteConnectOptions::new().filename(&path))
            .await
            .unwrap();
        Self {
            router,
            db,
            _dir: dir,
        }
    }

    pub async fn request(
        &self,
        method: &str,
        uri: &str,
        cookie: Option<&str>,
        body: &str,
    ) -> Reply {
        let mut request = Request::builder()
            .method(method)
            .uri(uri)
            .header("host", "cups.test");
        if let Some(cookie) = cookie {
            request = request.header("cookie", cookie);
        }
        if method == "POST" {
            request = request.header("content-type", "application/x-www-form-urlencoded");
        }
        let response = self
            .router
            .clone()
            .oneshot(request.body(Body::from(body.to_owned())).unwrap())
            .await
            .unwrap();
        let (parts, body) = response.into_parts();
        Reply {
            status: parts.status,
            headers: parts.headers,
            body: String::from_utf8_lossy(&body.collect().await.unwrap().to_bytes()).into_owned(),
        }
    }

    pub async fn signup(&self, email: &str) -> String {
        let reply = self
            .request(
                "POST",
                "/teacher/signup",
                None,
                &form(&[
                    ("email", email),
                    ("password", "testing123"),
                    ("password_confirm", "testing123"),
                ]),
            )
            .await;
        assert_eq!(reply.status, StatusCode::SEE_OTHER, "{}", reply.body);
        reply.cookie()
    }

    pub async fn create_class(&self, cookie: &str, name: &str) -> Reply {
        self.request(
            "POST",
            "/teacher/classes",
            Some(cookie),
            &form(&[("class_name", name)]),
        )
        .await
    }
}
