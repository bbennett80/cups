use std::{path::Path, time::Duration};

use chrono::{SecondsFormat, Utc};
use serde::Serialize;
use sqlx::{
    Row, SqlitePool,
    sqlite::{SqliteConnectOptions, SqlitePoolOptions},
};

pub(crate) async fn open(path: &Path) -> anyhow::Result<SqlitePool> {
    let db = SqlitePoolOptions::new()
        .max_connections(5)
        .connect_with(
            SqliteConnectOptions::new()
                .filename(path)
                .create_if_missing(true)
                .foreign_keys(true)
                .busy_timeout(Duration::from_secs(5)),
        )
        .await?;
    // Same schema as fastapi-port; existing databases require no migration.
    sqlx::raw_sql(include_str!("schema.sql"))
        .execute(&db)
        .await?;
    Ok(db)
}

pub(crate) fn now() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Micros, false)
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Teacher {
    pub id: i64,
    pub email: String,
}

#[derive(Serialize)]
pub(crate) struct Class {
    pub class_name: String,
    pub created_at: String,
}

pub(crate) struct SessionRecord {
    pub teacher: Teacher,
    pub expires_at: String,
}

pub(crate) async fn session_by_token(
    db: &SqlitePool,
    token: &str,
) -> Result<Option<SessionRecord>, sqlx::Error> {
    let row = sqlx::query(
        "SELECT t.id, t.email, ts.expires_at FROM teacher_sessions ts JOIN teachers t ON t.id = ts.teacher_id
         WHERE ts.token = ?",
    )
    .bind(token)
    .fetch_optional(db)
    .await?;
    Ok(row.map(|r| SessionRecord {
        teacher: Teacher {
            id: r.get("id"),
            email: r.get("email"),
        },
        expires_at: r.get("expires_at"),
    }))
}

pub(crate) async fn classes(db: &SqlitePool, teacher_id: i64) -> Result<Vec<Class>, sqlx::Error> {
    let rows = sqlx::query(
        "SELECT class_name, created_at FROM classes WHERE teacher_id = ? ORDER BY class_name ASC",
    )
    .bind(teacher_id)
    .fetch_all(db)
    .await?;
    Ok(rows
        .into_iter()
        .map(|r| Class {
            class_name: r.get("class_name"),
            created_at: r.get("created_at"),
        })
        .collect())
}

pub(crate) async fn owns_class(
    db: &SqlitePool,
    teacher_id: i64,
    class_id: &str,
) -> Result<bool, sqlx::Error> {
    Ok(
        sqlx::query("SELECT 1 FROM classes WHERE teacher_id = ? AND class_name = ?")
            .bind(teacher_id)
            .bind(class_id)
            .fetch_optional(db)
            .await?
            .is_some(),
    )
}
