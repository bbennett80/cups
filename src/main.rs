use std::{env, path::PathBuf};

use anyhow::Context;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "cups=info,tower_http=info".into()),
        )
        .init();
    let host = env::var("HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port = env::var("PORT").unwrap_or_else(|_| "8000".into());
    let db_path = PathBuf::from(env::var("DB_PATH").unwrap_or_else(|_| "teachers.db".into()));
    let asset_root = PathBuf::from(env::var("ASSET_ROOT").unwrap_or_else(|_| ".".into()));
    let app = cups::app(&db_path, &asset_root).await?;
    let listener = tokio::net::TcpListener::bind(format!("{host}:{port}"))
        .await
        .context("binding HTTP listener")?;
    tracing::info!(address = %listener.local_addr()?, "Runnercups listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown())
        .await?;
    Ok(())
}

async fn shutdown() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("install Ctrl+C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! { _ = ctrl_c => {}, _ = terminate => {} }
}
