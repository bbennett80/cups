# Runnercups

Rust/Axum classroom feedback app, ported from the `fastapi-port` branch.

Teachers create unique class URLs, students select a status color (green/yellow/red), and teacher views update live. The existing UI includes light/dark mode, QR code sharing, teacher accounts, and a class dashboard.

## Run

Install a current stable Rust toolchain (tested with Rust 1.96), then run from the project root:

```bash
cargo run --locked
```

Open <http://127.0.0.1:8000>. Create a teacher account, create a class such as `biology-101`, and share its student link. Keep `/biology-101/teacher` open to see feedback.

For a release build:

```bash
cargo build --release --locked
./target/release/cups
```

Templates and static files are loaded from disk; keep `templates/` and `static/` alongside the application, or set `ASSET_ROOT`. Python is no longer required. `Cargo.lock` pins the dependency versions.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `HOST` | `127.0.0.1` | Bind address; use `0.0.0.0` to listen on all interfaces |
| `PORT` | `8000` | HTTP port |
| `DB_PATH` | `teachers.db` | SQLite database file, created if missing |
| `ASSET_ROOT` | `.` | Directory containing `templates/` and `static/` |
| `RUST_LOG` | `cups=info,tower_http=info` | Tracing filter |

For example:

```bash
HOST=0.0.0.0 PORT=3000 DB_PATH=/var/lib/cups/teachers.db cargo run --release --locked
```

The database's parent directory must already exist and be writable. Behind an HTTPS reverse proxy, preserve `Host`, set `X-Forwarded-Proto: https`, and forward `/socket.io/` with support for both polling and WebSocket upgrades. The proxy should overwrite incoming `X-Forwarded-Proto` headers. The server shuts down gracefully on Ctrl+C or SIGTERM.

## Compatibility with the FastAPI app

The Rust app opens the same `teachers.db` schema. Existing teacher accounts, classes, PBKDF2-SHA256 password hashes, and unexpired `teacher_session` cookies continue to work without migration. New password hashes retain Python's `pbkdf2_sha256$iterations$salt$digest` format and 200,000 iterations. Teacher session cookies remain HttpOnly, SameSite=Lax, and valid for 30 days.

The existing Socket.IO 4.x browser client and event format are preserved:

| Direction | Event | Arguments |
| --- | --- | --- |
| Student → server | `register_student` | `timestamp, class_id` (two arguments) |
| Student → server | `color_change` | `"green"`, `"yellow"`, `"red"`, or `"inactive"` |
| Teacher → server | `join_teacher_class` | `class_id` |
| Server → teacher | `teacher_stats` | `{class_id, student_count, active_student_count, color2frac}` |
| Server → student | `deactivate_old_tabs` | `{student_id, timestamp}` |

Both HTTP polling and WebSocket transports work. Teacher subscriptions require a valid session and class ownership. Signing out also removes that session's live subscriptions. A student's newest tab replaces their previous registration, resets their color to inactive, and updates both classes if they switch. Counts include each connected student cookie once, and color fractions use active students as the denominator. Invalid colors are ignored. Class names follow the UI's 3–40 character constraint: lowercase letters, numbers, and interior hyphens.

As before, live student state is in memory and resets on restart. Run one application process; multiple instances would need shared live state. Accounts, sessions, and classes persist in SQLite. The UI still loads the Socket.IO client from cdnjs and QR images from api.qrserver.com.

## Routes

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Usage overview |
| GET | `/{class_id}` | Student class page |
| GET, POST | `/teacher/signup` | Teacher registration |
| GET, POST | `/teacher/signin` | Teacher sign in |
| POST | `/teacher/logout` | Sign out |
| GET | `/teacher/dashboard` | Teacher's classes |
| POST | `/teacher/classes` | Create a globally unique class URL |
| GET | `/{class_id}/teacher` | Live teacher view (owner only) |
| GET | `/static/{path}` | Styles and images |
| GET, POST | `/socket.io/` | Socket.IO transport |

## Implementation

- [Axum](https://github.com/tokio-rs/axum) and Tokio: HTTP routes and runtime
- [Socketioxide](https://github.com/Totodore/socketioxide): Socket.IO server
- MiniJinja: existing Jinja-compatible HTML templates
- SQLx: asynchronous SQLite access
- Tower HTTP: static files and request tracing

`src/http.rs` handles pages and forms, `src/auth.rs` handles passwords and sessions, `src/db.rs` and `src/schema.sql` handle persistence, and `src/realtime.rs` manages live classroom state. `src/main.rs` configures and serves the application.

## Verify

```bash
cargo fmt --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
```

Tests use temporary databases and cover HTTP pages and assets, authentication errors, ownership, redirects, legacy Python passwords/sessions, expiration, logout, and Socket.IO polling/WebSocket flows including colors, counts, class switching, and old-tab deactivation. The WebSocket test binds an ephemeral loopback port. Tests do not modify your local `teachers.db`.
