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

Templates and static files are loaded from disk; keep `templates/` and `static/` alongside the application, or set `ASSET_ROOT`. `Cargo.lock` pins the dependency versions.

### Dependencies

This Rust branch does not need `.venv`, a Python interpreter, pip packages, or a Python test runner. Cargo manages backend dependencies and writes build artifacts to `target/`. Node is needed only to run the browser tests; there is no npm install or frontend build step.

An existing `.venv/` or `__pycache__/` is left over from the Python app and is ignored by Git. You can remove those directories if you no longer use the `fastapi-port` branch; removing them does not affect the Rust app or `teachers.db`.

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
| Server → teacher | `teacher_access_ended` | `{reason}` when the live session expires or is revoked |

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

`src/http.rs` handles pages and forms, `src/auth.rs` handles passwords and sessions, and `src/db.rs` and `src/schema.sql` handle persistence. `src/classroom.rs` is the pure classroom state machine; `src/realtime.rs` adapts Socket.IO commands and effects. `static/state-machines.mjs` defines browser transitions and `static/app.mjs` renders them. `src/main.rs` configures and serves the application.

## State and reliability in Rust

The refactor makes application behavior explicit through Rust enums and guarded transitions:

- **Typed classroom behavior:** `StudentState`, `TeacherState`, and `Feedback` define participation and subscription states. Commands enter through `Classroom::transition(event, now)` and return typed outcomes and effects. Duplicate registration preserves feedback, superseded tabs cannot regain control, and an old tab disconnecting cannot remove its replacement.
- **Separation of behavior and I/O:** the classroom model performs no network or database operations and reads no clock. The Socketioxide adapter authorizes requests, applies transitions under a Tokio mutex, and delivers effects. This makes event ordering and expiry boundaries reproducible in tests.
- **Consistent authentication:** `SessionState` distinguishes missing, invalid, expired, and authenticated sessions. `ClassAccess` supplies the same ownership decision to HTTP pages and Socket.IO subscriptions. An authorization gate serializes subscription changes with logout, and expiry checks stop idle sessions from retaining live subscriptions.
- **Atomic account creation:** SQLx saves a new teacher and their initial session in one SQLite transaction. A session failure rolls back the account, and the session cookie is issued only after commit. Users can retry without encountering a partially created account.
- **Responsive request handling:** password hashing and verification run through Tokio's `spawn_blocking`, keeping PBKDF2 work off asynchronous request workers. SQLite access uses SQLx's asynchronous pool.

The browser uses matching explicit reducers for connection, submission, theme, and QR states. Students see server-confirmed feedback, teachers see when counts are stale, and rejected forms retain email or class input. Submit buttons prevent duplicate submissions, and the QR modal handles loading failures and keyboard focus.

For developers and agents, the [state-machine map](docs/state-machines.md) inventories 18 stateful flows with diagrams, guards, invariants, protocol contracts, and change recipes. Typed events and effects give backend changes a clear entry point; browser `data-state` attributes make UI state inspectable. No state-machine framework or additional runtime service is required.

## Verify

```bash
cargo fmt --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
node --test tests/browser/*.test.mjs
```

Tests use temporary databases and cover HTTP pages and assets, authentication errors, ownership, redirects, legacy Python passwords/sessions, expiration, logout, and Socket.IO polling/WebSocket flows including colors, counts, class switching, and old-tab deactivation. The WebSocket test binds an ephemeral loopback port. Tests do not modify your local `teachers.db`.

`bash scripts/check.sh` runs all checks. Node is required only for browser-state tests; the application still runs with Rust alone. Model and browser tests also cover reconnect/acknowledgement races, idempotent transitions, terminal states, session expiry, atomic signup rollback, and concurrent signups.
