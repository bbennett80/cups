mod common;

use axum::http::StatusCode;
use common::{TestApp, form};

#[tokio::test]
async fn public_pages_static_files_and_student_cookie() {
    let app = TestApp::new().await;
    for uri in [
        "/",
        "/teacher/signup",
        "/teacher/signin",
        "/biology-101",
        "/static/styles.css",
        "/static/favicon.ico",
        "/static/cups.jpg",
        "/static/app.mjs",
        "/static/state-machines.mjs",
    ] {
        let reply = app.request("GET", uri, None, "").await;
        assert_eq!(reply.status, StatusCode::OK, "{uri}: {}", reply.body);
    }
    let reply = app.request("GET", "/biology-101", None, "").await;
    let cookie = reply.cookie();
    assert!(cookie.starts_with("student_id="));
    assert!(reply.body.contains("data-page=\"student\""));
    assert!(reply.body.contains("role=\"status\""));
    assert!(
        reply
            .body
            .contains("data-color=\"green\" aria-pressed=\"false\" disabled")
    );
    assert!(reply.body.contains("/static/app.mjs?v="));
    let reply = app
        .request("GET", "/chemistry-101", Some(&cookie), "")
        .await;
    assert_eq!(reply.cookie(), cookie);
    let reply = app.request("GET", "/%3Cscript%3E", None, "").await;
    assert!(reply.body.contains("&lt;script&gt;"));
    let reply = app.request("GET", "/static/missing.css", None, "").await;
    assert_eq!(reply.status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn signup_validation_and_signin_errors() {
    let app = TestApp::new().await;
    for (email, password, confirm, expected) in [
        (
            "invalid",
            "testing123",
            "testing123",
            "Enter a valid email.",
        ),
        (
            "a@test.com",
            "short",
            "short",
            "Password must be at least 8 characters.",
        ),
        (
            "a@test.com",
            "testing123",
            "different",
            "Passwords do not match.",
        ),
    ] {
        let reply = app
            .request(
                "POST",
                "/teacher/signup",
                None,
                &form(&[
                    ("email", email),
                    ("password", password),
                    ("password_confirm", confirm),
                    ("next_path", "/biology-101/teacher"),
                ]),
            )
            .await;
        assert_eq!(reply.status, StatusCode::BAD_REQUEST);
        assert!(reply.body.contains(expected));
        assert!(
            reply
                .body
                .replace("&#x2f;", "/")
                .contains("value=\"/biology-101/teacher\""),
            "{}",
            reply.body
        );
    }
    app.signup("teacher@test.com").await;
    let reply = app
        .request(
            "POST",
            "/teacher/signup",
            None,
            &form(&[
                ("email", "teacher@test.com"),
                ("password", "testing123"),
                ("password_confirm", "testing123"),
            ]),
        )
        .await;
    assert_eq!(reply.status, StatusCode::BAD_REQUEST);
    assert!(reply.body.contains("already exists"));
    for email in ["teacher@test.com", "unknown@test.com"] {
        let reply = app
            .request(
                "POST",
                "/teacher/signin",
                None,
                &form(&[("email", email), ("password", "wrong")]),
            )
            .await;
        assert_eq!(reply.status, StatusCode::UNAUTHORIZED);
        assert!(reply.body.contains("Invalid email or password."));
    }
}

#[tokio::test]
async fn account_class_ownership_and_logout_lifecycle() {
    let app = TestApp::new().await;
    let cookie = app.signup("  TEACHER@Test.COM  ").await;
    let reply = app
        .request("GET", "/teacher/dashboard", Some(&cookie), "")
        .await;
    assert_eq!(reply.status, StatusCode::OK);
    assert!(reply.body.contains("teacher@test.com"));
    assert!(reply.body.contains("You have no classes yet."));
    assert_eq!(
        app.create_class(&cookie, " Biology-101 ").await.status,
        StatusCode::SEE_OTHER
    );
    let reply = app
        .request("GET", "/teacher/dashboard", Some(&cookie), "")
        .await;
    assert!(
        reply
            .body
            .replace("&#x2f;", "/")
            .contains("data-student-url=\"http://cups.test/biology-101\"")
    );
    assert_eq!(
        app.request("GET", "/biology-101/teacher", Some(&cookie), "")
            .await
            .status,
        StatusCode::OK
    );
    for invalid in ["x", "ab", "-abc", "abc-", "no spaces"] {
        assert_eq!(
            app.create_class(&cookie, invalid).await.status,
            StatusCode::BAD_REQUEST
        );
    }
    let other = app.signup("other@test.com").await;
    assert_eq!(
        app.create_class(&other, "biology-101").await.status,
        StatusCode::CONFLICT
    );
    let denied = app
        .request("GET", "/biology-101/teacher", Some(&other), "")
        .await;
    assert_eq!(denied.location(), "/teacher/dashboard");
    let denied = app.request("GET", "/biology-101/teacher", None, "").await;
    assert_eq!(
        denied.location(),
        "/teacher/signin?next_path=%2Fbiology-101%2Fteacher"
    );
    let denied = app.create_class("", "biology-102").await;
    assert_eq!(
        denied.location(),
        "/teacher/signin?next_path=%2Fteacher%2Fdashboard"
    );
    let reply = app
        .request(
            "GET",
            "/teacher/signup?next_path=%2Fbiology-101%2Fteacher",
            Some(&cookie),
            "",
        )
        .await;
    assert_eq!(reply.location(), "/biology-101/teacher");
    let signed_in = app
        .request(
            "POST",
            "/teacher/signin",
            None,
            &form(&[
                ("email", "TEACHER@test.com"),
                ("password", "testing123"),
                ("next_path", "/biology-101/teacher"),
            ]),
        )
        .await;
    assert_eq!(signed_in.status, StatusCode::SEE_OTHER);
    assert_eq!(signed_in.location(), "/biology-101/teacher");
    let set_cookie = signed_in.headers["set-cookie"].to_str().unwrap();
    for attribute in ["HttpOnly", "SameSite=Lax", "Path=/", "Max-Age=2592000"] {
        assert!(set_cookie.contains(attribute));
    }
    let logged_out = app
        .request("POST", "/teacher/logout", Some(&cookie), "")
        .await;
    assert_eq!(logged_out.location(), "/");
    assert!(
        logged_out.headers["set-cookie"]
            .to_str()
            .unwrap()
            .contains("Max-Age=0")
    );
    let denied = app
        .request("GET", "/teacher/dashboard", Some(&cookie), "")
        .await;
    assert_eq!(denied.status, StatusCode::SEE_OTHER);
}

#[tokio::test]
async fn legacy_python_passwords_sessions_and_expiration() {
    let app = TestApp::new().await;
    // Generated with Python hashlib.pbkdf2_hmac('sha256', ..., iterations=200_000).
    let hash = "pbkdf2_sha256$200000$00112233445566778899aabbccddeeff$2c161abb1289416c0acbb5012c81cd192a6c5f064de887c6b96b734197813cb5";
    sqlx::query("INSERT INTO teachers (id, email, password_hash, created_at) VALUES (1, 'legacy@test.com', ?, '2026-01-01T00:00:00+00:00')")
        .bind(hash).execute(&app.db).await.unwrap();
    sqlx::query("INSERT INTO teacher_sessions (token, teacher_id, created_at, expires_at) VALUES ('legacy-token', 1, '2026-01-01T00:00:00+00:00', '2099-01-01T00:00:00+00:00')")
        .execute(&app.db).await.unwrap();
    let reply = app
        .request(
            "GET",
            "/teacher/dashboard",
            Some("teacher_session=legacy-token"),
            "",
        )
        .await;
    assert_eq!(reply.status, StatusCode::OK);
    let reply = app
        .request(
            "POST",
            "/teacher/signin",
            None,
            &form(&[
                ("email", "legacy@test.com"),
                ("password", "legacy-password"),
            ]),
        )
        .await;
    assert_eq!(reply.status, StatusCode::SEE_OTHER);
    sqlx::query("UPDATE teacher_sessions SET expires_at = '2000-01-01T00:00:00+00:00'")
        .execute(&app.db)
        .await
        .unwrap();
    let reply = app
        .request(
            "GET",
            "/teacher/dashboard",
            Some("teacher_session=legacy-token"),
            "",
        )
        .await;
    assert_eq!(reply.status, StatusCode::SEE_OTHER);
    sqlx::query("UPDATE teachers SET password_hash = 'pbkdf2_sha256$bad$salt$digest'")
        .execute(&app.db)
        .await
        .unwrap();
    let reply = app
        .request(
            "POST",
            "/teacher/signin",
            None,
            &form(&[
                ("email", "legacy@test.com"),
                ("password", "legacy-password"),
            ]),
        )
        .await;
    assert_eq!(reply.status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn redirects_stay_on_site() {
    let app = TestApp::new().await;
    let cookie = app.signup("teacher@test.com").await;
    for next in [
        "https://evil.test",
        "//evil.test",
        "/\\evil.test",
        "/\r\nevil",
    ] {
        let reply = app
            .request(
                "GET",
                &format!("/teacher/signin?next_path={}", urlencoding::encode(next)),
                Some(&cookie),
                "",
            )
            .await;
        assert_eq!(reply.location(), "/teacher/dashboard");
    }
}

#[tokio::test]
async fn rejected_forms_preserve_input_and_encode_nested_return_paths() {
    let app = TestApp::new().await;
    let next = "/biology-101/teacher?filter=red&sort=name";
    let rejected = app
        .request(
            "POST",
            "/teacher/signup",
            None,
            &form(&[
                ("email", "teacher@test.com"),
                ("password", "short"),
                ("password_confirm", "short"),
                ("next_path", next),
            ]),
        )
        .await;
    assert_eq!(rejected.status, StatusCode::BAD_REQUEST);
    assert!(rejected.body.contains("data-state=\"rejected\""));
    assert!(rejected.body.contains("value=\"teacher@test.com\""));
    assert!(rejected.body.contains("role=\"alert\""));
    assert!(!rejected.body.contains("value=\"short\""));
    assert!(
        rejected.body.contains("filter%3Dred%26sort%3Dname"),
        "{}",
        rejected.body
    );
    let cookie = app.signup("teacher@test.com").await;
    let rejected = app.create_class(&cookie, "invalid class").await;
    assert!(rejected.body.contains("value=\"invalid class\""));
    assert!(rejected.body.contains("data-state=\"rejected\""));
}

#[tokio::test]
async fn session_failure_rolls_back_account_creation_and_allows_retry() {
    let app = TestApp::new().await;
    sqlx::raw_sql("CREATE TRIGGER fail_session BEFORE INSERT ON teacher_sessions BEGIN SELECT RAISE(ABORT, 'test failure'); END;")
        .execute(&app.db).await.unwrap();
    let body = form(&[
        ("email", "retry@test.com"),
        ("password", "testing123"),
        ("password_confirm", "testing123"),
    ]);
    let failed = app.request("POST", "/teacher/signup", None, &body).await;
    assert_eq!(failed.status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(!failed.headers.contains_key("set-cookie"));
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM teachers")
        .fetch_one(&app.db)
        .await
        .unwrap();
    assert_eq!(count, 0);
    sqlx::query("DROP TRIGGER fail_session")
        .execute(&app.db)
        .await
        .unwrap();
    let retried = app.request("POST", "/teacher/signup", None, &body).await;
    assert_eq!(retried.status, StatusCode::SEE_OTHER);
    assert!(retried.cookie().starts_with("teacher_session="));
}

#[tokio::test]
async fn concurrent_signups_have_one_account_and_one_session() {
    let app = TestApp::new().await;
    let body = form(&[
        ("email", "duplicate@test.com"),
        ("password", "testing123"),
        ("password_confirm", "testing123"),
    ]);
    let (first, second) = tokio::join!(
        app.request("POST", "/teacher/signup", None, &body),
        app.request("POST", "/teacher/signup", None, &body),
    );
    let mut statuses = [first.status.as_u16(), second.status.as_u16()];
    statuses.sort();
    assert_eq!(statuses, [303, 400]);
    for table in ["teachers", "teacher_sessions"] {
        let count: i64 = sqlx::query_scalar(&format!("SELECT COUNT(*) FROM {table}"))
            .fetch_one(&app.db)
            .await
            .unwrap();
        assert_eq!(count, 1);
    }
}
