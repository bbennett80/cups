mod common;

use std::{net::SocketAddr, time::Duration};

use axum::{body::Body, http::Request};
use common::TestApp;
use futures_util::{SinkExt, StreamExt};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tokio::{net::TcpStream, time::timeout};
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async,
    tungstenite::{Message, client::IntoClientRequest},
};
use tower::ServiceExt;

struct Client(WebSocketStream<MaybeTlsStream<TcpStream>>);

impl Client {
    async fn connect(address: SocketAddr, cookie: Option<&str>) -> Self {
        let mut request = format!("ws://{address}/socket.io/?EIO=4&transport=websocket")
            .into_client_request()
            .unwrap();
        if let Some(cookie) = cookie {
            request
                .headers_mut()
                .insert("cookie", cookie.parse().unwrap());
        }
        let (socket, _) = connect_async(request).await.unwrap();
        let mut client = Self(socket);
        assert!(client.packet().await.starts_with('0'));
        client.0.send(Message::Text("40".into())).await.unwrap();
        assert!(client.packet().await.starts_with("40"));
        client
    }

    async fn packet(&mut self) -> String {
        timeout(Duration::from_secs(3), async {
            loop {
                let message = self.0.next().await.expect("socket closed").unwrap();
                if let Message::Text(text) = message {
                    if text == "2" {
                        self.0.send(Message::Text("3".into())).await.unwrap();
                    } else {
                        return text.to_string();
                    }
                }
            }
        })
        .await
        .expect("timed out waiting for a Socket.IO packet")
    }

    async fn emit(&mut self, arguments: Value) {
        self.0
            .send(Message::Text(format!("42{arguments}").into()))
            .await
            .unwrap();
    }

    async fn event(&mut self, name: &str) -> Value {
        let packet = self.packet().await;
        assert!(packet.starts_with("42"), "{packet}");
        let event: Value = serde_json::from_str(&packet[2..]).unwrap();
        assert_eq!(event[0], name);
        event[1].clone()
    }

    async fn no_event(&mut self) {
        let packet = timeout(Duration::from_millis(100), self.packet()).await;
        assert!(
            packet.is_err(),
            "unauthorized or stale client received an event: {packet:?}"
        );
    }

    async fn disconnect(&mut self) {
        self.0.send(Message::Text("41".into())).await.unwrap();
        self.0.close(None).await.unwrap();
    }
}

fn assert_stats(stats: &Value, class: &str, total: u64, active: u64, fractions: [f64; 3]) {
    assert_eq!(stats["class_id"], class);
    assert_eq!(stats["student_count"], total);
    assert_eq!(stats["active_student_count"], active);
    for (color, expected) in ["green", "yellow", "red"].into_iter().zip(fractions) {
        assert_eq!(stats["color2frac"][color], expected, "{stats}");
    }
}

#[tokio::test]
async fn live_counts_colors_tabs_class_switching_and_authorization() {
    let app = TestApp::new().await;
    let owner = app.signup("owner@test.com").await;
    let other = app.signup("other@test.com").await;
    app.create_class(&owner, "biology-101").await;
    app.create_class(&owner, "chemistry-101").await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let router = app.router.clone();
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let mut teacher = Client::connect(address, Some(&owner)).await;
    teacher
        .emit(json!(["join_teacher_class", "biology-101"]))
        .await;
    assert_stats(
        &teacher.event("teacher_stats").await,
        "biology-101",
        0,
        0,
        [0.0; 3],
    );
    let mut chemistry = Client::connect(address, Some(&owner)).await;
    chemistry
        .emit(json!(["join_teacher_class", "chemistry-101"]))
        .await;
    assert_stats(
        &chemistry.event("teacher_stats").await,
        "chemistry-101",
        0,
        0,
        [0.0; 3],
    );

    let mut foreign = Client::connect(address, Some(&other)).await;
    foreign
        .emit(json!(["join_teacher_class", "biology-101"]))
        .await;
    foreign.no_event().await;
    let mut anonymous = Client::connect(address, None).await;
    anonymous
        .emit(json!(["join_teacher_class", "biology-101"]))
        .await;
    anonymous
        .emit(json!(["register_student", 1.0, "biology-101"]))
        .await;
    anonymous.no_event().await;
    teacher.no_event().await;

    let cookie1 = app.request("GET", "/biology-101", None, "").await.cookie();
    let mut student1 = Client::connect(address, Some(&cookie1)).await;
    student1
        .emit(json!(["register_student", 1.0, "biology-101"]))
        .await;
    assert_stats(
        &teacher.event("teacher_stats").await,
        "biology-101",
        1,
        0,
        [0.0; 3],
    );
    student1.emit(json!(["color_change", "green"])).await;
    assert_stats(
        &teacher.event("teacher_stats").await,
        "biology-101",
        1,
        1,
        [1.0, 0.0, 0.0],
    );

    let cookie2 = app.request("GET", "/biology-101", None, "").await.cookie();
    let mut student2 = Client::connect(address, Some(&cookie2)).await;
    student2
        .emit(json!(["register_student", 2.0, "biology-101"]))
        .await;
    assert_stats(
        &teacher.event("teacher_stats").await,
        "biology-101",
        2,
        1,
        [1.0, 0.0, 0.0],
    );
    student2.emit(json!(["color_change", "red"])).await;
    assert_stats(
        &teacher.event("teacher_stats").await,
        "biology-101",
        2,
        2,
        [0.5, 0.0, 0.5],
    );
    student2.emit(json!(["color_change", "purple"])).await;
    teacher.no_event().await;
    student2.emit(json!(["color_change", "yellow"])).await;
    assert_stats(
        &teacher.event("teacher_stats").await,
        "biology-101",
        2,
        2,
        [0.5, 0.5, 0.0],
    );

    let mut new_tab = Client::connect(address, Some(&cookie1)).await;
    new_tab
        .emit(json!(["register_student", 3.0, "biology-101"]))
        .await;
    let deactivate = student1.event("deactivate_old_tabs").await;
    assert_eq!(deactivate["timestamp"], 3.0);
    assert_eq!(
        deactivate["student_id"],
        cookie1.strip_prefix("student_id=").unwrap()
    );
    assert_stats(
        &teacher.event("teacher_stats").await,
        "biology-101",
        2,
        1,
        [0.0, 1.0, 0.0],
    );
    student1.emit(json!(["color_change", "green"])).await;
    teacher.no_event().await;
    // A stale tab reconnecting must not replace a newer tab.
    student1
        .emit(json!(["register_student", 1.0, "biology-101"]))
        .await;
    assert_eq!(
        student1.event("deactivate_old_tabs").await["timestamp"],
        3.0
    );
    teacher.no_event().await;

    new_tab.emit(json!(["color_change", "red"])).await;
    assert_stats(
        &teacher.event("teacher_stats").await,
        "biology-101",
        2,
        2,
        [0.0, 0.5, 0.5],
    );
    new_tab
        .emit(json!(["register_student", 4.0, "chemistry-101"]))
        .await;
    assert_stats(
        &teacher.event("teacher_stats").await,
        "biology-101",
        1,
        1,
        [0.0, 1.0, 0.0],
    );
    assert_stats(
        &chemistry.event("teacher_stats").await,
        "chemistry-101",
        1,
        0,
        [0.0; 3],
    );
    new_tab.disconnect().await;
    assert_stats(
        &chemistry.event("teacher_stats").await,
        "chemistry-101",
        0,
        0,
        [0.0; 3],
    );
    student2.disconnect().await;
    assert_stats(
        &teacher.event("teacher_stats").await,
        "biology-101",
        0,
        0,
        [0.0; 3],
    );

    app.request("POST", "/teacher/logout", Some(&owner), "")
        .await;
    teacher
        .emit(json!(["join_teacher_class", "biology-101"]))
        .await;
    let mut student = Client::connect(address, Some(&cookie2)).await;
    student
        .emit(json!(["register_student", 5.0, "biology-101"]))
        .await;
    teacher.no_event().await;
    foreign.no_event().await;
    server.abort();
}

struct PollingClient {
    uri: String,
    cookie: String,
}

async fn polling_request(
    app: &TestApp,
    method: &str,
    uri: &str,
    cookie: &str,
    payload: &str,
) -> String {
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .header("cookie", cookie)
        .header("content-type", "text/plain; charset=UTF-8")
        .body(Body::from(payload.to_owned()))
        .unwrap();
    let response = timeout(Duration::from_secs(3), app.router.clone().oneshot(request))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(response.status(), 200);
    String::from_utf8(
        response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap()
}

impl PollingClient {
    async fn connect(app: &TestApp, cookie: String) -> Self {
        let uri = "/socket.io/?EIO=4&transport=polling";
        let open = polling_request(app, "GET", uri, &cookie, "").await;
        assert!(open.starts_with('0'));
        let handshake: Value = serde_json::from_str(&open[1..]).unwrap();
        let uri = format!("{uri}&sid={}", handshake["sid"].as_str().unwrap());
        let client = Self { uri, cookie };
        polling_request(app, "POST", &client.uri, &client.cookie, "40").await;
        assert!(
            polling_request(app, "GET", &client.uri, &client.cookie, "")
                .await
                .starts_with("40")
        );
        client
    }

    async fn emit(&self, app: &TestApp, arguments: Value) {
        polling_request(
            app,
            "POST",
            &self.uri,
            &self.cookie,
            &format!("42{arguments}"),
        )
        .await;
    }

    async fn stats(&self, app: &TestApp) -> Value {
        loop {
            let payload = polling_request(app, "GET", &self.uri, &self.cookie, "").await;
            for packet in payload.split('\u{1e}') {
                if packet == "2" {
                    polling_request(app, "POST", &self.uri, &self.cookie, "3").await;
                    continue;
                }
                assert!(packet.starts_with("42"), "{packet}");
                let event: Value = serde_json::from_str(&packet[2..]).unwrap();
                assert_eq!(event[0], "teacher_stats");
                return event[1].clone();
            }
        }
    }
}

#[tokio::test]
async fn socket_io_polling_transport_preserves_browser_event_format() {
    let app = TestApp::new().await;
    let cookie = app.signup("teacher@test.com").await;
    app.create_class(&cookie, "biology-101").await;
    let teacher = PollingClient::connect(&app, cookie).await;
    teacher
        .emit(&app, json!(["join_teacher_class", "biology-101"]))
        .await;
    assert_stats(&teacher.stats(&app).await, "biology-101", 0, 0, [0.0; 3]);
    let cookie = app.request("GET", "/biology-101", None, "").await.cookie();
    let student = PollingClient::connect(&app, cookie).await;
    student
        .emit(&app, json!(["register_student", 123.5, "biology-101"]))
        .await;
    assert_stats(&teacher.stats(&app).await, "biology-101", 1, 0, [0.0; 3]);
    student.emit(&app, json!(["color_change", "green"])).await;
    assert_stats(
        &teacher.stats(&app).await,
        "biology-101",
        1,
        1,
        [1.0, 0.0, 0.0],
    );
    polling_request(&app, "POST", &student.uri, &student.cookie, "41").await;
    assert_stats(&teacher.stats(&app).await, "biology-101", 0, 0, [0.0; 3]);
}
