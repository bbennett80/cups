import { test } from "node:test";
import assert from "node:assert/strict";
import { studentState, studentTransition as student, teacherState, teacherTransition as teacher,
  validStats, formTransition as form, themeState, themeTransition as theme, resolvedTheme,
  qrState, qrTransition as qr } from "../../static/state-machines.mjs";

function readyStudent() {
  const joining = student(studentState(), { type: "CONNECTED" });
  return student(joining, { type: "REGISTERED", epoch: joining.epoch });
}
const stats = { class_id: "biology", student_count: 2, active_student_count: 1,
  color2frac: { green: 1, yellow: 0, red: 0 } };

test("student shows only server-confirmed feedback and rejects duplicate clicks", () => {
  const ready = readyStudent();
  const sending = student(ready, { type: "SELECT", color: "green" });
  assert.equal(sending.phase, "sending");
  assert.equal(sending.color, "inactive");
  assert.equal(student(sending, { type: "SELECT", color: "red" }), sending);
  const confirmed = student(sending, { type: "CONFIRMED", epoch: sending.epoch, revision: sending.revision });
  assert.equal(confirmed.color, "green");
  assert.equal(confirmed.phase, "ready");
});

test("offline choices are never queued; reconnect resets feedback and ignores stale acknowledgements", () => {
  const sending = student(readyStudent(), { type: "SELECT", color: "green" });
  const offline = student(sending, { type: "DISCONNECTED" });
  assert.equal(student(offline, { type: "SELECT", color: "red" }), offline);
  for (const type of ["CONFIRMED", "FAILED"]) {
    assert.equal(student(offline, { type, epoch: sending.epoch, revision: sending.revision }), offline);
  }
  const rejoining = student(offline, { type: "CONNECTED" });
  assert.equal(rejoining.phase, "registering");
  assert.equal(rejoining.color, "inactive");
  assert.equal(student(rejoining, { type: "FAILED", epoch: sending.epoch }), rejoining);
  const ready = student(rejoining, { type: "REGISTERED", epoch: rejoining.epoch });
  assert.equal(ready.phase, "ready");
});

test("lost registration acknowledgements cannot mark a disconnected student ready or failed", () => {
  const joining = student(studentState(), { type: "CONNECTED" });
  const offline = student(joining, { type: "DISCONNECTED" });
  for (const type of ["REGISTERED", "FAILED"]) {
    assert.equal(student(offline, { type, epoch: joining.epoch }), offline);
  }
});

test("superseded student is terminal until reload, including pending callbacks and reconnects", () => {
  const sending = student(readyStudent(), { type: "SELECT", color: "yellow" });
  const terminal = student(sending, { type: "SUPERSEDED" });
  for (const type of ["CONNECTED", "DISCONNECTED", "REGISTERED", "CONFIRMED", "FAILED", "SELECT"]) {
    assert.equal(student(terminal, { type, epoch: sending.epoch, revision: sending.revision, color: "red" }), terminal);
  }
  assert.equal(terminal.color, "inactive");
  assert.equal(terminal.pending, null);
});

test("failed submissions retain confirmed color and allow no further unconfirmed input", () => {
  let state = readyStudent();
  state = student(state, { type: "SELECT", color: "green" });
  state = student(state, { type: "CONFIRMED", epoch: state.epoch, revision: state.revision });
  state = student(state, { type: "SELECT", color: "red" });
  const failed = student(state, { type: "FAILED", epoch: state.epoch, revision: state.revision });
  assert.equal(failed.phase, "error");
  assert.equal(failed.color, "green");
  assert.equal(student(failed, { type: "SELECT", color: "yellow" }), failed);
});

test("teacher becomes live only after valid class data; disconnected values remain an explicit stale snapshot", () => {
  let state = teacher(teacherState(), { type: "CONNECTED" });
  assert.equal(state.phase, "joining");
  assert.equal(teacher(state, { type: "STATS", stats, classId: "chemistry" }), state);
  state = teacher(state, { type: "STATS", stats, classId: "biology" });
  assert.equal(state.phase, "live");
  const offline = teacher(state, { type: "DISCONNECTED" });
  assert.equal(offline.stats, stats);
  assert.equal(teacher(offline, { type: "STATS", stats, classId: "biology" }), offline);
  assert.equal(teacher(offline, { type: "FAILED", epoch: state.epoch, reason: "unavailable" }), offline);
  const joining = teacher(offline, { type: "CONNECTED" });
  assert.equal(teacher(joining, { type: "FAILED", epoch: state.epoch, reason: "expired" }), joining);
});

test("teacher denial and expiry cannot be undone by late stats or connection callbacks", () => {
  for (const reason of ["unauthenticated", "expired", "revoked", "forbidden"]) {
    const joining = teacher(teacherState(), { type: "CONNECTED" });
    const denied = teacher(joining, { type: "FAILED", reason });
    assert.equal(denied.phase, "denied");
    for (const type of ["CONNECTED", "DISCONNECTED", "STATS"]) {
      assert.equal(teacher(denied, { type, stats, classId: "biology" }), denied);
    }
  }
});

test("invalid counts and color fractions never become live data", () => {
  assert.equal(validStats(stats, "biology"), true);
  for (const invalid of [null, {}, { ...stats, student_count: -1 }, { ...stats, active_student_count: 3 },
    { ...stats, color2frac: { green: NaN, red: 0, yellow: 0 } }, { ...stats, color2frac: { green: 0, red: 0, yellow: 0 } }]) {
    assert.equal(validStats(invalid, "biology"), false);
  }
});

test("forms permit retries after rejection and recover from browser back/forward cache", () => {
  for (const phase of ["ready", "rejected"]) {
    const submitting = form({ phase }, { type: "SUBMIT" });
    assert.equal(submitting.phase, "submitting");
    assert.equal(form(submitting, { type: "SUBMIT" }), submitting);
    assert.equal(form(submitting, { type: "RESTORE", rejected: false }).phase, "ready");
    assert.equal(form(submitting, { type: "RESTORE", rejected: true }).phase, "rejected");
  }
});

test("theme follows the system until an explicit preference, and synchronizes storage changes", () => {
  let state = themeState(null, "dark");
  assert.equal(resolvedTheme(state), "dark");
  state = theme(state, { type: "SYSTEM_CHANGED", dark: false });
  assert.equal(resolvedTheme(state), "light");
  state = theme(state, { type: "TOGGLE" });
  assert.equal(state.preference, "dark");
  state = theme(state, { type: "SYSTEM_CHANGED", dark: false });
  assert.equal(resolvedTheme(state), "dark");
  state = theme(state, { type: "STORAGE_CHANGED", value: "light" });
  assert.equal(resolvedTheme(state), "light");
  state = theme(state, { type: "STORAGE_CHANGED", value: null });
  assert.equal(state.preference, "system");
});

test("QR callbacks cannot reopen a closed modal or replace the current class with an old image", () => {
  const first = qr(qrState(), { type: "OPEN", className: "biology", url: "https://cups.test/biology" });
  const closed = qr(first, { type: "CLOSE" });
  assert.equal(qr(closed, { type: "LOADED", generation: first.generation }), closed);
  const second = qr(closed, { type: "OPEN", className: "chemistry", url: "https://cups.test/chemistry" });
  assert.equal(qr(second, { type: "FAILED", generation: first.generation }), second);
  const ready = qr(second, { type: "LOADED", generation: second.generation });
  assert.equal(ready.phase, "ready");
  assert.equal(ready.className, "chemistry");
});

test("QR image failure preserves the shareable student link", () => {
  const loading = qr(qrState(), { type: "OPEN", className: "biology", url: "https://cups.test/biology" });
  const failed = qr(loading, { type: "FAILED", generation: loading.generation });
  assert.equal(failed.phase, "error");
  assert.equal(failed.url, loading.url);
  assert.equal(qr(failed, { type: "LOADED", generation: loading.generation }), failed);
});
