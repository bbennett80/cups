import { colors, studentState, studentTransition, teacherState, teacherTransition,
  formTransition, themeState, resolvedTheme, themeTransition, qrState, qrTransition } from "./state-machines.mjs";

const ACK_TIMEOUT = 5000;

function initStudent(page) {
  let state = studentState();
  const path = document.getElementById("cup_status_path");
  const status = document.getElementById("connection_status");
  const choices = [...page.querySelectorAll("[data-color]")];
  const timestamp = Number(document.getElementById("timestamp").textContent);
  const classId = document.getElementById("class_id").textContent;
  const retry = page.querySelector("[data-reload]");
  function dispatch(event) {
    state = studentTransition(state, event);
    page.dataset.state = state.phase;
    path.setAttribute("class", state.color);
    for (const choice of choices) {
      choice.disabled = state.phase !== "ready";
      choice.setAttribute("aria-pressed", String(choice.dataset.color === state.color));
    }
    status.textContent = {
      connecting: "Connecting…", registering: "Joining the class…",
      ready: state.color === "inactive" ? "Connected. Choose a color to share your status." : "Your status is shared with your teacher.",
      sending: "Sending your status…", offline: "Connection lost. Reconnecting; your last confirmed status is shown.",
      superseded: "This tab is no longer active. Reload this page to use it again.",
      error: "We could not confirm your connection. Reload to try again.",
    }[state.phase];
    retry.hidden = !["superseded", "error"].includes(state.phase);
  }
  dispatch({ type: "INIT" });
  if (typeof window.io !== "function") { dispatch({ type: "FAILED" }); return; }
  const socket = window.io({ autoConnect: false });
  function fail(error, response, context) {
    if (!error && response?.status === "ok") return false;
    dispatch({ type: "FAILED", reason: response?.reason, ...context });
    if (["error", "superseded"].includes(state.phase)) socket.disconnect();
    return true;
  }
  socket.on("connect", () => {
    dispatch({ type: "CONNECTED" });
    if (state.phase !== "registering") return;
    const epoch = state.epoch;
    socket.timeout(ACK_TIMEOUT).emit("register_student", timestamp, classId, (error, response) => {
      if (!fail(error, response, { epoch })) dispatch({ type: "REGISTERED", epoch });
    });
  });
  socket.on("disconnect", () => dispatch({ type: "DISCONNECTED" }));
  socket.on("connect_error", () => dispatch({ type: "DISCONNECTED" }));
  socket.on("deactivate_old_tabs", data => {
    const cookie = document.cookie.split(";").map(part => part.trim()).find(part => part.startsWith("student_id="))?.slice(11);
    if (data?.student_id === cookie && (data.superseded || data.timestamp > timestamp)) {
      dispatch({ type: "SUPERSEDED" });
      socket.disconnect();
    }
  });
  for (const choice of choices) choice.addEventListener("click", () => {
    if (state.phase !== "ready" || !socket.connected) return;
    dispatch({ type: "SELECT", color: choice.dataset.color });
    const { epoch, revision, pending } = state;
    // Offline choices never enter Socket.IO's send buffer, and color is shown only after ACK.
    socket.timeout(ACK_TIMEOUT).emit("color_change", pending, (error, response) => {
      if (!fail(error, response, { epoch, revision })) dispatch({ type: "CONFIRMED", epoch, revision });
    });
  });
  socket.connect();
}

function initTeacher(page) {
  let state = teacherState();
  const classId = document.getElementById("class_id").textContent.trim();
  const status = document.getElementById("connection_status");
  const action = page.querySelector("[data-reconnect-action]");
  function dispatch(event) {
    state = teacherTransition(state, event);
    page.dataset.state = state.phase;
    status.textContent = {
      connecting: "Connecting to live feedback…", joining: "Checking class access…",
      live: "Live feedback is connected.", offline: "Connection lost. Reconnecting; these counts may be out of date.",
      denied: state.reason === "forbidden" ? "This account does not own this class." : "Your session has ended. Sign in again to resume live feedback.",
      error: "Live feedback is unavailable. Reload to try again.",
    }[state.phase];
    action.hidden = !["denied", "error"].includes(state.phase);
    action.href = state.phase === "denied"
      ? (state.reason === "forbidden" ? "/teacher/dashboard" : "/teacher/signin?next_path=" + encodeURIComponent(`/${classId}/teacher`)) : "";
    action.textContent = state.phase === "denied" ? (state.reason === "forbidden" ? "Open dashboard" : "Sign in again") : "Reload";
    if (state.phase === "live" && state.stats) {
      document.getElementById("active_student_count").textContent = state.stats.active_student_count;
      document.getElementById("student_count").textContent = state.stats.student_count;
      for (const color of colors) document.getElementById(`meter_${color}`).style.setProperty("--fill-height", state.stats.color2frac[color] * 60 + "vh");
    }
  }
  dispatch({ type: "INIT" });
  if (typeof window.io !== "function") { dispatch({ type: "FAILED", reason: "unavailable" }); return; }
  const socket = window.io({ autoConnect: false, reconnection: true });
  socket.on("connect", () => {
    dispatch({ type: "CONNECTED" });
    const epoch = state.epoch;
    socket.timeout(ACK_TIMEOUT).emit("join_teacher_class", classId, (error, response) => {
      if (error || response?.status !== "ok") {
        dispatch({ type: "FAILED", epoch, reason: response?.reason || "unavailable" });
        if (["denied", "error"].includes(state.phase)) socket.disconnect();
      }
    });
  });
  socket.on("disconnect", () => dispatch({ type: "DISCONNECTED" }));
  socket.on("connect_error", () => dispatch({ type: "DISCONNECTED" }));
  socket.on("teacher_stats", stats => dispatch({ type: "STATS", stats, classId }));
  socket.on("teacher_access_ended", data => {
    dispatch({ type: "FAILED", reason: data?.reason || "unauthenticated" });
    socket.disconnect();
  });
  socket.connect();
}

function initForms() {
  for (const form of document.querySelectorAll("form[method='post']")) {
    let state = { phase: form.dataset.state || "ready" };
    const buttons = [...form.querySelectorAll("button[type='submit']")];
    const labels = buttons.map(button => button.textContent);
    function render() {
      form.dataset.state = state.phase;
      form.setAttribute("aria-busy", String(state.phase === "submitting"));
      buttons.forEach((button, index) => {
        button.disabled = state.phase === "submitting";
        button.textContent = state.phase === "submitting" ? "Please wait…" : labels[index];
      });
    }
    form.addEventListener("submit", event => {
      if (state.phase === "submitting") { event.preventDefault(); return; }
      state = formTransition(state, { type: "SUBMIT" });
      render();
    });
    window.addEventListener("pageshow", () => {
      state = formTransition(state, { type: "RESTORE", rejected: Boolean(form.querySelector("[aria-invalid='true']")) || form.dataset.initialState === "rejected" });
      render();
    });
    form.dataset.initialState = state.phase;
    render();
  }
}

function initTheme() {
  const root = document.documentElement;
  const media = window.matchMedia?.("(prefers-color-scheme: dark)");
  let state = themeState(root.dataset.themePreference, media?.matches ? "dark" : "light");
  const toggle = document.getElementById("theme_toggle");
  function dispatch(event) {
    state = themeTransition(state, event);
    const theme = resolvedTheme(state);
    root.dataset.theme = theme;
    root.dataset.themePreference = state.preference;
    toggle?.setAttribute("aria-label", theme === "dark" ? "Switch to light mode" : "Switch to dark mode");
  }
  toggle?.addEventListener("click", () => {
    dispatch({ type: "TOGGLE" });
    try { localStorage.setItem("theme", state.preference); } catch { /* In-memory preference still works. */ }
  });
  media?.addEventListener("change", event => dispatch({ type: "SYSTEM_CHANGED", dark: event.matches }));
  window.addEventListener("storage", event => {
    if (event.key === "theme" || event.key === null) dispatch({ type: "STORAGE_CHANGED", value: event.newValue });
  });
  dispatch({ type: "INIT" });
}

function initQrModal(modal) {
  let state = qrState();
  let returnFocus = null;
  let previousOverflow = "";
  let inertSiblings = [];
  const close = document.getElementById("close_qr_modal");
  const status = document.getElementById("qr_status");
  const link = document.getElementById("qr_student_url");
  function dispatch(event) {
    state = qrTransition(state, event);
    const open = state.phase !== "closed";
    modal.dataset.state = state.phase;
    modal.classList.toggle("open", open);
    modal.setAttribute("aria-hidden", String(!open));
    document.getElementById("qr_modal_subtitle").textContent = "Class: " + state.className;
    link.textContent = state.url;
    link.href = state.url;
    status.textContent = { closed: "", loading: "Loading QR code…", ready: "QR code ready.", error: "QR code could not load. You can still share the student link below." }[state.phase];
    document.getElementById("qr_image").hidden = state.phase !== "ready";
  }
  function closeModal() {
    if (state.phase === "closed") return;
    dispatch({ type: "CLOSE" });
    for (const sibling of inertSiblings) sibling.inert = false;
    inertSiblings = [];
    document.body.style.overflow = previousOverflow;
    returnFocus?.focus();
  }
  for (const button of document.querySelectorAll(".student-qr-trigger")) button.addEventListener("click", () => {
    if (state.phase === "closed") {
      returnFocus = button;
      previousOverflow = document.body.style.overflow;
      inertSiblings = [...document.body.children].filter(element => element !== modal && !element.inert);
      for (const sibling of inertSiblings) sibling.inert = true;
      document.body.style.overflow = "hidden";
    }
    dispatch({ type: "OPEN", className: button.dataset.className, url: button.dataset.studentUrl });
    const generation = state.generation;
    const image = new Image(220, 220);
    image.id = "qr_image";
    image.alt = "Student class QR code";
    image.hidden = true;
    image.addEventListener("load", () => dispatch({ type: "LOADED", generation }));
    image.addEventListener("error", () => dispatch({ type: "FAILED", generation }));
    document.getElementById("qr_image").replaceWith(image);
    image.src = "https://api.qrserver.com/v1/create-qr-code/?size=220x220&data=" + encodeURIComponent(state.url);
    close.focus();
  });
  close.addEventListener("click", closeModal);
  modal.addEventListener("click", event => { if (event.target.dataset.closeModal === "true") closeModal(); });
  document.addEventListener("keydown", event => {
    if (state.phase === "closed") return;
    if (event.key === "Escape") { event.preventDefault(); closeModal(); }
    if (event.key === "Tab") {
      if (event.shiftKey && document.activeElement === close) { event.preventDefault(); link.focus(); }
      else if (!event.shiftKey && document.activeElement === link) { event.preventDefault(); close.focus(); }
    }
  });
  dispatch({ type: "INIT" });
}

initTheme();
initForms();
const student = document.querySelector("[data-page='student']");
const teacher = document.querySelector("[data-page='teacher']");
if (student) initStudent(student);
if (teacher) initTeacher(teacher);
const modal = document.getElementById("qr_modal");
if (modal) initQrModal(modal);
