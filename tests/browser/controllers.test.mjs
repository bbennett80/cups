// Exercise the production DOM/socket controllers without npm dependencies.
// This harness verifies event wiring and rendered state; it is not a layout engine.
import { test } from "node:test";
import assert from "node:assert/strict";

class Element {
  constructor(document, dataset = {}) {
    this.document = document;
    this.dataset = dataset;
    this.listeners = new Map();
    this.attributes = new Map();
    this.children = [];
    this.selectors = new Map();
    this.textContent = "";
    this.hidden = false;
    this.inert = false;
    this.disabled = false;
    this.style = { overflow: "", setProperty(name, value) { this[name] = value; } };
    const classes = new Set();
    this.classList = { toggle(name, enabled) { enabled ? classes.add(name) : classes.delete(name); } };
  }
  addEventListener(type, callback) {
    const callbacks = this.listeners.get(type) || [];
    callbacks.push(callback);
    this.listeners.set(type, callbacks);
  }
  fire(type, details = {}) {
    const event = { type, target: this, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, ...details };
    for (const callback of this.listeners.get(type) || []) callback(event);
    return event;
  }
  setAttribute(key, value) { this.attributes.set(key, value); }
  getAttribute(key) { return this.attributes.get(key); }
  querySelector(selector) { return this.selectors.get(selector) || null; }
  querySelectorAll(selector) { return this.selectors.get(selector) || []; }
  focus() { this.document.activeElement = this; }
  replaceWith(element) { this.document.ids.set(this.id, element); }
}

class Socket {
  constructor() { this.listeners = new Map(); this.commands = []; this.connected = false; }
  on(type, callback) { this.listeners.set(type, callback); }
  fire(type, value) { this.listeners.get(type)?.(value); }
  connect() { this.connected = true; this.fire("connect"); }
  disconnect() { this.connected = false; this.fire("disconnect"); }
  timeout() { return this; }
  emit(name, ...args) { this.commands.push({ name, args: args.slice(0, -1), callback: args.at(-1) }); }
  take(name) {
    const index = this.commands.findIndex(command => command.name === name);
    assert.notEqual(index, -1, `missing command ${name}`);
    return this.commands.splice(index, 1)[0];
  }
}

let generation = 0;
async function setup(kind, { library = true, rejected = false } = {}) {
  const document = new Element(null);
  document.document = document;
  document.ids = new Map();
  document.cookie = "student_id=student-cookie";
  document.documentElement = new Element(document, { themePreference: "system" });
  document.body = new Element(document);
  const add = (id, dataset = {}) => {
    const element = new Element(document, dataset);
    element.id = id;
    document.ids.set(id, element);
    return element;
  };
  const root = document.documentElement;
  add("theme_toggle");
  document.getElementById = id => document.ids.get(id);
  document.querySelector = selector => document.selectors.get(selector) || null;
  document.querySelectorAll = selector => document.selectors.get(selector) || [];
  document.selectors.set("form[method='post']", []);
  const media = new Element(document);
  media.matches = false;
  const window = new Element(document);
  window.matchMedia = () => media;
  const socket = new Socket();
  if (library) window.io = () => socket;
  let page;
  if (kind === "student" || kind === "teacher") {
    page = new Element(document);
    document.selectors.set(`[data-page='${kind}']`, page);
    add("class_id").textContent = "biology-101";
    add("connection_status");
    const action = new Element(document);
    page.selectors.set(kind === "student" ? "[data-reload]" : "[data-reconnect-action]", action);
    if (kind === "student") {
      add("cup_status_path");
      add("timestamp").textContent = "123.5";
      page.selectors.set("[data-color]", ["green", "yellow", "red"].map(color => new Element(document, { color })));
    } else {
      for (const id of ["meter_green", "meter_yellow", "meter_red", "active_student_count", "student_count"]) add(id);
    }
  }
  let submit, form;
  if (kind === "form") {
    form = new Element(document, { state: rejected ? "rejected" : "ready" });
    submit = new Element(document);
    submit.textContent = "Create Account";
    form.selectors.set("button[type='submit']", [submit]);
    document.selectors.set("form[method='post']", [form]);
  }
  let opener;
  const images = [];
  if (kind === "qr") {
    page = add("qr_modal");
    for (const id of ["close_qr_modal", "qr_status", "qr_student_url", "qr_modal_subtitle", "qr_image"]) add(id);
    opener = new Element(document, { className: "biology-101", studentUrl: "https://cups.test/biology-101" });
    const background = new Element(document);
    document.body.children = [background, page];
    document.selectors.set(".student-qr-trigger", [opener]);
  }
  const saved = new Map();
  const globals = {
    document, window, localStorage: { setItem(key, value) { saved.set(key, value); } },
    Image: class extends Element { constructor() { super(document); images.push(this); } },
  };
  const previous = new Map(Object.keys(globals).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  Object.assign(globalThis, globals);
  await import(`../../static/app.mjs?test=${++generation}`);
  return { document, window, root, media, socket, page, submit, form, opener, images, saved, cleanup() {
    for (const [key, descriptor] of previous) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  } };
}

test("student controller waits for registration and color ACKs, and never buffers offline clicks", async t => {
  const ui = await setup("student"); t.after(ui.cleanup);
  const choices = ui.page.querySelectorAll("[data-color]");
  assert.equal(ui.page.dataset.state, "registering");
  assert(choices.every(choice => choice.disabled));
  const registration = ui.socket.take("register_student");
  assert.deepEqual(registration.args, [123.5, "biology-101"]);
  registration.callback(null, { status: "ok", state: "registered" });
  assert.equal(ui.page.dataset.state, "ready");
  assert(choices.every(choice => !choice.disabled));
  choices[0].fire("click");
  assert.equal(ui.page.dataset.state, "sending");
  assert.equal(ui.document.getElementById("cup_status_path").getAttribute("class"), "inactive");
  ui.socket.take("color_change").callback(null, { status: "ok", state: "updated" });
  assert.equal(ui.document.getElementById("cup_status_path").getAttribute("class"), "green");
  assert.equal(choices[0].getAttribute("aria-pressed"), "true");
  ui.socket.disconnect();
  choices[1].fire("click");
  assert.equal(ui.socket.commands.length, 0);
  assert.equal(ui.page.dataset.state, "offline");
  assert.match(ui.document.getElementById("connection_status").textContent, /Connection lost/);
});

test("student controller ignores old callbacks after reconnect and respects equal-timestamp supersession", async t => {
  const ui = await setup("student"); t.after(ui.cleanup);
  const old = ui.socket.take("register_student");
  ui.socket.disconnect();
  old.callback(new Error("timeout"));
  assert.equal(ui.page.dataset.state, "offline");
  ui.socket.connect();
  const current = ui.socket.take("register_student");
  old.callback(null, { status: "ok" });
  assert.equal(ui.page.dataset.state, "registering");
  current.callback(null, { status: "ok" });
  ui.socket.fire("deactivate_old_tabs", { student_id: "student-cookie", timestamp: 123.5, superseded: true });
  assert.equal(ui.page.dataset.state, "superseded");
  assert.equal(ui.socket.connected, false);
  assert.equal(ui.page.querySelector("[data-reload]").hidden, false);
});

test("missing Socket.IO script renders recoverable error instead of leaving controls active", async t => {
  const ui = await setup("student", { library: false }); t.after(ui.cleanup);
  assert.equal(ui.page.dataset.state, "error");
  assert(ui.page.querySelectorAll("[data-color]").every(choice => choice.disabled));
  assert.equal(ui.page.querySelector("[data-reload]").hidden, false);
});

test("teacher controller labels stale snapshots and renders session recovery", async t => {
  const ui = await setup("teacher"); t.after(ui.cleanup);
  const subscription = ui.socket.take("join_teacher_class");
  ui.socket.fire("teacher_stats", { class_id: "biology-101", student_count: 2, active_student_count: 1, color2frac: { green: 1, yellow: 0, red: 0 } });
  subscription.callback(null, { status: "ok" });
  assert.equal(ui.page.dataset.state, "live");
  assert.equal(ui.document.getElementById("student_count").textContent, 2);
  ui.socket.disconnect();
  assert.equal(ui.page.dataset.state, "offline");
  assert.match(ui.document.getElementById("connection_status").textContent, /out of date/);
  ui.socket.connect();
  ui.socket.take("join_teacher_class").callback(null, { status: "error", reason: "expired" });
  assert.equal(ui.page.dataset.state, "denied");
  assert.equal(ui.socket.connected, false);
  const action = ui.page.querySelector("[data-reconnect-action]");
  assert.match(action.href, /next_path=%2Fbiology-101%2Fteacher/);
  assert.equal(action.hidden, false);
});

test("form controller blocks duplicate submits and restores buttons after history navigation", async t => {
  const ui = await setup("form", { rejected: true }); t.after(ui.cleanup);
  assert.equal(ui.form.fire("submit").defaultPrevented, false);
  assert.equal(ui.form.dataset.state, "submitting");
  assert.equal(ui.submit.disabled, true);
  assert.equal(ui.submit.textContent, "Please wait…");
  assert.equal(ui.form.fire("submit").defaultPrevented, true);
  ui.window.fire("pageshow", { persisted: true });
  assert.equal(ui.form.dataset.state, "rejected");
  assert.equal(ui.submit.disabled, false);
  assert.equal(ui.submit.textContent, "Create Account");
  assert.equal(ui.form.fire("submit").defaultPrevented, false);
});

test("theme controller survives storage failures and follows cross-tab preference resets", async t => {
  const ui = await setup("theme"); t.after(ui.cleanup);
  globalThis.localStorage.setItem = () => { throw new Error("storage blocked"); };
  ui.document.getElementById("theme_toggle").fire("click");
  assert.equal(ui.root.dataset.theme, "dark");
  assert.match(ui.document.getElementById("theme_toggle").getAttribute("aria-label"), /light/);
  ui.media.fire("change", { matches: false });
  assert.equal(ui.root.dataset.theme, "dark");
  ui.window.fire("storage", { key: "theme", newValue: null });
  assert.equal(ui.root.dataset.themePreference, "system");
  assert.equal(ui.root.dataset.theme, "light");
});

test("QR controller preserves link on failure, contains focus, and ignores obsolete image callbacks", async t => {
  const ui = await setup("qr"); t.after(ui.cleanup);
  ui.opener.fire("click");
  assert.equal(ui.page.dataset.state, "loading");
  assert.equal(ui.document.body.children[0].inert, true);
  const first = ui.images[0];
  first.fire("error");
  assert.equal(ui.page.dataset.state, "error");
  assert.equal(ui.document.getElementById("qr_student_url").href, "https://cups.test/biology-101");
  assert.match(ui.document.getElementById("qr_status").textContent, /still share/);
  ui.document.fire("keydown", { key: "Tab", shiftKey: true });
  assert.equal(ui.document.activeElement, ui.document.getElementById("qr_student_url"));
  ui.document.fire("keydown", { key: "Escape" });
  assert.equal(ui.page.dataset.state, "closed");
  assert.equal(ui.document.body.children[0].inert, false);
  assert.equal(ui.document.activeElement, ui.opener);
  ui.opener.dataset.className = "chemistry-101";
  ui.opener.dataset.studentUrl = "https://cups.test/chemistry-101";
  ui.opener.fire("click");
  first.fire("load");
  assert.equal(ui.page.dataset.state, "loading");
  ui.images[1].fire("load");
  assert.equal(ui.page.dataset.state, "ready");
  assert.equal(ui.document.getElementById("qr_image").hidden, false);
  assert.equal(ui.document.getElementById("qr_student_url").href, "https://cups.test/chemistry-101");
});
