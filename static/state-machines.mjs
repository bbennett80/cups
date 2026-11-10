// Pure transitions shared by browser controllers and Node tests. Effects live in app.mjs.
export const colors = Object.freeze(["green", "yellow", "red"]);
const accessFailures = new Set(["unauthenticated", "expired", "revoked", "forbidden"]);
const terminalStudent = new Set(["superseded", "error"]);
const terminalTeacher = new Set(["denied", "error"]);

export function studentState() {
  return { phase: "connecting", epoch: 0, revision: 0, color: "inactive", pending: null };
}

export function studentTransition(state, event) {
  if (event.type === "SUPERSEDED") return { ...state, phase: "superseded", color: "inactive", pending: null };
  if (terminalStudent.has(state.phase)) return state;
  switch (event.type) {
    case "CONNECTED":
      return { ...state, phase: "registering", epoch: state.epoch + 1,
        revision: state.revision + 1, color: "inactive", pending: null };
    case "REGISTERED":
      return state.phase === "registering" && event.epoch === state.epoch
        ? { ...state, phase: "ready" } : state;
    case "SELECT":
      return state.phase === "ready" && colors.includes(event.color)
        ? { ...state, phase: "sending", revision: state.revision + 1, pending: event.color } : state;
    case "CONFIRMED":
      return state.phase === "sending" && event.epoch === state.epoch && event.revision === state.revision
        ? { ...state, phase: "ready", color: state.pending, pending: null } : state;
    case "FAILED":
      return event.epoch === undefined || (["registering", "sending"].includes(state.phase) && event.epoch === state.epoch &&
        (event.revision === undefined || event.revision === state.revision))
        ? { ...state, phase: event.reason === "superseded" ? "superseded" : "error",
          color: event.reason === "superseded" ? "inactive" : state.color, pending: null } : state;
    case "DISCONNECTED":
      return { ...state, phase: "offline", epoch: state.epoch + 1, revision: state.revision + 1, pending: null };
    default: return state;
  }
}

export function teacherState() { return { phase: "connecting", epoch: 0, stats: null, reason: null }; }

export function validStats(stats, classId) {
  if (!stats || stats.class_id !== classId || !Number.isInteger(stats.student_count) ||
      !Number.isInteger(stats.active_student_count) || stats.active_student_count < 0 ||
      stats.student_count < stats.active_student_count) return false;
  const fractions = colors.map(color => stats.color2frac?.[color]);
  return fractions.every(value => Number.isFinite(value) && value >= 0 && value <= 1) &&
    Math.abs(fractions.reduce((a, b) => a + b, 0) - (stats.active_student_count > 0 ? 1 : 0)) < 0.000001;
}

export function teacherTransition(state, event) {
  if (terminalTeacher.has(state.phase)) return state;
  switch (event.type) {
    case "CONNECTED": return { ...state, phase: "joining", epoch: state.epoch + 1, reason: null };
    case "STATS":
      return ["joining", "live"].includes(state.phase) && validStats(event.stats, event.classId)
        ? { ...state, phase: "live", stats: event.stats } : state;
    case "FAILED":
      return event.epoch === undefined || event.epoch === state.epoch
        ? { ...state, phase: accessFailures.has(event.reason) ? "denied" : "error", reason: event.reason } : state;
    case "DISCONNECTED": return { ...state, phase: "offline", epoch: state.epoch + 1 };
    default: return state;
  }
}

export function formTransition(state, event) {
  if (event.type === "SUBMIT" && state.phase !== "submitting") return { phase: "submitting" };
  if (event.type === "RESTORE") return { phase: event.rejected ? "rejected" : "ready" };
  return state;
}

export function themeState(preference, system) {
  return { preference: ["light", "dark"].includes(preference) ? preference : "system", system: system === "dark" ? "dark" : "light" };
}
export function resolvedTheme(state) { return state.preference === "system" ? state.system : state.preference; }
export function themeTransition(state, event) {
  switch (event.type) {
    case "TOGGLE": return { ...state, preference: resolvedTheme(state) === "dark" ? "light" : "dark" };
    case "SYSTEM_CHANGED": return { ...state, system: event.dark ? "dark" : "light" };
    case "STORAGE_CHANGED": return themeState(event.value, state.system);
    default: return state;
  }
}

export function qrState() { return { phase: "closed", generation: 0, className: "", url: "" }; }
export function qrTransition(state, event) {
  switch (event.type) {
    case "OPEN": return { phase: "loading", generation: state.generation + 1, className: event.className, url: event.url };
    case "CLOSE": return { ...state, phase: "closed", generation: state.generation + 1 };
    case "LOADED":
    case "FAILED": return state.phase === "loading" && event.generation === state.generation
      ? { ...state, phase: event.type === "LOADED" ? "ready" : "error" } : state;
    default: return state;
  }
}
