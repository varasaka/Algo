"use strict";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const inr = (v) => v == null ? "—" : "₹" + Number(v).toLocaleString("en-IN", { maximumFractionDigits: 0 });
const pct = (v) => v == null ? "—" : (v * 100).toFixed(1) + "%";
const reasonText = (r) => String(r ?? "")
  .replace(/^stop_hit@([\d.]+)/, (_, p) => "stop-loss hit at " + Number(p).toFixed(2))
  .replace(/^signal=\+?0\.00$/, "strategy exit").replace(/^signal=([+-][\d.]+)/, "signal $1")
  .replace("end_of_backtest", "still open at end").replace(/_/g, " ");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

let settings = null, strategies = [], statusTimer = null;

async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  const data = await r.json().catch(() => ({}));
  if (r.status === 401 && data.error === "locked") { showLock(); throw new Error("locked"); }
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

function toast(msg, error = false) {
  const t = $("#toast");
  t.textContent = msg; t.className = "toast" + (error ? " error" : ""); t.hidden = false;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => (t.hidden = true), error ? 6000 : 3000);
}

async function guard(btn, fn) {
  if (btn) btn.disabled = true;
  try { return await fn(); }
  catch (e) { if (e.message !== "locked") toast(e.message, true); }
  finally { if (btn) btn.disabled = false; }
}

/* ---------------------------------------------------------- lock screen */
async function checkSession() {
  const s = await (await fetch("/api/session")).json();
  if (s.authed) { $("#lock").hidden = true; return true; }
  showLock(s);
  return false;
}
function showLock(s = { pin_set: true }) {
  $("#lock").hidden = false;
  $("#lock-msg").textContent = s.pin_set ? "Enter your PIN" : "Create a PIN to protect this app";
  $("#lock-btn").textContent = s.pin_set ? "Unlock" : "Set PIN";
  $("#lock-form").dataset.mode = s.pin_set ? "unlock" : "set";
  $("#pin").focus();
}
$("#lock-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const pin = $("#pin").value, mode = e.target.dataset.mode;
  const r = await fetch(mode === "set" ? "/api/pin" : "/api/unlock", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ pin }) });
  const d = await r.json();
  if (!r.ok) { $("#lock-err").textContent = d.error || "failed"; return; }
  $("#pin").value = ""; $("#lock-err").textContent = ""; $("#lock").hidden = true;
  init();
});

/* ---------------------------------------------------------------- tabs */
function showTab() {
  const name = (location.hash || "#dashboard").slice(1);
  $$(".tab").forEach((t) => (t.hidden = t.id !== "tab-" + name));
  $$("#tabs a").forEach((a) => a.classList.toggle("active", a.dataset.tab === name));
  if (name === "activity") loadJournal();
}
window.addEventListener("hashchange", showTab);

/* ------------------------------------------------------------ dashboard */
async function refreshStatus() {
  let s;
  try { s = await api("/api/status"); } catch { return; }
  const badge = $("#mode-badge");
  badge.textContent = s.mode.toUpperCase(); badge.className = "badge " + s.mode;
  $("#run-dot").classList.toggle("on", s.running);
  $("#run-dot").title = s.running ? "running" : "stopped";
  $("#st-status").textContent = s.halted ? "Halted" : s.running ? "Running" : "Stopped";
  $("#st-status").className = "value " + (s.halted ? "bad" : s.running ? "good" : "");
  $("#st-market").textContent = (s.market_open ? "Market open" : "Market closed") + " · " + s.now.slice(11, 16) + " IST";
  $("#st-equity").textContent = s.account ? inr(s.account.equity) : "—";
  $("#st-cash").textContent = s.account ? "Cash " + inr(s.account.cash) + " · Capital " + inr(s.account.capital) : (s.account_error || "");
  $("#st-last").textContent = s.last_cycle ? s.last_cycle.replace("T", " ").slice(0, 16) : "Never";
  $("#st-broker").textContent = { paper: "Paper broker", kite: "Zerodha Kite", paytm: "Paytm Money" }[s.broker] + (s.broker !== "paper" ? (s.connected ? " · logged in" : " · not logged in") : "");
  $("#conn-status").textContent = s.broker === "paper" ? "Paper trading: no login needed" : s.connected ? "Logged in for today" : "Not logged in";
  $("#conn-status").className = s.connected ? "good" : "bad";

  const msgs = [];
  if (s.last_error) msgs.push("⚠ " + s.last_error);
  if (s.broker !== "paper" && !s.connected) msgs.push("Log in to your broker (Broker tab). Brokers require a fresh login every day.");
  if (s.mode === "live") msgs.push("LIVE mode: real orders are placed automatically.");
  $("#banner").hidden = !msgs.length; $("#banner").innerHTML = msgs.map(esc).join("<br>");

  const tb = $("#pos-table tbody"); tb.innerHTML = "";
  (s.positions || []).forEach((p) => {
    tb.insertAdjacentHTML("beforeend", `<tr><td>${esc(p.symbol)}</td><td class="num">${p.qty}</td><td class="num">${p.price ? p.price.toFixed(2) : "—"}</td><td class="num">${inr(p.value)}</td><td class="num">${p.stop ?? "—"}</td><td>${p.managed ? "✓" : ""}</td></tr>`);
  });
  $("#pos-empty").hidden = (s.positions || []).length > 0;
  const sig = s.signals || {};
  $("#signals").innerHTML = Object.keys(sig).length
    ? Object.entries(sig).map(([k, v]) => `<span class="chip">${esc(k)} <b class="${v > 0 ? "good" : v < 0 ? "bad" : ""}">${v > 0 ? "▲" : v < 0 ? "▼" : "•"} ${v.toFixed(2)}</b></span>`).join("")
    : '<span class="muted">Shown after the first run.</span>';
  drawChart($("#eq-chart"), s.equity_history || []);
  $("#btn-start").disabled = s.running; $("#btn-stop").disabled = !s.running;
}

function drawChart(el, points) {
  if (!points || points.length < 2) { el.innerHTML = '<p class="muted">The chart appears after the bot has run at least twice (once per trading day on daily candles).</p>'; return; }
  const W = 900, H = 200, P = 34;
  const ys = points.map((p) => p.equity), min = Math.min(...ys), max = Math.max(...ys), span = max - min || 1;
  const x = (i) => P + (i * (W - P - 8)) / (points.length - 1);
  const y = (v) => 8 + (H - 30) * (1 - (v - min) / span);
  const line = points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.equity).toFixed(1)}`).join("");
  const area = `${line}L${x(points.length - 1)},${H - 22}L${P},${H - 22}Z`;
  const label = (t) => String(t).slice(0, 10);
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Equity curve">
    <path d="${area}" fill="var(--chart-fill)"/><path d="${line}" fill="none" stroke="var(--chart-line)" stroke-width="2" vector-effect="non-scaling-stroke"/>
    <text x="2" y="14">${inr(max)}</text><text x="2" y="${H - 26}">${inr(min)}</text>
    <text x="${P}" y="${H - 4}">${label(points[0].time)}</text><text x="${W - 8}" y="${H - 4}" text-anchor="end">${label(points[points.length - 1].time)}</text></svg>`;
}

$("#btn-start").onclick = (e) => guard(e.target, async () => { await api("/api/start", {}); toast("Auto-trading started"); refreshStatus(); });
$("#btn-stop").onclick = (e) => guard(e.target, async () => { await api("/api/stop", {}); toast("Stopped"); refreshStatus(); });
$("#btn-run").onclick = (e) => guard(e.target, async () => {
  const r = await api("/api/run-now", {});
  const acts = r.entries.filter((x) => ["enter", "exit", "rebalance", "order"].includes(x.action));
  toast(acts.length ? `${acts.length} order decision(s) made` : "Run complete: no trades needed");
  refreshStatus();
});
$("#btn-kill").onclick = (e) => {
  if (!confirm("Stop the bot and close ALL positions it manages?")) return;
  guard(e.target, async () => { await api("/api/kill", {}); toast("Kill switch: positions closed, bot stopped"); refreshStatus(); });
};

/* --------------------------------------------------------------- broker */
function fillBroker() {
  $$("input[name=broker]").forEach((r) => (r.checked = r.value === settings.broker));
  $$("input[name=mode]").forEach((r) => (r.checked = r.value === settings.mode));
  ["kite_api_key", "kite_api_secret", "paytm_api_key", "paytm_api_secret"].forEach((k) => ($("#" + k).value = settings[k] || ""));
  $("#kite-redirect").textContent = location.origin + "/broker/kite/callback";
  $("#paytm-redirect").textContent = location.origin + "/broker/paytm/callback";
  showCreds();
}
function showCreds() {
  const b = ($("input[name=broker]:checked") || {}).value;
  $("#cred-kite").hidden = b !== "kite"; $("#cred-paytm").hidden = b !== "paytm";
  $("#login-card").hidden = b === "paper";
}
$$("input[name=broker]").forEach((r) => r.addEventListener("change", showCreds));

async function saveSettings(changes, msg = "Saved") {
  settings = await api("/api/settings", changes);
  toast(msg); fillAll(); refreshStatus();
}
$("#btn-save-broker").onclick = (e) => guard(e.target, () => {
  const broker = $("input[name=broker]:checked").value;
  const ch = { broker };
  ["kite_api_key", "kite_api_secret", "paytm_api_key", "paytm_api_secret"].forEach((k) => (ch[k] = $("#" + k).value.trim()));
  if (broker === "paper") ch.mode = "paper";
  else if (settings.mode === "paper" || settings.broker !== broker) ch.mode = "shadow";
  return saveSettings(ch, broker === "paper" ? "Paper trading selected" : "Broker saved. Mode set to Shadow for safety.");
});
$("#btn-login").onclick = (e) => guard(e.target, async () => { const { url } = await api("/api/broker/login-url"); location.href = url; });
$("#btn-test").onclick = (e) => guard(e.target, async () => {
  const r = await api("/api/broker/test", {});
  $("#test-out").hidden = false; $("#test-out").textContent = JSON.stringify(r, null, 2);
});
$("#btn-logout").onclick = (e) => guard(e.target, async () => { await api("/api/broker/logout", {}); toast("Logged out"); refreshStatus(); });
$("#btn-token").onclick = (e) => guard(e.target, async () => {
  await api("/api/broker/token", { request_token: $("#manual-token").value.trim() });
  $("#manual-token").value = ""; toast("Connected"); refreshStatus();
});
$("#btn-save-mode").onclick = (e) => {
  const mode = $("input[name=mode]:checked").value;
  if (mode === "live") { $("#live-phrase").value = ""; $("#live-dialog").showModal(); return; }
  guard(e.target, () => saveSettings({ mode }, `Mode: ${mode}`));
};
$("#live-dialog").addEventListener("close", () => {
  if ($("#live-dialog").returnValue !== "ok") { fillBroker(); return; }
  guard(null, async () => {
    await api("/api/confirm-live", { phrase: $("#live-phrase").value });
    settings = await api("/api/settings"); fillAll(); refreshStatus(); toast("LIVE trading enabled");
  });
});

/* ------------------------------------------------------------- strategy */
function fillStrategy() {
  $("#symbols").value = settings.symbols.join(", ");
  ["capital", "product", "timeframe", "run_at", "square_off_at", "order_style", "limit_buffer_pct", "data_source"].forEach((k) => ($("#" + k).value = settings[k]));
  $("#holidays").value = (settings.holidays || []).join("\n");
  const sel = $("#strategy");
  sel.innerHTML = strategies.map((s) => `<option value="${s.name}">${s.name.replace(/_/g, " ")}</option>`).join("");
  sel.value = settings.strategy;
  renderParams(settings.strategy_params);
  $$("[data-risk]").forEach((el) => {
    const v = settings.risk[el.dataset.risk];
    if (el.type === "checkbox") el.checked = !!v;
    else el.value = v == null ? "" : el.hasAttribute("data-pct") ? +(v * 100).toFixed(2) : v;
  });
}
function renderParams(values = {}) {
  const s = strategies.find((x) => x.name === $("#strategy").value);
  if (!s) return;
  $("#strategy-doc").textContent = s.doc;
  $("#params").innerHTML = Object.entries(s.params).filter(([k]) => k !== "allow_short").map(([k, def]) => {
    const v = values[k] ?? def;
    return typeof def === "boolean"
      ? `<label class="check"><input type="checkbox" data-param="${k}" ${v ? "checked" : ""}> ${k.replace(/_/g, " ")}</label>`
      : `<label>${k.replace(/_/g, " ")} <input type="number" step="any" data-param="${k}" value="${v}"></label>`;
  }).join("") || '<p class="muted">No parameters.</p>';
}
$("#strategy").addEventListener("change", () => renderParams({}));
$("#strategy-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const params = {};
  $$("[data-param]").forEach((el) => (params[el.dataset.param] = el.type === "checkbox" ? el.checked : Number(el.value)));
  const risk = { ...settings.risk };
  $$("[data-risk]").forEach((el) => {
    if (el.type === "checkbox") risk[el.dataset.risk] = el.checked;
    else risk[el.dataset.risk] = el.value === "" ? null : el.hasAttribute("data-pct") ? Number(el.value) / 100 : Number(el.value);
  });
  if (risk.allow_short) params.allow_short = true;
  const ch = { symbols: $("#symbols").value, strategy: $("#strategy").value, strategy_params: params, risk,
    holidays: $("#holidays").value.split(/\s+/).filter(Boolean) };
  ["capital", "product", "timeframe", "run_at", "square_off_at", "order_style", "limit_buffer_pct", "data_source"].forEach((k) => (ch[k] = $("#" + k).value));
  guard(e.submitter, () => saveSettings(ch, "Strategy saved"));
});

/* ------------------------------------------------------------- backtest */
$("#btn-bt").onclick = (e) => guard(e.target, async () => {
  e.target.textContent = "Running…";
  try {
    const r = await api("/api/backtest", {});
    const m = r.metrics;
    const tiles = [["Total return", pct(m.total_return), m.total_return >= 0], ["Buy & hold", pct(m.benchmark_return), m.benchmark_return >= 0],
      ["CAGR", pct(m.cagr), m.cagr >= 0], ["Max drawdown", pct(m.max_drawdown), false], ["Sharpe", m.sharpe?.toFixed(2), m.sharpe >= 0],
      ["Win rate", pct(m.win_rate)], ["Trades", m.trades], ["Time in market", pct(m.exposure)]];
    $("#bt-metrics").innerHTML = tiles.map(([k, v, g]) => `<div class="stat"><span class="label">${k}</span><span class="value ${g === undefined ? "" : g ? "good" : "bad"}">${v ?? "—"}</span></div>`).join("");
    drawChart($("#bt-chart"), r.equity);
    $("#bt-trades tbody").innerHTML = r.trades.slice(-25).reverse().map((t) => `<tr><td>${esc(t.symbol)}</td><td>${String(t.entry_time).slice(0, 10)}</td><td>${String(t.exit_time).slice(0, 10)}</td><td class="num">${t.qty}</td><td class="num ${t.pnl >= 0 ? "good" : "bad"}">${Math.round(t.pnl).toLocaleString("en-IN")}</td><td>${esc(reasonText(t.exit_reason))}</td></tr>`).join("");
    $("#bt-out").hidden = false;
  } finally { e.target.textContent = "Run backtest"; }
});

/* ------------------------------------------------------------- activity */
async function loadJournal() {
  let rows; try { rows = await api("/api/journal?n=300"); } catch { return; }
  $("#journal").innerHTML = rows.reverse().map((r) => {
    const t = String(r.time || "").replace("T", " ").slice(0, 16);
    const what = r.symbol ? `<span class="tag">${esc(r.action)}</span><b>${esc(r.symbol)}</b> ${r.qty != null ? (r.qty > 0 ? "+" : "") + r.qty : ""} ${r.price ? "@ " + Number(r.price).toFixed(2) : ""} <span class="muted">${esc(reasonText(r.reason))}${r.note ? " · " + esc(r.note) : ""}</span>`
      : `<span class="tag">${esc(r.action)}</span><span class="${r.level === "error" ? "err" : ""}">${esc(r.reason)}</span>`;
    return `<li><span class="t">${t}</span><span>${what}</span></li>`;
  }).join("") || '<li class="muted">Nothing yet.</li>';
}
$("#pin-form").addEventListener("submit", (e) => {
  e.preventDefault();
  guard(e.submitter, async () => { await api("/api/pin", { pin: $("#new-pin").value }); $("#new-pin").value = ""; toast("PIN updated"); });
});

/* ----------------------------------------------------------------- init */
function fillAll() { fillBroker(); fillStrategy(); }
async function init() {
  if (!(await checkSession())) return;
  [settings, strategies] = await Promise.all([api("/api/settings"), api("/api/strategies")]);
  fillAll(); showTab(); refreshStatus();
  clearInterval(statusTimer); statusTimer = setInterval(() => { if (!document.hidden) refreshStatus(); }, 10000);
  const q = new URLSearchParams(location.search);
  if (q.get("login") === "ok") toast("Broker login successful");
  if (q.get("login_error")) toast("Login failed: " + q.get("login_error"), true);
  if (q.has("login") || q.has("login_error")) history.replaceState(null, "", location.pathname + location.hash);
}
if ("serviceWorker" in navigator && window.isSecureContext) navigator.serviceWorker.register("/sw.js").catch(() => {});
init();
