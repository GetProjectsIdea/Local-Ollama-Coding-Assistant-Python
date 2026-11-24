// Local Coder UI: streaming chat over Server-Sent Events, with saved local history.
(() => {
  const $ = (id) => document.getElementById(id);
  const HEAD = { "Content-Type": "application/json", "X-Local-Coder": "1" };
  const HINTS = {
    explain: "Paste code. Optional: ask about a specific line.",
    generate: "Describe what you want; the file name decides the language.",
    fix: "Paste the code AND the full error message.",
    docstring: "Paste a function or class to document.",
    tests: "Paste the code to test (pytest / JUnit / Jest picked automatically).",
    review: "Paste code for a friendly review.",
    chat: "Ask any programming question."
  };
  let mode = "explain", sessionId = null, controller = null;

  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  function inline(s) {
    return s.replace(/`([^`\n]+)`/g, "<code>$1</code>")
            .replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>")
            .replace(/(^|\s)_([^_\n]+)_(?=\s|$)/g, "$1<i>$2</i>");
  }
  // Tiny, safe markdown: everything is escaped first, then fences/inline marks are added.
  function render(md) {
    const parts = md.split(/```/);
    let html = "";
    parts.forEach((part, i) => {
      if (i % 2 === 1) {
        const nl = part.indexOf("\n");
        const lang = nl >= 0 ? part.slice(0, nl).trim() : "";
        const body = nl >= 0 ? part.slice(nl + 1) : part;
        html += `<pre data-lang="${esc(lang)}"><button class="copy" type="button">copy</button><code>${esc(body.replace(/\n$/, ""))}</code></pre>`;
      } else {
        const lines = esc(part).split("\n");
        let out = [], list = null;
        for (const line of lines) {
          const m = line.match(/^\s*(?:[-*]|\d+\.)\s+(.*)$/);
          if (m) { if (!list) { list = []; } list.push(`<li>${inline(m[1])}</li>`); continue; }
          if (list) { out.push(`<ul>${list.join("")}</ul>`); list = null; }
          const h = line.match(/^#{1,4}\s+(.*)$/);
          if (h) out.push(`<p><b>${inline(h[1])}</b></p>`);
          else if (line.trim()) out.push(`<p>${inline(line)}</p>`);
        }
        if (list) out.push(`<ul>${list.join("")}</ul>`);
        html += out.join("");
      }
    });
    return html;
  }

  function addMsg(role, md) {
    $("welcome")?.remove();
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    div.innerHTML = `<div class="who">${role === "user" ? "YOU" : "LOCAL CODER"}</div><div class="body"></div><div class="meta"></div>`;
    div.querySelector(".body").innerHTML = render(md || "");
    $("thread").appendChild(div);
    $("thread").scrollTop = $("thread").scrollHeight;
    return div;
  }

  function metaHtml(stats, checks, model) {
    const bits = [];
    if (model) bits.push(esc(model));
    if (stats && stats.output_tokens) bits.push(`${stats.output_tokens} tokens`);
    if (stats && stats.tokens_per_second) bits.push(`${stats.tokens_per_second} tok/s`);
    if (stats && stats.wall_seconds != null) bits.push(`${stats.wall_seconds} s`);
    let html = bits.map((b) => `<span>${b}</span>`).join("");
    (checks || []).forEach((c) => {
      const cls = c.ok === true ? "ok" : c.ok === false ? "bad" : "na";
      const txt = c.ok === true ? "syntax OK" : c.ok === false ? `line ${c.line}: ${esc(c.message)}` : "not checked";
      html += `<span class="badge ${cls}">block ${c.index} ${esc(c.language)}: ${txt}</span>`;
    });
    return html;
  }

  document.addEventListener("click", (e) => {
    if (e.target.classList.contains("copy")) {
      const code = e.target.parentElement.querySelector("code").textContent;
      navigator.clipboard?.writeText(code);
      e.target.textContent = "copied";
      setTimeout(() => (e.target.textContent = "copy"), 1200);
    }
  });

  function setMode(m) {
    mode = m;
    document.querySelectorAll(".mode").forEach((b) => b.classList.toggle("active", b.dataset.mode === m));
    $("error").classList.toggle("hidden", m !== "fix");
    $("modehint").textContent = HINTS[m] || "";
    $("question").placeholder = m === "generate" ? "e.g. a palindrome checker with comments" : "Ask a question (optional)";
  }
  $("modes").addEventListener("click", (e) => { if (e.target.dataset.mode) setMode(e.target.dataset.mode); });

  async function loadStatus() {
    try {
      const r = await fetch("/api/health"); const j = await r.json();
      $("status").textContent = j.ok ? `Ollama ${j.ollama} · ${j.backend}` : "Ollama offline";
      $("status").className = "pill " + (j.ok ? "ok" : "bad");
      if (!j.ok) $("status").title = j.hint || j.error;
    } catch { $("status").textContent = "server offline"; $("status").className = "pill bad"; }
    try {
      const r = await fetch("/api/models"); const j = await r.json();
      if (j.models && j.models.length) {
        $("model").innerHTML = j.models.map((m) =>
          `<option value="${esc(m.name)}" ${m.name === j.default ? "selected" : ""}>${esc(m.name)} (${m.size_gb} GB${m.parameters ? ", " + esc(m.parameters) : ""})</option>`).join("");
        if (!j.models.some((m) => m.name === j.default)) $("model").insertAdjacentHTML("afterbegin", `<option value="${esc(j.default)}">${esc(j.default)} (not installed)</option>`);
      }
    } catch {}
    try {
      const r = await fetch("/api/hardware"); const h = await r.json();
      $("hw").innerHTML = `<b>This laptop</b><br>${h.ram_gb} GB RAM · ${h.cpus} threads<br>${esc(h.tier.label)}<br>Best pick: <b>${esc(h.best)}</b><br>Suggested num_ctx: ${h.suggested_num_ctx}`;
    } catch {}
  }

  async function loadSessions() {
    const q = encodeURIComponent($("search").value || "");
    const r = await fetch(`/api/sessions?q=${q}`); const j = await r.json();
    $("sessions").innerHTML = j.sessions.map((s) =>
      `<li data-id="${s.id}" class="${s.id === sessionId ? "active" : ""}"><span>${esc(s.title)}<small>${esc(s.mode)} · ${esc(s.model)} · ${s.n} msgs</small></span><button class="del" title="delete" data-del="${s.id}">✕</button></li>`).join("");
  }
  $("search").addEventListener("input", loadSessions);
  $("sessions").addEventListener("click", async (e) => {
    const del = e.target.dataset.del;
    if (del) {
      e.stopPropagation();
      await fetch(`/api/sessions/${del}`, { method: "DELETE", headers: HEAD });
      if (String(sessionId) === del) newChat();
      return loadSessions();
    }
    const li = e.target.closest("li"); if (!li) return;
    const r = await fetch(`/api/sessions/${li.dataset.id}`); const s = await r.json();
    sessionId = s.id; $("thread").innerHTML = ""; setMode(s.mode);
    s.messages.forEach((m) => {
      const d = addMsg(m.role, m.content);
      if (m.role === "assistant") d.querySelector(".meta").innerHTML = metaHtml(m.meta.stats, m.meta.checks, m.meta.model);
    });
    loadSessions();
  });
  function newChat() { sessionId = null; $("thread").innerHTML = ""; loadSessions(); }
  $("newchat").addEventListener("click", newChat);

  $("stop").addEventListener("click", () => controller && controller.abort());

  $("form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const body = {
      mode, model: $("model").value, language: $("language").value, level: $("level").value,
      code: $("code").value, error: mode === "fix" ? $("error").value : "", question: $("question").value,
      filename: $("filename").value, session_id: sessionId
    };
    if (!body.code.trim() && !body.question.trim() && !body.error.trim()) return;
    let shown = body.question || "";
    if (body.code.trim()) shown += `\n\`\`\`\n${body.code.trim()}\n\`\`\``;
    if (body.error.trim()) shown += `\nError:\n\`\`\`\n${body.error.trim()}\n\`\`\``;
    const label = document.querySelector(`.mode[data-mode="${mode}"]`).textContent;
    addMsg("user", `**${label}** ${shown}`);
    const out = addMsg("assistant", "");
    const bodyEl = out.querySelector(".body");
    bodyEl.innerHTML = `<p class="thinking">thinking on your laptop…</p>`;
    $("send").disabled = true; $("stop").classList.remove("hidden");
    controller = new AbortController();
    let text = "";
    try {
      const resp = await fetch("/api/chat", { method: "POST", headers: HEAD, body: JSON.stringify(body), signal: controller.signal });
      if (!resp.ok) { const j = await resp.json().catch(() => ({})); throw new Error(j.error || resp.statusText); }
      const reader = resp.body.getReader(); const dec = new TextDecoder(); let buf = "";
      while (true) {
        const { value, done } = await reader.read(); if (done) break;
        buf += dec.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const chunk = buf.slice(0, idx); buf = buf.slice(idx + 2);
          const dataLine = chunk.split("\n").find((l) => l.startsWith("data: "));
          if (!dataLine) continue;
          const ev = JSON.parse(dataLine.slice(6));
          if (ev.type === "session") { sessionId = ev.session_id; }
          else if (ev.type === "token") { text += ev.text; bodyEl.innerHTML = render(text); $("thread").scrollTop = $("thread").scrollHeight; }
          else if (ev.type === "error") { bodyEl.innerHTML = render(text) + `<p class="err">✗ ${esc(ev.message)}${ev.hint ? "<br>hint: " + esc(ev.hint) : ""}</p>`; }
          else if (ev.type === "done") { bodyEl.innerHTML = render(ev.answer); out.querySelector(".meta").innerHTML = metaHtml(ev.stats, ev.checks, body.model); }
        }
      }
    } catch (err) {
      bodyEl.innerHTML = render(text) + `<p class="err">✗ ${esc(String(err.message || err))}</p>`;
    } finally {
      $("send").disabled = false; $("stop").classList.add("hidden"); controller = null; loadSessions();
    }
  });

  setMode("explain"); loadStatus(); loadSessions();
})();
