const urlInput = document.getElementById("url-input");
const processBtn = document.getElementById("process-btn");
const statusRow = document.getElementById("status-row");
const progressFill = document.getElementById("progress-fill");
const statusText = document.getElementById("status-text");
const errorText = document.getElementById("error-text");
const chatPanel = document.getElementById("chat-panel");
const chatLog = document.getElementById("chat-log");
const questionInput = document.getElementById("question-input");
const askBtn = document.getElementById("ask-btn");
const resetBtn = document.getElementById("reset-btn");
const inputPanel = document.getElementById("input-panel");

let currentVideoId = null;
let pollTimer = null;

function showError(msg) {
  errorText.textContent = msg;
  errorText.classList.remove("hidden");
}

function clearError() {
  errorText.classList.add("hidden");
  errorText.textContent = "";
}

async function processVideo() {
  const url = urlInput.value.trim();
  if (!url) return;

  clearError();
  processBtn.disabled = true;
  statusRow.classList.remove("hidden");
  progressFill.style.width = "0%";
  statusText.textContent = "queued";
  chatPanel.classList.add("hidden");

  try {
    const res = await fetch("/api/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    if (!res.ok) throw new Error(`Server error (${res.status})`);
    const data = await res.json();
    currentVideoId = data.video_id;
    pollStatus();
  } catch (err) {
    showError(err.message);
    processBtn.disabled = false;
  }
}

function pollStatus() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/status/${currentVideoId}`);
      if (!res.ok) throw new Error("Lost track of processing job");
      const job = await res.json();

      progressFill.style.width = `${job.pct}%`;
      statusText.textContent = job.stage;

      if (job.error) {
        clearInterval(pollTimer);
        showError(job.error);
        processBtn.disabled = false;
        return;
      }

      if (job.done) {
        clearInterval(pollTimer);
        processBtn.disabled = false;
        statusRow.classList.add("hidden");
        inputPanel.classList.add("hidden");
        chatPanel.classList.remove("hidden");
        if (chatLog.children.length === 0) {
          await loadHistory();
        }
        questionInput.focus();
      }
    } catch (err) {
      clearInterval(pollTimer);
      showError(err.message);
      processBtn.disabled = false;
    }
  }, 1200);
}

async function loadHistory() {
  try {
    const res = await fetch(`/api/history/${currentVideoId}`);
    if (!res.ok) throw new Error("Could not load history");
    const turns = await res.json();

    if (turns.length === 0) {
      addMessage("bot", "Video processed. Ask me anything about it.");
      return;
    }

    addMessage("bot", `Welcome back — showing ${turns.length} previous question${turns.length === 1 ? "" : "s"} about this lecture.`);
    turns.forEach((turn) => {
      addMessage("user", turn.question);
      addMessage("bot", turn.answer);
    });
  } catch (err) {
    // history is a nice-to-have; don't block the chat if it fails to load
    addMessage("bot", "Video processed. Ask me anything about it.");
  }
}

function timeLabel() {
  const d = new Date();
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function addMessage(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `msg-wrap ${role.replace(" loading", "")}`;

  const tag = document.createElement("div");
  tag.className = "msg-tag";
  tag.textContent = role.startsWith("user") ? `you · ${timeLabel()}` : `lecture rag · ${timeLabel()}`;

  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;

  wrap.appendChild(tag);
  wrap.appendChild(div);
  chatLog.appendChild(wrap);
  chatLog.scrollTop = chatLog.scrollHeight;
  return div;
}

async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question || !currentVideoId) return;

  addMessage("user", question);
  questionInput.value = "";
  askBtn.disabled = true;
  const loadingMsg = addMessage("bot loading", "Thinking");

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: currentVideoId, question }),
    });
    if (!res.ok) throw new Error(`Server error (${res.status})`);
    const data = await res.json();
    loadingMsg.textContent = data.answer;
    loadingMsg.classList.remove("loading");
  } catch (err) {
    loadingMsg.textContent = `Error: ${err.message}`;
    loadingMsg.classList.remove("loading");
  } finally {
    askBtn.disabled = false;
  }
}

function resetToInput() {
  currentVideoId = null;
  chatLog.innerHTML = "";
  chatPanel.classList.add("hidden");
  inputPanel.classList.remove("hidden");
  urlInput.value = "";
  clearError();
  urlInput.focus();
}

processBtn.addEventListener("click", processVideo);
urlInput.addEventListener("keydown", (e) => { if (e.key === "Enter") processVideo(); });
askBtn.addEventListener("click", askQuestion);
questionInput.addEventListener("keydown", (e) => { if (e.key === "Enter") askQuestion(); });
resetBtn.addEventListener("click", resetToInput);