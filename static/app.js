/* Chat front end.

   Everything here is display logic - the browser holds the conversation and
   only sends the current question to /api/ask. Nothing is re-rendered between
   messages, which is the whole reason the animations stay smooth. */

const messages = document.getElementById("messages");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const showScores = document.getElementById("show-scores");
const starters = document.getElementById("starters");
const sidebar = document.getElementById("sidebar");

// A reply usually comes back in a few milliseconds. Without a floor, the typing
// indicator appears and vanishes in the same frame, which reads as a glitch
// rather than as a response.
const MIN_TYPING_MS = 420;

let busy = false;

function scrollToEnd() {
  messages.scrollTo({ top: messages.scrollHeight, behavior: "smooth" });
}

function addUserMessage(text) {
  const el = document.createElement("div");
  el.className = "msg user";
  el.innerHTML = `<div class="avatar">You</div><div class="bubble-wrap"><div class="bubble"></div></div>`;
  // textContent, never innerHTML, for anything the user typed - otherwise
  // typing an HTML tag into the box would inject it into the page.
  el.querySelector(".bubble").textContent = text;
  messages.appendChild(el);
  scrollToEnd();
}

function addTypingIndicator() {
  const el = document.createElement("div");
  el.className = "msg bot";
  el.innerHTML = `<div class="avatar">AI</div><div class="bubble-wrap">
      <div class="bubble typing"><i></i><i></i><i></i></div></div>`;
  messages.appendChild(el);
  scrollToEnd();
  return el;
}

function addBotMessage(data) {
  const el = document.createElement("div");
  el.className = data.confident ? "msg bot" : "msg bot unsure";
  el.innerHTML = `<div class="avatar">AI</div><div class="bubble-wrap"><div class="bubble"></div></div>`;

  const wrap = el.querySelector(".bubble-wrap");
  wrap.querySelector(".bubble").textContent = data.text;

  if (showScores.checked) {
    const meta = document.createElement("div");
    meta.className = "meta";

    const bar = document.createElement("span");
    bar.className = "score-bar";
    const fill = document.createElement("span");
    fill.className = data.confident ? "score-fill" : "score-fill low";
    bar.appendChild(fill);

    const label = document.createElement("span");
    label.textContent = data.matched_question
      ? `${data.score.toFixed(3)} · matched: “${data.matched_question}”`
      : `${data.score.toFixed(3)} · below threshold`;

    meta.append(bar, label);
    wrap.appendChild(meta);

    // Set the width on the next frame so the browser has painted the 0-width
    // state first and has something to animate away from.
    requestAnimationFrame(() => {
      fill.style.width = `${Math.min(data.score, 1) * 100}%`;
    });
  }

  if (data.suggestions && data.suggestions.length) {
    const box = document.createElement("div");
    box.className = "suggestions";
    data.suggestions.forEach((question) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = question;
      btn.addEventListener("click", () => submitQuestion(question));
      box.appendChild(btn);
    });
    wrap.appendChild(box);
  }

  messages.appendChild(el);
  scrollToEnd();
}

async function submitQuestion(text) {
  const question = text.trim();
  if (!question || busy) return;

  busy = true;
  sendBtn.disabled = true;
  starters.classList.add("gone");
  input.value = "";

  addUserMessage(question);
  const typing = addTypingIndicator();
  const startedAt = performance.now();

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) throw new Error(`server responded ${res.status}`);
    const data = await res.json();

    const elapsed = performance.now() - startedAt;
    await new Promise((r) => setTimeout(r, Math.max(0, MIN_TYPING_MS - elapsed)));

    typing.remove();
    addBotMessage(data);
  } catch (err) {
    typing.remove();
    addBotMessage({
      text: `Could not reach the server (${err.message}). Is app.py still running?`,
      confident: false,
      score: 0,
      matched_question: null,
      suggestions: [],
    });
  } finally {
    busy = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuestion(input.value);
});

// Starter chips and the sidebar's question list both just ask the question.
document.querySelectorAll(".starters button, .q-link").forEach((btn) => {
  btn.addEventListener("click", () => {
    submitQuestion(btn.textContent);
    sidebar.classList.remove("open");
  });
});

document.getElementById("menu-btn").addEventListener("click", () => {
  sidebar.classList.toggle("open");
});

input.focus();
