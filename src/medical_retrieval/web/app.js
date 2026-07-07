const queryInput = document.querySelector("#query");
const searchBtn = document.querySelector("#searchBtn");
const streamBtn = document.querySelector("#streamBtn");
const meta = document.querySelector("#meta");
const results = document.querySelector("#results");
const answer = document.querySelector("#answer");

function escapeHtml(text) {
  return String(text).replace(/[&<>"]/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[ch]));
}

async function runSearch() {
  const q = queryInput.value.trim();
  if (!q) return;
  const res = await fetch(`/answer?q=${encodeURIComponent(q)}&top_k=5`);
  const data = await res.json();
  meta.textContent = `意图：${data.intents.join(", ")} | 实体：${data.entities}`;
  answer.textContent = data.answer;
  results.innerHTML = data.sources.map((item, index) => `
    <div class="result-item">
      <div class="result-title">${index + 1}. ${escapeHtml(item.title)}</div>
      <div class="result-sub">${escapeHtml(item.id)} | ${escapeHtml(item.department)}</div>
    </div>
  `).join("");
}

async function runStream() {
  const q = queryInput.value.trim();
  if (!q) return;
  answer.textContent = "";
  results.innerHTML = "";
  meta.textContent = "正在流式生成...";
  const res = await fetch(`/stream?q=${encodeURIComponent(q)}&top_k=5`);
  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    for (const line of chunk.split("\n")) {
      if (!line.startsWith("data: ")) continue;
      const payload = JSON.parse(line.slice(6));
      answer.textContent += payload.text;
      if (payload.done) meta.textContent = "流式输出完成";
    }
  }
}

searchBtn.addEventListener("click", runSearch);
streamBtn.addEventListener("click", runStream);
runSearch();
