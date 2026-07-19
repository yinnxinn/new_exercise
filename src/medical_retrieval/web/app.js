const state = {
  tags: [],
  status: null,
};

const $ = selector => document.querySelector(selector);
const queryInput = $("#query");
const searchBtn = $("#searchBtn");
const streamBtn = $("#streamBtn");
const meta = $("#meta");
const results = $("#results");
const answer = $("#answer");
const vectorQuery = $("#vectorQuery");
const departmentFilter = $("#departmentFilter");
const tagFilter = $("#tagFilter");
const vectorSearchBtn = $("#vectorSearchBtn");
const vectorFilter = $("#vectorFilter");
const vectorResults = $("#vectorResults");
const tagTypeFilter = $("#tagTypeFilter");
const tagTable = $("#tagTable");
const loadTagsBtn = $("#loadTagsBtn");
const refreshStatusBtn = $("#refreshStatusBtn");

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"]/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[ch]));
}

function formatNumber(value) {
  if (value === undefined || value === null || value === "") return "-";
  return Number(value).toLocaleString("zh-CN");
}

function shortText(text, max = 180) {
  const value = String(text || "");
  return value.length > max ? `${value.slice(0, max)}...` : value;
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

function renderStatus(data) {
  state.status = data;
  const mysql = data.mysql || {};
  const milvus = data.milvus || {};
  $("#mysqlStatus").textContent = mysql.available ? "在线" : "不可用";
  $("#milvusStatus").textContent = milvus.available ? "在线" : "不可用";
  $("#docCount").textContent = formatNumber(mysql.medical_documents || data.files?.dialogue_docs);
  $("#tagCount").textContent = formatNumber(mysql.medical_tags || data.data_stats?.tags);

  $("#fileStats").innerHTML = [
    ["转换文档", data.files?.dialogue_docs],
    ["Milvus payload", data.files?.milvus_payload],
    ["源文件数", data.data_stats?.source_files],
    ["跳过行", data.data_stats?.skipped_rows],
    ["重复行", data.data_stats?.duplicate_rows],
  ].map(([key, value]) => `<div><span>${key}</span><strong>${formatNumber(value)}</strong></div>`).join("");

  $("#indexStats").innerHTML = [
    ["Collection", milvus.collection],
    ["向量数量", milvus.num_entities],
    ["维度", milvus.embedding_dim],
    ["模型", milvus.embedding_model],
  ].map(([key, value]) => `<div><span>${key}</span><strong>${escapeHtml(value ?? "-")}</strong></div>`).join("");
}

async function loadStatus() {
  try {
    renderStatus(await fetchJson("/system/status"));
  } catch (err) {
    $("#mysqlStatus").textContent = "异常";
    $("#milvusStatus").textContent = "异常";
  }
}

function renderTags(tags) {
  tagTable.innerHTML = tags.map(tag => `
    <button class="tag-row" data-code="${escapeHtml(tag.tag_code)}" data-type="${escapeHtml(tag.tag_type)}">
      <span class="tag-code">${escapeHtml(tag.tag_code)}</span>
      <span>${escapeHtml(tag.tag_name)}</span>
      <span class="tag-type">${escapeHtml(tag.tag_type)}</span>
      <strong>${formatNumber(tag.document_count)}</strong>
    </button>
  `).join("");

  tagTable.querySelectorAll(".tag-row").forEach(row => {
    row.addEventListener("click", () => {
      tagFilter.value = row.dataset.code;
      document.querySelector('[data-tab="vectorPanel"]').click();
      runVectorSearch();
    });
  });
}

function populateFilters(tags) {
  const departments = tags.filter(tag => tag.tag_type === "department");
  departmentFilter.innerHTML = '<option value="">全部科室</option>' + departments.map(tag =>
    `<option value="${escapeHtml(tag.tag_code)}">${escapeHtml(tag.tag_name)} (${formatNumber(tag.document_count)})</option>`
  ).join("");

  tagFilter.innerHTML = '<option value="">全部 Tag</option>' + tags.slice(0, 80).map(tag =>
    `<option value="${escapeHtml(tag.tag_code)}">${escapeHtml(tag.tag_code)} (${formatNumber(tag.document_count)})</option>`
  ).join("");
}

async function loadTags() {
  const type = tagTypeFilter.value;
  const data = await fetchJson(`/tags?limit=120${type ? `&tag_type=${encodeURIComponent(type)}` : ""}`);
  state.tags = data.tags || [];
  renderTags(state.tags);
  if (!type) populateFilters(state.tags);
}

async function runSearch() {
  const q = queryInput.value.trim();
  if (!q) return;
  meta.textContent = "检索中";
  const data = await fetchJson(`/answer?q=${encodeURIComponent(q)}&top_k=5`);
  meta.textContent = `意图：${(data.intents || []).join(", ") || "无"} | 实体：${data.entities || "无"}`;
  answer.textContent = data.answer || "";
  results.innerHTML = (data.sources || []).map((item, index) => `
    <article class="result-item">
      <div class="result-title">${index + 1}. ${escapeHtml(item.title)}</div>
      <div class="result-sub">${escapeHtml(item.id)} | ${escapeHtml(item.department)}</div>
    </article>
  `).join("");
}

async function runStream() {
  const q = queryInput.value.trim();
  if (!q) return;
  answer.textContent = "";
  results.innerHTML = "";
  meta.textContent = "正在流式输出";
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

async function runVectorSearch() {
  const q = vectorQuery.value.trim();
  if (!q) return;
  vectorResults.innerHTML = '<div class="empty">检索中</div>';
  const params = new URLSearchParams({ q, top_k: "8" });
  if (departmentFilter.value) params.set("department_tag", departmentFilter.value);
  if (tagFilter.value) params.set("tag_code", tagFilter.value);
  const data = await fetchJson(`/milvus/search?${params.toString()}`);
  vectorFilter.textContent = data.filter || data.error || "";
  vectorResults.innerHTML = (data.results || []).map((item, index) => `
    <article class="qa-item">
      <div class="qa-head">
        <strong>${index + 1}. ${escapeHtml(item.title || item.external_id)}</strong>
        <span>${escapeHtml(item.department)} · ${Number(item.score || 0).toFixed(4)}</span>
      </div>
      <p>${escapeHtml(shortText(item.question, 160))}</p>
      <div class="answer-snippet">${escapeHtml(shortText(item.answer, 260))}</div>
      <div class="result-sub">document_id=${escapeHtml(item.document_id)} · ${escapeHtml(item.department_tag)}</div>
    </article>
  `).join("") || '<div class="empty">没有结果</div>';
}

function bindTabs() {
  document.querySelectorAll(".tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(item => item.classList.remove("active"));
      document.querySelectorAll(".panel").forEach(item => item.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(tab.dataset.tab).classList.add("active");
    });
  });
}

searchBtn.addEventListener("click", runSearch);
streamBtn.addEventListener("click", runStream);
vectorSearchBtn.addEventListener("click", runVectorSearch);
loadTagsBtn.addEventListener("click", loadTags);
refreshStatusBtn.addEventListener("click", loadStatus);
queryInput.addEventListener("keydown", event => { if (event.key === "Enter") runSearch(); });
vectorQuery.addEventListener("keydown", event => { if (event.key === "Enter") runVectorSearch(); });

bindTabs();
loadStatus();
loadTags().then(runSearch).catch(() => runSearch());