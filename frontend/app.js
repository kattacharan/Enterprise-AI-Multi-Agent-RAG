// Enterprise Knowledge Assistant - Frontend State & Controller
let currentConversationId = null;
let selectedUploadFile = null;

// DOM Elements
const conversationListEl = document.getElementById("conversation-list");
const kbDocumentListEl = document.getElementById("kb-document-list");
const messagesWrapperEl = document.getElementById("messages-wrapper");
const messagesListEl = document.getElementById("messages-list");
const welcomeHeroEl = document.getElementById("welcome-hero");
const chatFormEl = document.getElementById("chat-form");
const queryInputEl = document.getElementById("query-input");
const btnNewChatEl = document.getElementById("btn-new-chat");
const btnReindexEl = document.getElementById("btn-reindex");
const vectorCountEl = document.getElementById("vector-count");
const docCountEl = document.getElementById("doc-count");

// Upload Modal Elements
const uploadModalEl = document.getElementById("upload-modal");
const btnUploadModalEl = document.getElementById("btn-upload-modal");
const btnCloseModalEl = document.getElementById("btn-close-modal");
const btnCancelUploadEl = document.getElementById("btn-cancel-upload");
const btnConfirmUploadEl = document.getElementById("btn-confirm-upload");
const dropZoneEl = document.getElementById("drop-zone");
const fileInputEl = document.getElementById("file-input");
const selectedFileInfoEl = document.getElementById("selected-file-info");
const selectedFilenameEl = document.getElementById("selected-filename");
const selectedFilesizeEl = document.getElementById("selected-filesize");
const uploadProgressEl = document.getElementById("upload-progress");

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
  lucide.createIcons();
  setupEventListeners();
  loadConversations();
  loadKnowledgeBaseDocs();
});

function setupEventListeners() {
  // Chat submission
  chatFormEl.addEventListener("submit", handleChatSubmit);

  // Auto-grow textarea
  queryInputEl.addEventListener("input", () => {
    queryInputEl.style.height = "auto";
    queryInputEl.style.height = queryInputEl.scrollHeight + "px";
  });

  // Enter to submit (Shift+Enter for newline)
  queryInputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatFormEl.dispatchEvent(new Event("submit"));
    }
  });

  // New Chat
  btnNewChatEl.addEventListener("click", () => {
    currentConversationId = null;
    welcomeHeroEl.classList.remove("hidden");
    messagesWrapperEl.classList.add("hidden");
    messagesListEl.innerHTML = "";
    document.querySelectorAll(".conv-item").forEach((el) => el.classList.remove("active"));
  });

  // Reindex
  btnReindexEl.addEventListener("click", handleReindex);

  // Sample Query Chips
  document.querySelectorAll(".query-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-query");
      queryInputEl.value = q;
      chatFormEl.dispatchEvent(new Event("submit"));
    });
  });

  // Modal Controls
  btnUploadModalEl.addEventListener("click", () => {
    uploadModalEl.classList.remove("hidden");
    resetUploadModal();
  });

  btnCloseModalEl.addEventListener("click", () => uploadModalEl.classList.add("hidden"));
  btnCancelUploadEl.addEventListener("click", () => uploadModalEl.classList.add("hidden"));

  // Drag & Drop
  dropZoneEl.addEventListener("click", () => fileInputEl.click());
  fileInputEl.addEventListener("change", (e) => handleFileSelect(e.target.files[0]));

  ["dragenter", "dragover"].forEach((eventName) => {
    dropZoneEl.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZoneEl.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZoneEl.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZoneEl.classList.remove("dragover");
    });
  });

  dropZoneEl.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  });

  btnConfirmUploadEl.addEventListener("click", handleUploadConfirm);
}

// Conversations Management
async function loadConversations() {
  try {
    const res = await fetch("/api/conversations");
    const data = await res.json();
    conversationListEl.innerHTML = "";

    if (!data || data.length === 0) {
      conversationListEl.innerHTML = `<div class="empty-state-sm">No previous conversations</div>`;
      return;
    }

    data.forEach((conv) => {
      const item = document.createElement("div");
      item.className = `conv-item ${conv.id === currentConversationId ? "active" : ""}`;
      item.innerHTML = `
        <span class="conv-title" title="${conv.title}">${escapeHtml(conv.title)}</span>
        <button class="btn-delete-conv" title="Delete conversation" data-id="${conv.id}">
          <i data-lucide="trash-2"></i>
        </button>
      `;

      item.addEventListener("click", (e) => {
        if (e.target.closest(".btn-delete-conv")) return;
        selectConversation(conv.id);
      });

      const delBtn = item.querySelector(".btn-delete-conv");
      delBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        await deleteConversation(conv.id);
      });

      conversationListEl.appendChild(item);
    });
    lucide.createIcons();
  } catch (err) {
    console.error("Error loading conversations:", err);
  }
}

async function selectConversation(convId) {
  currentConversationId = convId;
  document.querySelectorAll(".conv-item").forEach((el) => el.classList.remove("active"));
  
  try {
    const res = await fetch(`/api/conversations/${convId}`);
    const data = await res.json();
    
    welcomeHeroEl.classList.add("hidden");
    messagesWrapperEl.classList.remove("hidden");
    messagesListEl.innerHTML = "";

    if (data.messages && data.messages.length > 0) {
      data.messages.forEach((msg) => {
        if (msg.role === "user") {
          renderUserMessage(msg.content);
        } else {
          renderAssistantMessage(msg.content, msg.citations, msg.agent_steps, msg.verification_score);
        }
      });
    }
    loadConversations();
    scrollToBottom();
  } catch (err) {
    console.error("Error fetching conversation history:", err);
  }
}

async function deleteConversation(convId) {
  try {
    await fetch(`/api/conversations/${convId}`, { method: "DELETE" });
    if (currentConversationId === convId) {
      currentConversationId = null;
      welcomeHeroEl.classList.remove("hidden");
      messagesWrapperEl.classList.add("hidden");
      messagesListEl.innerHTML = "";
    }
    loadConversations();
  } catch (err) {
    console.error("Error deleting conversation:", err);
  }
}

// Knowledge Base Documents
async function loadKnowledgeBaseDocs() {
  try {
    const res = await fetch("/api/documents");
    const data = await res.json();

    vectorCountEl.textContent = data.total_vectors_in_faiss || 0;
    docCountEl.textContent = data.total_documents || 0;
    kbDocumentListEl.innerHTML = "";

    if (!data.documents || data.documents.length === 0) {
      kbDocumentListEl.innerHTML = `<div class="empty-state-sm">Knowledge base is empty</div>`;
      return;
    }

    data.documents.forEach((doc) => {
      const item = document.createElement("div");
      item.className = "kb-item";
      const sizeKb = Math.round((doc.file_size_bytes || 0) / 1024);
      item.innerHTML = `
        <i data-lucide="file-code" class="kb-item-icon"></i>
        <div class="kb-item-info">
          <div class="kb-item-name" title="${doc.filename}">${escapeHtml(doc.filename)}</div>
          <div class="kb-item-meta">${doc.chunk_count} chunks • ${sizeKb} KB</div>
        </div>
      `;
      kbDocumentListEl.appendChild(item);
    });
    lucide.createIcons();
  } catch (err) {
    console.error("Error loading documents:", err);
  }
}

async function handleReindex() {
  btnReindexEl.disabled = true;
  btnReindexEl.innerHTML = `<i data-lucide="loader-2" class="spin"></i> <span>Reindexing...</span>`;
  lucide.createIcons();

  try {
    await fetch("/api/documents/reindex", { method: "POST" });
    await loadKnowledgeBaseDocs();
  } catch (err) {
    console.error("Reindex error:", err);
  } finally {
    btnReindexEl.disabled = false;
    btnReindexEl.innerHTML = `<i data-lucide="refresh-cw"></i> <span>Reindex</span>`;
    lucide.createIcons();
  }
}

// Chat Query Execution
async function handleChatSubmit(e) {
  e.preventDefault();
  const query = queryInputEl.value.trim();
  if (!query) return;

  queryInputEl.value = "";
  queryInputEl.style.height = "auto";

  // Hide welcome hero, reveal stream
  welcomeHeroEl.classList.add("hidden");
  messagesWrapperEl.classList.remove("hidden");

  // Render User Message
  renderUserMessage(query);

  // Render Loading Assistant Card with Live Step Indicator
  const loadingCard = createLoadingCard();
  messagesListEl.appendChild(loadingCard);
  scrollToBottom();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        conversation_id: currentConversationId
      })
    });

    if (!res.ok) {
      throw new Error(`API error: ${res.statusText}`);
    }

    const data = await res.json();
    currentConversationId = data.conversation_id;

    // Remove loading card
    loadingCard.remove();

    // Render Completed Assistant Message
    renderAssistantMessage(
      data.response,
      data.citations,
      data.agent_steps,
      data.verification_score,
      data.verification_status
    );

    // Refresh conversation list to update titles
    loadConversations();
  } catch (err) {
    console.error("Chat error:", err);
    loadingCard.remove();
    renderAssistantMessage(
      `⚠️ **An error occurred while running the multi-agent workflow:**\n\`${err.message}\``,
      [],
      [],
      0.0,
      "ERROR"
    );
  }
}

// Rendering UI Cards
function renderUserMessage(text) {
  const card = document.createElement("div");
  card.className = "message-card";
  card.innerHTML = `
    <div class="avatar avatar-user">
      <i data-lucide="user"></i>
    </div>
    <div class="message-body">
      <div class="message-header">
        <span class="sender-name">You</span>
      </div>
      <div class="message-content">
        <p>${escapeHtml(text)}</p>
      </div>
    </div>
  `;
  messagesListEl.appendChild(card);
  lucide.createIcons();
  scrollToBottom();
}

function createLoadingCard() {
  const card = document.createElement("div");
  card.className = "message-card";
  card.innerHTML = `
    <div class="avatar avatar-ai">
      <i data-lucide="cpu"></i>
    </div>
    <div class="message-body">
      <div class="message-header">
        <span class="sender-name">Enterprise Assistant</span>
        <span class="verification-badge badge-partial">
          <i data-lucide="loader-2" class="spin"></i> Executing LangGraph Agents...
        </span>
      </div>
      <div class="message-content">
        <p style="color: var(--text-muted); font-style: italic;">
          Coordinating Planner ➔ FAISS Retriever ➔ Hallucination Verifier ➔ Synthesizer...
        </p>
      </div>
    </div>
  `;
  lucide.createIcons();
  return card;
}

function renderAssistantMessage(content, citations = [], steps = [], verificationScore = null, verificationStatus = "PASSED") {
  const card = document.createElement("div");
  card.className = "message-card";

  // Parse markdown
  const formattedHtml = marked.parse(content || "");

  // Verification Badge
  let badgeHtml = "";
  if (verificationScore !== null && verificationScore !== undefined) {
    const isVerified = verificationScore >= 0.6;
    badgeHtml = `
      <span class="verification-badge ${isVerified ? "badge-verified" : "badge-partial"}">
        <i data-lucide="${isVerified ? "check-circle-2" : "alert-circle"}"></i>
        ${isVerified ? "Verified Grounded" : "Partial Match"} (Score: ${verificationScore})
      </span>
    `;
  }

  // Citations Section
  let citationsHtml = "";
  if (citations && citations.length > 0) {
    const citationCards = citations.map(c => `
      <div class="citation-card">
        <div class="citation-card-header">
          <span class="citation-number">[${c.index}] ${escapeHtml(c.filename)}</span>
          <span class="citation-score">Relevance: ${c.score}</span>
        </div>
        <div class="citation-snippet">"${escapeHtml(c.snippet)}"</div>
      </div>
    `).join("");

    citationsHtml = `
      <div class="citations-box">
        <div class="citations-box-title">
          <i data-lucide="bookmark"></i> Source Citations & References (${citations.length})
        </div>
        <div class="citation-chips-container">
          ${citationCards}
        </div>
      </div>
    `;
  }

  // Multi-Agent Execution Steps Section
  let stepsHtml = "";
  if (steps && steps.length > 0) {
    const stepItems = steps.map((s, idx) => `
      <div class="timeline-step">
        <div class="step-marker">${idx + 1}</div>
        <div class="step-details">
          <span class="step-agent">${escapeHtml(s.agent)} — <span style="font-weight:400;color:var(--text-muted);">${escapeHtml(s.action)} (${s.timestamp})</span></span>
          <span class="step-thought">${escapeHtml(s.thought)}</span>
        </div>
      </div>
    `).join("");

    stepsHtml = `
      <div class="agent-steps-timeline">
        <div class="timeline-toggle" onclick="this.nextElementSibling.classList.toggle('hidden')">
          <span><i data-lucide="activity"></i> Multi-Agent Execution Trace (${steps.length} Steps)</span>
          <i data-lucide="chevron-down"></i>
        </div>
        <div class="timeline-content hidden">
          ${stepItems}
        </div>
      </div>
    `;
  }

  card.innerHTML = `
    <div class="avatar avatar-ai">
      <i data-lucide="brain-circuit"></i>
    </div>
    <div class="message-body">
      <div class="message-header">
        <span class="sender-name">Enterprise Assistant</span>
        ${badgeHtml}
      </div>
      <div class="message-content">
        ${formattedHtml}
      </div>
      ${citationsHtml}
      ${stepsHtml}
    </div>
  `;

  messagesListEl.appendChild(card);
  lucide.createIcons();
  scrollToBottom();
}

// Upload Handling
function handleFileSelect(file) {
  if (!file) return;
  selectedUploadFile = file;
  selectedFilenameEl.textContent = file.name;
  const sizeKb = Math.round(file.size / 1024);
  selectedFilesizeEl.textContent = `(${sizeKb} KB)`;
  selectedFileInfoEl.classList.remove("hidden");
  btnConfirmUploadEl.disabled = false;
}

async function handleUploadConfirm() {
  if (!selectedUploadFile) return;

  btnConfirmUploadEl.disabled = true;
  uploadProgressEl.classList.remove("hidden");

  const formData = new FormData();
  formData.append("file", selectedUploadFile);

  try {
    const res = await fetch("/api/documents/upload", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Upload failed");
    }

    await loadKnowledgeBaseDocs();
    uploadModalEl.classList.add("hidden");
    resetUploadModal();
  } catch (err) {
    alert(`Upload Error: ${err.message}`);
    uploadProgressEl.classList.add("hidden");
    btnConfirmUploadEl.disabled = false;
  }
}

function resetUploadModal() {
  selectedUploadFile = null;
  fileInputEl.value = "";
  selectedFileInfoEl.classList.add("hidden");
  uploadProgressEl.classList.add("hidden");
  btnConfirmUploadEl.disabled = true;
}

// Helpers
function scrollToBottom() {
  const container = document.querySelector(".chat-container");
  container.scrollTop = container.scrollHeight;
}

function escapeHtml(text) {
  if (!text) return "";
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
