// Simple Chat page logic with nice UI features
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const messages = document.getElementById("messages");
const filesPane = document.getElementById("filesPane");
const filesList = document.getElementById("filesList");

// NEW: Keep an in-memory record of the chat so we can send the full
// history to the backend on every request. We encode each turn as a
// simple object: { role: "user"|"assistant", text: string }.
// This state is only kept for the current page session.
let chatHistory = [];

// Helper to build base API URL
function getApiUrl() {
  return `http://${APP_CONFIG.api_host}:${APP_CONFIG.api_port}`;
}

// Search bar logic (debounced)
const docSearchInput = document.getElementById("docSearchInput");
let searchTimeout = null;
if (docSearchInput) {
  docSearchInput.addEventListener("input", () => {
    const query = docSearchInput.value.trim();

    // Debounce to avoid spamming API
    if (searchTimeout) clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      if (query.length === 0) {
        // Empty query → clear list & badge
        updateFiles([]);
        return;
      }

      const apiUrl = getApiUrl();
      fetch(`${apiUrl}/search_docs?query=${encodeURIComponent(query)}`)
        .then((resp) => {
          if (!resp.ok) throw new Error(`Search failed: ${resp.status}`);
          return resp.json();
        })
        .then((data) => {
          updateFiles(data.files || [], data.chunks || {});
        })
        .catch((err) => {
          console.error(err);
        });
    }, 300); // 300 ms debounce
  });
}

// Simple configuration
let APP_CONFIG = {
  api_host: "localhost",
  api_port: 8000,
  client_timeout: 30000,  // Increased timeout
  retrieval_k: 5
};

// Try to load config from server (but don't fail if it doesn't work)
async function loadConfig() {
  try {
    const response = await fetch("http://localhost:8000/config");
    if (response.ok) {
      APP_CONFIG = await response.json();
      console.log("Loaded config from server:", APP_CONFIG);
    }
  } catch (error) {
    console.log("Using default config (server not responding)");
  }
}

// Load config when page loads
loadConfig();

// Simple message adding function
function addMessage(text, role) {
  // Record message in history *only* if it has substantive content.
  if (role === "user" || role === "assistant") {
    const cleaned = (text || "").trim();
    if (cleaned.length > 0) {
      chatHistory.push({ role, text: cleaned });
    }
  }
  const div = document.createElement("div");
  div.classList.add("message", role);
  
  // Use simple text content - no HTML formatting
  div.textContent = text;
  
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}

// Nice loading message with dots animation
function addLoadingMessage() {
  const div = document.createElement("div");
  div.classList.add("message", "assistant", "loading");
  div.innerHTML = `
    <div class="loading-dots">
      <span></span><span></span><span></span>
    </div>
    <span class="loading-text">Searching and generating response...</span>
  `;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}

// Enhanced file display with chunks
function updateFiles(fileArr = [], chunks = null) {
  // Always keep sidebar visible
  filesList.innerHTML = "";
  
  // Update the file count badge in the header
  const fileCountBadge = document.getElementById("fileCount");
  if (fileCountBadge) {
    fileCountBadge.textContent = fileArr.length;
  }
  
  if (!fileArr || !fileArr.length) {
    filesList.innerHTML = '<div class="no-chunks">No documents</div>';
    return;
  }
  
  fileArr.forEach((fname, index) => {
    const fileItem = document.createElement("div");
    fileItem.className = "file-item";
    
    // Get chunks for this file if available
    const fileChunks = chunks && chunks[fname] ? chunks[fname] : [];
    
    const isImage = /\.(png|jpe?g|gif|bmp)$/i.test(fname);
    const apiUrl = getApiUrl();

    // Build image preview HTML if applicable
    const imagePreviewHtml = isImage
      ? `<div class="image-preview"><img src="${apiUrl}/files/${encodeURIComponent(fname)}" alt="${fname}" /></div>`
      : "";
    
    fileItem.innerHTML = `
      <div class="file-header" onclick="toggleFileChunks('file-${index}')">
        <div class="file-info">
          <span class="file-icon">📄</span>
          <span class="file-name">${fname}</span>
          <span class="file-rank">#${index + 1}</span>
        </div>
        <span class="expand-icon" id="expand-file-${index}">▼</span>
      </div>
      <div class="file-chunks" id="file-${index}" style="display: none;">
        ${imagePreviewHtml}
        ${fileChunks.length > 0 ? 
          fileChunks.map((chunk, i) => `
            <div class="chunk-item">
              <div class="chunk-header">
                <span class="chunk-label">Chunk ${i + 1}</span>
                <span class="chunk-score">Score: ${chunk.score ? chunk.score.toFixed(3) : 'N/A'}</span>
              </div>
              <div class="chunk-preview">
                ${chunk.text ? chunk.text.substring(0, 150) + (chunk.text.length > 150 ? '...' : '') : 'No content available'}
              </div>
              <button class="expand-chunk-btn" onclick="toggleChunk('chunk-${index}-${i}')">
                Show Full Chunk
              </button>
              <div class="chunk-full" id="chunk-${index}-${i}" style="display: none;">
                ${chunk.text || 'No content available'}
              </div>
            </div>
          `).join('')
          : '<div class="no-chunks">No chunk details available</div>'
        }
      </div>
    `;
    
    filesList.appendChild(fileItem);
  });
}

// Simple error display
function showError(message) {
  const errorDiv = addMessage("❌ Error: " + message, "assistant");
  errorDiv.classList.add("error");
}

// Main chat form handler
chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const prompt = chatInput.value.trim();
  if (!prompt) return;

  // Clear search bar (sidebar will be populated by RAG results)
  if (docSearchInput) docSearchInput.value = "";

  // Build a text representation of the *previous* history (i.e. all turns
  // except the current user prompt). We do this *before* pushing the new
  // user message so the current question is not duplicated in the history.
  const historyText = chatHistory
    .map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.text}`)
    .join("\n");

  // Add user message to UI and history
  addMessage(prompt, "user");
  chatInput.value = "";
  
  // Clear previous files
  updateFiles([]);

  // Add loading message with dots
  const loadingDiv = addLoadingMessage();
  
  // Create assistant message div (hidden initially)
  const assistantDiv = addMessage("", "assistant");
  assistantDiv.style.display = "none";

  // Set up event source with simple error handling
  const apiUrl = `http://${APP_CONFIG.api_host}:${APP_CONFIG.api_port}`;
  // NEW: pass the encoded chat history alongside the prompt so the backend
  // can build the full conversation context.
  const evtSrc = new EventSource(
    `${apiUrl}/chat_stream?prompt=${encodeURIComponent(prompt)}&history=${encodeURIComponent(historyText)}`
  );

  let hasStartedStreaming = false;
  let accumulatedText = '';
  
  // Connection timeout
  const connectionTimeout = setTimeout(() => {
    evtSrc.close();
    loadingDiv.remove();
    showError("Connection timeout. Please check if the server is running.");
  }, APP_CONFIG.client_timeout);

  // Handle metadata (files) - with full chunk support
  evtSrc.addEventListener("meta", (ev) => {
    try {
      const meta = JSON.parse(ev.data || "{}");
      updateFiles(meta.files || [], meta.chunks || {});
      console.log("Retrieved files:", meta.files);
    } catch (err) {
      console.warn("Failed to parse meta event:", err);
    }
  });

  // Handle streaming tokens
  evtSrc.addEventListener("token", (ev) => {
    if (!hasStartedStreaming) {
      hasStartedStreaming = true;
      loadingDiv.remove();
      assistantDiv.style.display = "block";
      clearTimeout(connectionTimeout);
    }
    
    accumulatedText += ev.data;
    // Simple text content - no formatting (but preserve line breaks)
    assistantDiv.textContent = accumulatedText;
    messages.scrollTop = messages.scrollHeight;
  });

  // Handle completion
  evtSrc.addEventListener("done", () => {
    evtSrc.close();
    clearTimeout(connectionTimeout);
    if (!hasStartedStreaming) {
      loadingDiv.remove();
      showError("No response received from the server.");
    } else {
      // When streaming completes successfully, persist the assistant's turn
      // into the chat history so it is available for the next round.
      chatHistory.push({ role: "assistant", text: accumulatedText });
    }
  });
  
  // Handle errors
  evtSrc.addEventListener("error", (ev) => {
    console.error("SSE Error:", ev);
    evtSrc.close();
    clearTimeout(connectionTimeout);
    loadingDiv.remove();
    showError("Connection failed. Please check if the server is running on http://localhost:8000");
  });
  
  evtSrc.onerror = () => {
    console.error("SSE connection error");
    evtSrc.close();
    clearTimeout(connectionTimeout);
    if (!hasStartedStreaming) {
      loadingDiv.remove();
      showError("Failed to connect to server. Please check if it's running.");
    }
  };
});

// Global functions for toggling file chunks
window.toggleFileChunks = function(fileId) {
  const chunksDiv = document.getElementById(fileId);
  const expandIcon = document.getElementById('expand-' + fileId);
  
  if (chunksDiv.style.display === 'none') {
    chunksDiv.style.display = 'block';
    expandIcon.textContent = '▲';
  } else {
    chunksDiv.style.display = 'none';
    expandIcon.textContent = '▼';
  }
};

window.toggleChunk = function(chunkId) {
  const chunkDiv = document.getElementById(chunkId);
  const button = event.target;
  
  if (chunkDiv.style.display === 'none') {
    chunkDiv.style.display = 'block';
    button.textContent = 'Hide Full Chunk';
  } else {
    chunkDiv.style.display = 'none';
    button.textContent = 'Show Full Chunk';
  }
};

// Ensure sidebar shows with zero documents on first load
document.addEventListener("DOMContentLoaded", () => {
  updateFiles([]);
});

// -----------------------------
// Inline file upload handling (top bar)
// -----------------------------

const uploadZone = document.getElementById("uploadZone");
const fileInputEl = document.getElementById("fileInput");
const uploadStatus = document.getElementById("uploadStatus");

// Create progress bar element
const progressContainer = document.createElement("div");
progressContainer.id = "progressContainer";
progressContainer.style.display = "none";
progressContainer.innerHTML = `
  <div class="progress-bar">
    <div class="progress-fill" id="progressFill"></div>
  </div>
  <div class="progress-text" id="progressText">Uploading...</div>
`;
uploadStatus.appendChild(progressContainer);

uploadZone.addEventListener("click", () => fileInputEl.click());

["dragover", "dragenter"].forEach((evt) => {
  uploadZone.addEventListener(evt, (e) => {
    e.preventDefault();
    uploadZone.classList.add("active");
  });
});

["dragleave", "drop"].forEach((evt) => {
  uploadZone.addEventListener(evt, (e) => {
    e.preventDefault();
    uploadZone.classList.remove("active");
  });
});

uploadZone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) {
    fileInputEl.files = e.dataTransfer.files;
    handleUpload();
  }
});

fileInputEl.addEventListener("change", handleUpload);

async function handleUpload() {
  const file = fileInputEl.files[0];
  if (!file) return;

  // Ensure progress elements exist (in case DOM was reset)
  if (!document.getElementById("progressText")) {
    uploadStatus.appendChild(progressContainer);
  }

  // Show progress bar
  progressContainer.style.display = "block";
  uploadStatus.style.color = "#3b82f6";

  // Always query after (re-)attachment
  const progressFill = document.getElementById("progressFill");
  const progressText = document.getElementById("progressText");
  if (!progressFill || !progressText) {
    console.error("Progress elements missing – aborting upload UI update");
  }

  // Simulate progress for visual feedback
  let progress = 0;
  const progressInterval = setInterval(() => {
    progress += Math.random() * 30;
    if (progress > 90) progress = 90;
    progressFill.style.width = progress + "%";
    progressText.textContent = `Uploading... ${Math.round(progress)}%`;
  }, 200);

  const formData = new FormData();
  formData.append("file", file);

  try {
    progressText.textContent = "Processing file...";

    const apiUrl = getApiUrl();
    const res = await fetch(`${apiUrl}/upload`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }

    const data = await res.json();

    // Complete progress
    clearInterval(progressInterval);
    progressFill.style.width = "100%";
    progressText.textContent = "Upload complete!";

    setTimeout(() => {
      progressContainer.style.display = "none";
      uploadStatus.style.color = "#059669";
      const vectorsText = data.vectors_added ? ` (${data.vectors_added} chunks processed)` : "";
      uploadStatus.textContent = `✓ ${data.filename} processed${vectorsText}`;

      // Refresh search bar results if user typed something earlier
      if (docSearchInput && docSearchInput.value.trim().length > 0) {
        docSearchInput.dispatchEvent(new Event("input"));
      }
    }, 1000);

  } catch (err) {
    clearInterval(progressInterval);
    progressContainer.style.display = "none";
    uploadStatus.style.color = "#dc2626";
    uploadStatus.textContent = "❌ Upload failed: " + err.message;
    console.error("Upload error:", err);
  }
} 