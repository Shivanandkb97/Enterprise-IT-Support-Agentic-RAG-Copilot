
const chat = document.getElementById('chat');
const form = document.getElementById('chatForm');
const question = document.getElementById('question');
const trace = document.getElementById('trace');
const sourceUsed = document.getElementById('sourceUsed');

// Escape HTML for plain text content
function escapeHtml(s = '') {
    return String(s).replace(/[&<>'"]/g, c => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        "'": '&#39;',
        '"': '&quot;'
    }[c]));
}

// Render Markdown safely
function formatText(s = '') {
    if (typeof marked === 'undefined' || typeof DOMPurify === 'undefined') {
        return escapeHtml(String(s)).replace(/\n/g, '<br>');
    }

    const html = marked.parse(String(s));

    return DOMPurify.sanitize(html, {
        USE_PROFILES: { html: true }
    });
}

// Add a normal message
function addMessage(role, text, source = '', citations = []) {
    const wrap = document.createElement('div');
    wrap.className = `message ${role}`;

    const citeHtml = citations.length
        ? `<div class="citations">
            <strong>Sources</strong><br>
            ${citations.map(c => {
                const title = escapeHtml(c.title || 'Source');
                const url = c.url || '';

                if (url) {
                    try {
                        const parsedUrl = new URL(url, window.location.origin);

                        if (['http:', 'https:'].includes(parsedUrl.protocol)) {
                            return `<a href="${escapeHtml(parsedUrl.href)}" target="_blank" rel="noopener noreferrer">${title}</a>`;
                        }
                    } catch {
                        // Invalid URL
                    }
                }

                return title;
            }).join('<br>')}
        </div>`
        : '';

    wrap.innerHTML = `
        <div class="avatar">${role === 'user' ? 'You' : 'AI'}</div>
        <div class="bubble">
            ${formatText(text)}
            ${source ? `<div class="answer-source">Source: ${escapeHtml(source)}</div>` : ''}
            ${citeHtml}
        </div>
    `;

    chat.appendChild(wrap);
    chat.scrollTop = chat.scrollHeight;

    return wrap;
}

// Create assistant message with animated typing dots
function createStreamingMessage() {
    const wrap = document.createElement('div');
    wrap.className = 'message assistant';

    wrap.innerHTML = `
        <div class="avatar">AI</div>
        <div class="bubble">
            <div class="thinking-status">
                <span class="thinking-label">Understanding your question...</span>

                <span class="typing-dots" aria-label="Assistant is thinking">
                    <span></span>
                    <span></span>
                    <span></span>
                </span>
            </div>

            <div class="streaming-content"></div>
        </div>
    `;

    chat.appendChild(wrap);
    chat.scrollTop = chat.scrollHeight;

    return {
        wrap,
        bubble: wrap.querySelector('.bubble'),
        status: wrap.querySelector('.thinking-status'),
        statusLabel: wrap.querySelector('.thinking-label'),
        content: wrap.querySelector('.streaming-content')
    };
}

// Update assistant status
function updateStatus(message, ui) {
    if (message && ui.statusLabel) {
        ui.statusLabel.textContent = message;
        ui.status.style.display = 'flex';
    }

    chat.scrollTop = chat.scrollHeight;
}

// Append generated tokens
function appendToken(token, ui, answer) {
    answer.value += token;

    ui.content.innerHTML = formatText(answer.value);

    updateStatus('Generating your answer...', ui);

    chat.scrollTop = chat.scrollHeight;

    return answer.value;
}

// Add final source and citations
function finishStreaming(data, ui) {
    if (ui.status && ui.status.isConnected) {
        ui.status.remove();
    }

    if (data.source_used) {
        const source = document.createElement('div');
        source.className = 'answer-source';
        source.textContent = `Source: ${data.source_used}`;
        ui.bubble.appendChild(source);
    }

    const citations = data.citations || data.citation || [];

    if (citations.length) {
        const citeDiv = document.createElement('div');
        citeDiv.className = 'citations';

        const title = document.createElement('strong');
        title.textContent = 'Sources';

        citeDiv.appendChild(title);
        citeDiv.appendChild(document.createElement('br'));

        citations.forEach(c => {
            const url = c.url || '';
            const text = c.title || 'Source';

            if (url) {
                try {
                    const parsedUrl = new URL(url, window.location.origin);

                    if (['http:', 'https:'].includes(parsedUrl.protocol)) {
                        const link = document.createElement('a');

                        link.href = parsedUrl.href;
                        link.target = '_blank';
                        link.rel = 'noopener noreferrer';
                        link.textContent = text;

                        citeDiv.appendChild(link);
                    } else {
                        citeDiv.appendChild(document.createTextNode(text));
                    }
                } catch {
                    citeDiv.appendChild(document.createTextNode(text));
                }
            } else {
                citeDiv.appendChild(document.createTextNode(text));
            }

            citeDiv.appendChild(document.createElement('br'));
        });

        ui.bubble.appendChild(citeDiv);
    }

    renderTrace(data.trace || []);

    sourceUsed.textContent = data.source_used || 'Completed';

    chat.scrollTop = chat.scrollHeight;
}

// Render LangGraph trace
function renderTrace(items = []) {
    trace.innerHTML = items.length
        ? items.map(x =>
            `<div class="trace-item">${escapeHtml(String(x))}</div>`
          ).join('')
        : '<div class="empty">No trace.</div>';
}

// Handle individual SSE events
function handleStreamEvent(event, ui, answer, metadata) {
    if (!event || !event.type) return;

    switch (event.type) {
        case 'status':
            updateStatus(event.message || 'Processing...', ui);
            break;

        case 'trace':
            metadata.trace.push(event.message);
            renderTrace(metadata.trace);
            break;

        case 'token':
            appendToken(event.content || '', ui, answer);
            break;

        case 'final':
            metadata.finalData = event;
            break;

        case 'error':
            throw new Error(event.message || 'Streaming failed');
    }
}

// Send question to Agent
async function askAgent(q) {
    addMessage('user', q);

    question.value = '';

    const btn = form.querySelector('button');
    btn.disabled = true;

    // Show assistant message immediately
    const ui = createStreamingMessage();

    updateStatus('Understanding your question...', ui);

    renderTrace(['Starting LangGraph workflow...']);
    sourceUsed.textContent = 'Running';

    const answer = { value: '' };

    const metadata = {
        trace: [],
        finalData: null
    };

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: q
            })
        });

        if (!res.ok) {
            let errorMessage = 'Request failed';

            try {
                const errorData = await res.json();
                errorMessage = errorData.detail || errorMessage;
            } catch {
                // Keep default error message
            }

            throw new Error(errorMessage);
        }

        // Check whether backend supports SSE streaming
        const contentType = res.headers.get('content-type') || '';

        if (contentType.includes('text/event-stream')) {
            const reader = res.body.getReader();
            const decoder = new TextDecoder();

            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();

                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                // Support LF and CRLF line endings
                buffer = buffer.replace(/\r\n/g, '\n');

                const events = buffer.split('\n\n');
                buffer = events.pop() || '';

                for (const eventBlock of events) {
                    const dataLines = eventBlock
                        .split('\n')
                        .filter(line => line.startsWith('data:'))
                        .map(line => line.slice(5).trim());

                    if (!dataLines.length) continue;

                    const rawData = dataLines.join('\n');

                    if (rawData === '[DONE]') continue;

                    const event = JSON.parse(rawData);

                    handleStreamEvent(
                        event,
                        ui,
                        answer,
                        metadata
                    );
                }
            }

            // Flush remaining decoder data
            buffer += decoder.decode();
            buffer = buffer.replace(/\r\n/g, '\n');

            // Process remaining event
            if (buffer.trim()) {
                const dataLines = buffer
                    .split('\n')
                    .filter(line => line.startsWith('data:'))
                    .map(line => line.slice(5).trim());

                if (dataLines.length) {
                    const rawData = dataLines.join('\n');

                    if (rawData !== '[DONE]') {
                        handleStreamEvent(
                            JSON.parse(rawData),
                            ui,
                            answer,
                            metadata
                        );
                    }
                }
            }

            const finalData = metadata.finalData || {};

            // Fallback if backend did not send final answer
            if (!finalData.answer && answer.value) {
                finalData.answer = answer.value;
            }

            finishStreaming(finalData, ui);

        } else {
            // Backward compatibility with JSON API
            const data = await res.json();

            ui.content.innerHTML = formatText(data.answer || '');

            finishStreaming(data, ui);
        }

    } catch (e) {
        if (ui.status && ui.status.isConnected) {
            ui.status.classList.add('thinking-error');
            ui.status.textContent = `Error: ${e.message}`;
        }

        renderTrace(['Request failed']);
        sourceUsed.textContent = 'Error';

    } finally {
        btn.disabled = false;
        question.focus();
    }
}

// Chat form submission
form.addEventListener('submit', e => {
    e.preventDefault();

    const q = question.value.trim();

    if (q) {
        askAgent(q);
    }
});

// Example questions
document.querySelectorAll('.example').forEach(b => {
    b.addEventListener('click', () => {
        if (!form.querySelector('button').disabled) {
            askAgent(b.textContent.trim());
        }
    });
});


// ======================================
// DOCUMENT UPLOAD MODAL
// ======================================

const modal = document.getElementById('uploadModal');
const openUploadBtn = document.getElementById('openUpload');
const closeUploadBtn = document.getElementById('closeUpload');
const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');
const adminKeyInput = document.getElementById('adminKey');
const uploadStatus = document.getElementById('uploadStatus');

let uploadInProgress = false;
let currentUploadMessage = '';

// Open modal
openUploadBtn.addEventListener('click', () => {
    modal.classList.remove('hidden');

    if (currentUploadMessage) {
        uploadStatus.textContent = currentUploadMessage;
    }
});

// Close modal without cancelling upload
function closeUploadModal() {
    modal.classList.add('hidden');
}

closeUploadBtn.addEventListener('click', closeUploadModal);

// Close modal using Escape key
document.addEventListener('keydown', e => {
    if (
        e.key === 'Escape' &&
        !modal.classList.contains('hidden')
    ) {
        closeUploadModal();
    }
});

// Update upload status
function setUploadStatus(message, state = '') {
    currentUploadMessage = message;

    uploadStatus.textContent = message;
    uploadStatus.className = `upload-status ${state}`.trim();
}

// Upload and index company documents
uploadBtn.addEventListener('click', async () => {

    // Prevent duplicate submissions
    if (uploadInProgress) return;

    const file = fileInput.files[0];
    const key = adminKeyInput.value.trim();

    if (!file) {
        setUploadStatus('Please choose a document first.', 'error');
        return;
    }

    if (!key) {
        setUploadStatus('Please enter the admin API key.', 'error');
        adminKeyInput.focus();
        return;
    }

    uploadInProgress = true;

    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Uploading & indexing...';

    setUploadStatus(
        `Uploading "${file.name}" and indexing it into the knowledge base. This may take a few minutes.`,
        'loading'
    );

    const fd = new FormData();
    fd.append('file', file);

    try {
        const r = await fetch('/api/ingest', {
            method: 'POST',
            headers: {
                'X-Admin-Key': key
            },
            body: fd
        });

        let d = {};

        try {
            d = await r.json();
        } catch {
            throw new Error('The server returned an invalid response.');
        }

        if (!r.ok) {
            throw new Error(d.detail || 'Document upload failed.');
        }

        const successMessage =
            `Successfully indexed ${d.file || file.name}` +
            (d.chunks !== undefined ? ` (${d.chunks} chunks).` : '.');

        setUploadStatus(successMessage, 'success');

        // Notify user in chat if modal was closed
        if (modal.classList.contains('hidden')) {
            addMessage(
                'assistant',
                `Your document **${d.file || file.name}** has been uploaded and indexed successfully. You can now ask questions about it.`
            );
        }

        fileInput.value = '';

    } catch (e) {
        const errorMessage = `Upload failed: ${e.message}`;

        setUploadStatus(errorMessage, 'error');

        // Notify user in chat if modal was closed
        if (modal.classList.contains('hidden')) {
            addMessage('assistant', errorMessage);
        }

    } finally {
        uploadInProgress = false;

        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Upload Document';
    }
});
