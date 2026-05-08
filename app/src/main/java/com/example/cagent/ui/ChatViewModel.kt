package com.example.cagent.ui

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.cagent.llm.LlmEngine
import com.example.cagent.model.ChatMessage
import com.example.cagent.model.Role
import com.example.cagent.storage.ModelStore
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

data class ChatUiState(
    val messages: List<ChatMessage>,
    val draft: String,
    val status: String,
    val canSend: Boolean,
    val isGenerating: Boolean,
)

class ChatViewModel(app: Application) : AndroidViewModel(app) {
    private val tag = "ChatViewModel"
    private val engine = LlmEngine()

    @Volatile private var handle: Long = 0L
    @Volatile private var stopRequested: Boolean = false
    @Volatile private var activeGeneration: Job? = null

    // All native engine calls (init / complete / free) MUST run on the same
    // dedicated thread. llama.cpp internal state is not safe to touch from
    // multiple threads concurrently, and serialising on a single dispatcher
    // keeps things simple while keeping the UI thread free.
    private val executor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "llm-engine").apply { isDaemon = true }
    }
    private val llmDispatcher: CoroutineDispatcher = executor.asCoroutineDispatcher()

    private val _state = MutableStateFlow(
        ChatUiState(
            messages = listOf(
                ChatMessage(role = Role.System, text = "You are a helpful bilingual assistant (中文/English).")
            ),
            draft = "",
            status = "Model not loaded",
            canSend = false,
            isGenerating = false,
        )
    )
    val state: StateFlow<ChatUiState> = _state

    fun loadDefaultModelIfPresent() {
        if (handle != 0L) return

        val ctx = getApplication<Application>()
        val f = ModelStore.defaultModelFile(ctx)
        if (!f.exists() || f.length() <= 0) {
            _state.value = _state.value.copy(
                status = "No model. Import a .gguf in Models page.",
                canSend = false,
            )
            return
        }

        val modelBytes = f.length()
        val sizeMb = modelBytes / (1024 * 1024)
        // 7B Q4_K_M (~3.5-4.5GB) on phone GPUs can become memory-bound.
        // Use a smaller KV cache for large models to reduce prefill latency.
        val targetCtx = if (modelBytes >= LARGE_MODEL_BYTES) LARGE_MODEL_CTX else DEFAULT_CTX
        _state.value = _state.value.copy(
            status = "Loading model… ${sizeMb} MB (n_ctx=$targetCtx). This can take 30–60s on first run.",
            canSend = false,
        )

        viewModelScope.launch {
            val h = withContext(llmDispatcher) {
                try {
                    Log.i(tag, "engine.init path=${f.absolutePath} size=${f.length()}")
                    engine.init(f.absolutePath, targetCtx)
                } catch (t: Throwable) {
                    // Catches UnsatisfiedLinkError / OOM / generic Throwable.
                    // Native SIGSEGV would still kill the process; this only
                    // helps with checked / managed exceptions.
                    Log.e(tag, "engine.init threw", t)
                    0L
                }
            }
            handle = h
            _state.value = if (h == 0L) {
                _state.value.copy(
                    status = "Model load failed. Check Logcat (tag=offlinechat-native). " +
                        "If this is a 4 GB+ Q4_K_M model on a 16 GB phone the process may have been " +
                        "killed by the system; try a smaller quant (Q4_0 / 3B model).",
                    canSend = false,
                )
            } else {
                _state.value.copy(status = "Model loaded. (n_ctx=$targetCtx)", canSend = true)
            }
        }
    }

    private fun serializeMessages(messages: List<ChatMessage>): String {
        return buildString {
            for (msg in messages) {
                // Skip empty assistant messages (the streaming placeholder)
                if (msg.role == Role.Assistant && msg.text.isEmpty()) continue
                append(1.toChar()) // SOH delimiter
                append(msg.role.name.lowercase())
                append(1.toChar())
                append(msg.text)
            }
            append(1.toChar()) // trailing delimiter
        }
    }

    /**
     * Trim old conversation turns to stay within the LLM's n_ctx budget.
     *
     * We use a conservative char-based estimate (~3 chars/token) and reserve
     * room for the assistant's response (~768 tokens). Always keep the system
     * message, then keep as many recent user/assistant pairs as fit.
     */
    private fun trimMessages(messages: List<ChatMessage>): List<ChatMessage> {
        val estimatedNctx = if (handle != 0L) DEFAULT_CTX else DEFAULT_CTX
        val maxPromptTokens = estimatedNctx - 768
        val charsPerToken = 3
        val budget = maxPromptTokens * charsPerToken

        // System message is always kept.
        val systemMsgs = messages.filter { it.role == Role.System }
        val conversation = messages.filter { it.role != Role.System }

        val systemChars = systemMsgs.sumOf { it.text.length }
        val totalChars = systemChars + conversation.sumOf { it.text.length }
        if (totalChars <= budget) return messages

        // Build (user, assistant) pairs from newest to oldest, keep until budget fills.
        val nonSystem = conversation.toMutableList()
        val keptNewestFirst = mutableListOf<Pair<ChatMessage?, ChatMessage?>>()
        var usedChars = systemChars

        while (nonSystem.isNotEmpty()) {
            val assistant = nonSystem.removeLastOrNull()
            val user = nonSystem.removeLastOrNull()
            if (user == null && assistant == null) break

            val turnChars = (user?.text?.length ?: 0) + (assistant?.text?.length ?: 0)
            // Always keep at least one turn even if it exceeds budget.
            val isLastTurn = keptNewestFirst.isEmpty()
            if (!isLastTurn && usedChars + turnChars > budget) break

            keptNewestFirst.add(user to assistant)
            usedChars += turnChars
        }

        // Restore chronological order.
        val kept = mutableListOf<ChatMessage>()
        for ((user, assistant) in keptNewestFirst.asReversed()) {
            user?.let { kept.add(it) }
            assistant?.let { kept.add(it) }
        }

        return systemMsgs + kept
    }

    fun updateDraft(text: String) {
        _state.value = _state.value.copy(draft = text)
    }

    fun newChat() {
        _state.value = _state.value.copy(
            messages = listOf(ChatMessage(role = Role.System, text = "You are a helpful bilingual assistant (中文/English).")),
            status = if (handle != 0L) "Model loaded." else _state.value.status,
        )
    }

    fun send() {
        val prompt = _state.value.draft.trim()
        if (prompt.isBlank()) return
        if (handle == 0L) {
            _state.value = _state.value.copy(status = "Model not loaded yet.")
            return
        }
        if (_state.value.isGenerating) return
        stopRequested = false

        // The empty assistant message we'll stream into. Tracked by id so we
        // can update it from arbitrary positions in the list.
        val assistantMsg = ChatMessage(role = Role.Assistant, text = "")
        val assistantId = assistantMsg.id

        _state.value = _state.value.copy(
            draft = "",
            messages = _state.value.messages +
                ChatMessage(role = Role.User, text = prompt) +
                assistantMsg,
            status = "Generating…",
            canSend = false,
            isGenerating = true,
        )

        activeGeneration = viewModelScope.launch {
            val tStart = System.currentTimeMillis()

            // Status ticker so the user sees progress during a long first-time
            // prefill (when no tokens have streamed yet).
            val ticker = launch {
                while (isActive) {
                    val s = (System.currentTimeMillis() - tStart) / 1000
                    val cur = _state.value.messages.firstOrNull { it.id == assistantId }?.text.orEmpty()
                    _state.value = _state.value.copy(
                        status = if (cur.isEmpty()) "Generating… ${s}s (prefilling)" else "Generating… ${s}s"
                    )
                    delay(500)
                }
            }

            // Streaming callback — invoked on the engine thread, NOT the UI thread.
            // Mutating StateFlow.value from any thread is safe.
            val callback = LlmEngine.TokenCallback { piece ->
                if (stopRequested) return@TokenCallback false
                _state.update { current ->
                    val updated = current.messages.map { m ->
                        if (m.id == assistantId) m.copy(text = m.text + piece) else m
                    }
                    current.copy(messages = updated)
                }
                true
            }

            val raw = try {
                withContext(llmDispatcher) {
                    val trimmed = trimMessages(_state.value.messages)
                    val history = serializeMessages(trimmed)
                    try {
                        engine.complete(handle, history, callback)
                    } catch (t: Throwable) {
                        Log.e(tag, "engine.complete threw", t)
                        "[Error] ${t::class.java.simpleName}: ${t.message}"
                    }
                }
            } finally {
                ticker.cancel()
            }

            val elapsedMs = System.currentTimeMillis() - tStart
            val streamed = _state.value.messages.firstOrNull { it.id == assistantId }?.text.orEmpty()

            // If streaming gave us nothing AND the final return is also blank,
            // surface a clear diagnostic placeholder.
            if (streamed.isEmpty() && raw.isNullOrBlank()) {
                _state.update { current ->
                    val updated = current.messages.map { m ->
                        if (m.id == assistantId) m.copy(
                            text = "[empty reply — model returned 0 tokens. Check Logcat tag=offlinechat-native]"
                        ) else m
                    }
                    current.copy(messages = updated)
                }
            } else if (streamed.isEmpty() && !raw.isNullOrBlank()) {
                // Callback never fired (e.g. JNI couldn't resolve onToken) but
                // the non-streaming return path filled `raw`. Use that.
                _state.update { current ->
                    val updated = current.messages.map { m ->
                        if (m.id == assistantId) m.copy(text = raw) else m
                    }
                    current.copy(messages = updated)
                }
            }

            Log.i(tag, "complete done elapsedMs=$elapsedMs streamedLen=${streamed.length} rawLen=${raw?.length ?: 0}")

            val wasStopped = stopRequested
            activeGeneration = null
            _state.value = _state.value.copy(
                status = if (wasStopped) "Stopped (${elapsedMs / 1000}s)" else "Ready (${elapsedMs / 1000}s)",
                canSend = true,
                isGenerating = false,
            )
        }
    }

    fun stop() {
        if (!_state.value.isGenerating) return
        stopRequested = true
        _state.value = _state.value.copy(status = "Stopping…")
        val h = handle
        if (h != 0L) {
            viewModelScope.launch {
                withContext(llmDispatcher) {
                    try {
                        engine.abort(h)
                    } catch (t: Throwable) {
                        Log.w(tag, "engine.abort threw", t)
                    }
                }
            }
        }
    }

    override fun onCleared() {
        val h = handle
        handle = 0L
        // Free on the engine thread, then shut the executor down.
        executor.execute {
            try {
                if (h != 0L) engine.free(h)
            } catch (t: Throwable) {
                Log.w(tag, "engine.free threw", t)
            }
        }
        executor.shutdown()
        super.onCleared()
    }

    companion object {
        // 2048 is a safe ceiling for 7B Q4_K_M on most 12-16 GB phones.
        // Bump to 4096 once you've confirmed the model + device are stable.
        private const val DEFAULT_CTX = 2048
        private const val LARGE_MODEL_CTX = 1024
        private const val LARGE_MODEL_BYTES = 3_000_000_000L
    }
}
