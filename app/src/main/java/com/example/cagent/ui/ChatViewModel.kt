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
)

class ChatViewModel(app: Application) : AndroidViewModel(app) {
    private val tag = "ChatViewModel"
    private val engine = LlmEngine()

    @Volatile private var handle: Long = 0L

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

        val sizeMb = f.length() / (1024 * 1024)
        _state.value = _state.value.copy(
            status = "Loading model… ${sizeMb} MB. This can take 30–60s on first run.",
            canSend = false,
        )

        viewModelScope.launch {
            val h = withContext(llmDispatcher) {
                try {
                    Log.i(tag, "engine.init path=${f.absolutePath} size=${f.length()}")
                    engine.init(f.absolutePath, DEFAULT_CTX)
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
                _state.value.copy(status = "Model loaded. (n_ctx=$DEFAULT_CTX)", canSend = true)
            }
        }
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
        )

        viewModelScope.launch {
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
                    try {
                        engine.complete(handle, prompt, callback)
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

            _state.value = _state.value.copy(
                status = "Ready (${elapsedMs / 1000}s)",
                canSend = true,
            )
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
    }
}
