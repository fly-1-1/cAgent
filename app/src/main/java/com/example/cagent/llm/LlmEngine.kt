package com.example.cagent.llm

/**
 * JNI bridge to the native llama.cpp-backed inference engine.
 */
class LlmEngine {

    /**
     * Per-token callback invoked from native during [complete].
     *
     * Called on the engine thread (the same single-thread executor that drives
     * `init` / `complete` / `free`). Implementations should be cheap and must
     * not block on the UI thread, otherwise generation throughput will suffer.
     *
     * @return `true` to keep generating, `false` to ask native to stop early.
     */
    fun interface TokenCallback {
        fun onToken(piece: String): Boolean
    }

    external fun init(modelPath: String, contextLen: Int): Long
    external fun free(handle: Long)

    /**
     * Run a chat-completion turn. The full reply is also returned at the end
     * (concatenation of all callback pieces) for callers that don't care about
     * streaming.
     */
    external fun complete(handle: Long, prompt: String, callback: TokenCallback?): String

    companion object {
        init {
            System.loadLibrary("offlinechat")
        }
    }
}
