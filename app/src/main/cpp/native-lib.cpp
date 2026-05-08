#include <jni.h>
#include <android/log.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <ctime>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "llama.h"

namespace {
    constexpr const char * TAG = "offlinechat-native";

    struct Engine {
        llama_model   * model = nullptr;
        llama_context * ctx   = nullptr;
        llama_sampler * smpl  = nullptr;
        const llama_vocab * vocab = nullptr;
        std::atomic<bool> abort{false};

        ~Engine() {
            if (smpl)  llama_sampler_free(smpl);
            if (ctx)   llama_free(ctx);
            if (model) llama_model_free(model);
        }
    };

    static std::string token_to_piece(const llama_vocab * vocab, llama_token tok) {
        std::string out;
        out.resize(256);
        const int n = llama_token_to_piece(vocab, tok, out.data(), (int) out.size(), 0, true);
        if (n <= 0) return "";
        out.resize(n);
        return out;
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_cagent_llm_LlmEngine_init(JNIEnv *env, jobject /*thiz*/, jstring modelPath, jint contextLen) {
    const char * cpath = env->GetStringUTFChars(modelPath, nullptr);
    std::string pathCopy = cpath ? cpath : "";
    env->ReleaseStringUTFChars(modelPath, cpath);

    if (pathCopy.empty()) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "init: empty model path");
        return 0;
    }

    __android_log_print(ANDROID_LOG_INFO, TAG, "init: path=%s ctx=%d", pathCopy.c_str(), contextLen);

    try {
        llama_backend_init();

        auto mparams = llama_model_default_params();
        mparams.use_mmap = true;
        mparams.use_mlock = false;
#if defined(GGML_OPENCL)
        mparams.n_gpu_layers = 99;
        __android_log_print(ANDROID_LOG_INFO, TAG, "GPU mode: n_gpu_layers=99 (OpenCL/Adreno)");
#else
        mparams.n_gpu_layers = 0;
        __android_log_print(ANDROID_LOG_INFO, TAG, "CPU-only mode");
#endif

        llama_model * model = llama_model_load_from_file(pathCopy.c_str(), mparams);
        if (!model) {
            __android_log_print(ANDROID_LOG_ERROR, TAG, "Failed to load model (unsupported / corrupt GGUF?)");
            return 0;
        }

        auto cparams = llama_context_default_params();
        // Clamp n_ctx so we don't blow the device's RAM with a giant KV cache.
        const uint32_t requestedCtx = (uint32_t) std::max(256, contextLen);
        cparams.n_ctx = std::min<uint32_t>(requestedCtx, 4096);
        // OpenCL path: prefer higher batch + offload attention for speed.
#if defined(GGML_OPENCL)
        cparams.n_batch = 256;
        cparams.n_ubatch = 256;
        cparams.offload_kqv = true;
#else
        cparams.n_batch = 128;
        cparams.n_ubatch = 128;
        cparams.offload_kqv = false;
#endif
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            __android_log_print(ANDROID_LOG_ERROR, TAG, "Failed to create context (out of memory?)");
            llama_model_free(model);
            return 0;
        }

        const long online = sysconf(_SC_NPROCESSORS_ONLN);
        // If GPU offload partially falls back to CPU kernels, 2 threads is often
        // too conservative for 7B; use more cores without saturating all cores.
        const int threads = (int) std::clamp<long>(online > 0 ? online / 2 : 4, 3, 6);
        llama_set_n_threads(ctx, threads, threads);
        __android_log_print(ANDROID_LOG_INFO, TAG, "threads=%d n_batch=%u n_ubatch=%u",
                            threads, cparams.n_batch, cparams.n_ubatch);

        // Lower the calling thread's scheduling priority (nice +5) so that
        // when llama.cpp / ggml saturates the CPU during prefill or decode,
        // the kernel still preempts us in favour of the UI thread.
        setpriority(PRIO_PROCESS, 0, 5);

        auto sparams = llama_sampler_chain_default_params();
        llama_sampler * smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        auto * e = new (std::nothrow) Engine();
        if (!e) {
            llama_sampler_free(smpl);
            llama_free(ctx);
            llama_model_free(model);
            return 0;
        }

        e->model = model;
        e->ctx = ctx;
        e->smpl = smpl;
        e->vocab = llama_model_get_vocab(model);

        __android_log_print(ANDROID_LOG_INFO, TAG,
                            "Model loaded: n_ctx=%u threads=%d", llama_n_ctx(ctx), threads);

        // Sanity check: does this model's tokenizer actually recognise the
        // ChatML control markers as single special tokens? Old GGUFs (e.g.
        // Xorbits/Qwen-7B-Chat-GGUF from 2023) often don't, in which case the
        // model receives "<", "|", "i", "m", "_", ... as raw characters and
        // either replies with garbage or just hangs in prefill.
        {
            const char * marker = "<|im_start|>";
            llama_token tk[8] = {0};
            int n = llama_tokenize(e->vocab, marker, (int32_t) std::strlen(marker),
                                   tk, 8,
                                   /*add_special=*/false, /*parse_special=*/true);
            if (n == 1) {
                __android_log_print(ANDROID_LOG_INFO, TAG,
                                    "ChatML markers OK: <|im_start|> = %d", tk[0]);
            } else {
                __android_log_print(ANDROID_LOG_WARN, TAG,
                                    "WARNING: <|im_start|> tokenises to %d sub-tokens. "
                                    "This GGUF lacks proper ChatML special tokens. "
                                    "Replies will likely be empty/garbled. "
                                    "Try a newer GGUF (Qwen2.5-Instruct).", n);
            }
        }

        // Log whether the GGUF carries a chat template in its metadata.
        const char * tmpl = llama_model_chat_template(model, nullptr);
        __android_log_print(ANDROID_LOG_INFO, TAG,
                            "chat template metadata: %s",
                            tmpl ? tmpl : "(none — will fall back to manual ChatML)");

        return (jlong) (uintptr_t) e;
    } catch (const std::exception & ex) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "init: std::exception: %s", ex.what());
        return 0;
    } catch (...) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "init: unknown exception");
        return 0;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_cagent_llm_LlmEngine_free(JNIEnv *env, jobject /*thiz*/, jlong handle) {
    (void) env;
    auto * e = (Engine *) (uintptr_t) handle;
    delete e;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_cagent_llm_LlmEngine_abort(JNIEnv *env, jobject /*thiz*/, jlong handle) {
    (void) env;
    auto * e = (Engine *) (uintptr_t) handle;
    if (e) e->abort.store(true, std::memory_order_relaxed);
}

// UTF-8 -> UTF-16 (jchar) conversion, used to build a jstring without the
// pitfalls of NewStringUTF (which expects Modified UTF-8).
static std::vector<jchar> utf8_to_utf16(const std::string & s) {
    std::vector<jchar> u16;
    u16.reserve(s.size());
    const unsigned char * p = (const unsigned char *) s.data();
    const unsigned char * end = p + s.size();
    while (p < end) {
        uint32_t cp = 0;
        int extra = 0;
        unsigned char c = *p++;
        if (c < 0x80) { cp = c; extra = 0; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; extra = 1; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; extra = 2; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; extra = 3; }
        else { cp = 0xFFFD; extra = 0; }
        for (int k = 0; k < extra; k++) {
            if (p >= end || (*p & 0xC0) != 0x80) { cp = 0xFFFD; break; }
            cp = (cp << 6) | (*p++ & 0x3F);
        }
        if (cp <= 0xFFFF) {
            u16.push_back((jchar) cp);
        } else if (cp <= 0x10FFFF) {
            cp -= 0x10000;
            u16.push_back((jchar) (0xD800 | (cp >> 10)));
            u16.push_back((jchar) (0xDC00 | (cp & 0x3FF)));
        } else {
            u16.push_back((jchar) 0xFFFD);
        }
    }
    return u16;
}

// Returns the length (in bytes) of the largest prefix that does not end in the
// middle of a UTF-8 multi-byte sequence. This is used for streaming: llama.cpp
// token pieces can split multi-byte characters (e.g. Chinese) across tokens.
static size_t utf8_safe_prefix_len(const std::string & s) {
    const unsigned char * p = (const unsigned char *) s.data();
    const unsigned char * end = p + s.size();
    size_t consumed = 0;
    while (p < end) {
        unsigned char c = *p;
        int len = 1;
        if (c < 0x80) {
            len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            len = 4;
        } else {
            len = 1;
        }

        if (p + len > end) {
            // Incomplete at end: stop.
            break;
        }

        bool ok = true;
        for (int i = 1; i < len; i++) {
            if ((p[i] & 0xC0) != 0x80) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            // Invalid lead byte or continuation: consume 1 byte and continue.
            p += 1;
            consumed += 1;
            continue;
        }

        p += len;
        consumed += (size_t) len;
    }
    return consumed;
}

static jstring make_jstring(JNIEnv * env, const std::string & s) {
    auto u16 = utf8_to_utf16(s);
    return env->NewString(u16.data(), (jsize) u16.size());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_cagent_llm_LlmEngine_complete(JNIEnv *env, jobject /*thiz*/,
                                               jlong handle, jstring prompt,
                                               jobject callback /* TokenCallback or null */) {
    auto * e = (Engine *) (uintptr_t) handle;
    if (!e || !e->ctx || !e->model || !e->vocab || !e->smpl) {
        return env->NewStringUTF("Engine not initialized");
    }

    // Reset abort flag for this turn.
    e->abort.store(false, std::memory_order_relaxed);

    // Resolve TokenCallback.onToken once for this call.
    jclass    cb_cls   = nullptr;
    jmethodID cb_onTok = nullptr;
    if (callback != nullptr) {
        cb_cls = env->GetObjectClass(callback);
        if (cb_cls != nullptr) {
            cb_onTok = env->GetMethodID(cb_cls, "onToken", "(Ljava/lang/String;)Z");
        }
        if (cb_onTok == nullptr) {
            __android_log_print(ANDROID_LOG_WARN, TAG,
                                "complete: TokenCallback.onToken(Ljava/lang/String;)Z not found");
            if (env->ExceptionCheck()) env->ExceptionClear();
            callback = nullptr; // fall back to non-streaming
        }
    }

    auto invoke_cb = [&](const std::string & piece) -> bool {
        if (callback == nullptr || cb_onTok == nullptr) return true;
        // Build a jstring from a UTF-8-safe chunk.
        jstring jpiece = make_jstring(env, piece);
        jboolean keepGoing = env->CallBooleanMethod(callback, cb_onTok, jpiece);
        if (jpiece) env->DeleteLocalRef(jpiece);
        if (env->ExceptionCheck()) {
            __android_log_print(ANDROID_LOG_WARN, TAG, "complete: TokenCallback threw — stopping");
            env->ExceptionDescribe();
            env->ExceptionClear();
            return false;
        }
        return keepGoing != JNI_FALSE;
    };

    // Buffer streaming pieces to avoid splitting multi-byte UTF-8 characters
    // across callback boundaries (which would show as garbled text).
    std::string pending_utf8;
    auto stream_piece = [&](const std::string & piece) -> bool {
        if (callback == nullptr || cb_onTok == nullptr) return true;
        pending_utf8 += piece;
        const size_t safe = utf8_safe_prefix_len(pending_utf8);
        if (safe == 0) return true;
        const std::string chunk = pending_utf8.substr(0, safe);
        pending_utf8.erase(0, safe);
        return invoke_cb(chunk);
    };

    try {

    // Clear KV/memory so each call is an independent chat turn for now.
    llama_memory_clear(llama_get_memory(e->ctx), true);

    const char * cPrompt = env->GetStringUTFChars(prompt, nullptr);
    std::string userText = cPrompt ? cPrompt : "";
    env->ReleaseStringUTFChars(prompt, cPrompt);

    const char * sysText = "You are a helpful bilingual assistant (中文/English).";

    // Prefer the model's built-in chat template when the GGUF provides one.
    // Otherwise fall back to a hand-written ChatML string.
    std::string full;
    {
        const char * tmpl = llama_model_chat_template(e->model, nullptr);
        if (tmpl) {
            llama_chat_message msgs[2];
            msgs[0].role    = "system";
            msgs[0].content = sysText;
            msgs[1].role    = "user";
            msgs[1].content = userText.c_str();

            int needed = llama_chat_apply_template(tmpl, msgs, 2,
                                                    /*add_ass=*/true,
                                                    nullptr, 0);
            if (needed > 0) {
                std::vector<char> buf((size_t) needed + 1, 0);
                int written = llama_chat_apply_template(tmpl, msgs, 2,
                                                         /*add_ass=*/true,
                                                         buf.data(), (int32_t) buf.size());
                if (written > 0) {
                    full.assign(buf.data(), (size_t) written);
                }
            }
        }

        if (full.empty()) {
            full =
                std::string("<|im_start|>system\n") + sysText +
                "<|im_end|>\n"
                "<|im_start|>user\n" + userText +
                "<|im_end|>\n"
                "<|im_start|>assistant\n";
        }
    }

    // Tokenize. We supply the full ChatML template ourselves, so we pass
    //   add_special   = false  -> do NOT prepend BOS (Qwen ChatML doesn't use one
    //                              and adding one makes some old GGUFs emit EOS
    //                              on the very first sample, producing an empty reply).
    //   parse_special = true   -> let the tokenizer recognise the literal
    //                              "<|im_start|>" / "<|im_end|>" markers as their
    //                              dedicated control tokens when the model defines them.
    std::vector<llama_token> tokens(full.size() + 8);
    int32_t n_tok = llama_tokenize(e->vocab, full.c_str(), (int32_t) full.size(),
                                   tokens.data(), (int32_t) tokens.size(),
                                   /*add_special=*/false, /*parse_special=*/true);
    if (n_tok < 0) {
        tokens.resize((size_t) (-n_tok));
        n_tok = llama_tokenize(e->vocab, full.c_str(), (int32_t) full.size(),
                               tokens.data(), (int32_t) tokens.size(),
                               /*add_special=*/false, /*parse_special=*/true);
    }
    if (n_tok <= 0) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "complete: tokenize returned %d", n_tok);
        return env->NewStringUTF("Tokenization failed");
    }
    tokens.resize(n_tok);

    __android_log_print(ANDROID_LOG_INFO, TAG,
                        "complete: prompt_chars=%zu n_tok=%d first=%d last=%d",
                        full.size(), n_tok,
                        (int) tokens.front(), (int) tokens.back());

    // Reset sampler for a fresh completion
    llama_sampler_reset(e->smpl);

    // Decode prompt in chunks so we can respond to Stop during prefill.
    const int32_t n_tok_all = (int32_t) tokens.size();
    const int32_t chunk_cap = std::max<int32_t>(1, llama_n_batch(e->ctx));

    auto now_ms = []() -> int64_t {
        timespec ts{};
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (int64_t) ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
    };
    __android_log_print(ANDROID_LOG_INFO, TAG, "complete: prefill begin (n_tok=%d)…", n_tok);
    const int64_t t_prefill_start = now_ms();
    int32_t rc = 0;
    for (int32_t start = 0; start < n_tok_all; start += chunk_cap) {
        if (e->abort.load(std::memory_order_relaxed)) {
            __android_log_print(ANDROID_LOG_INFO, TAG, "complete: aborted during prefill");
            rc = 0;
            break;
        }
        const int32_t n_chunk = std::min<int32_t>(chunk_cap, n_tok_all - start);
        llama_batch batch = llama_batch_init(n_chunk, 0, 1);
        batch.n_tokens = n_chunk;
        for (int i = 0; i < n_chunk; i++) {
            batch.token[i] = tokens[start + i];
            batch.pos[i] = start + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            const bool last_token = (start + i) == (n_tok_all - 1);
            batch.logits[i] = last_token ? 1 : 0;
        }
        rc = llama_decode(e->ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) break;
    }
    const int64_t t_prefill_end = now_ms();
    if (rc != 0) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "llama_decode(prompt) rc=%d", rc);
        return env->NewStringUTF("Decode prompt failed");
    }
    __android_log_print(ANDROID_LOG_INFO, TAG,
                        "complete: prefill done in %lld ms",
                        (long long) (t_prefill_end - t_prefill_start));

    // Generate tokens
    std::string out;
    out.reserve(1024);

    int n_generated = 0;
    // Prevent answers from being cut off too early. 256 is often insufficient
    // for code-heavy responses. Keep this conservative for mobile RAM.
    const int max_tokens = 768;
    bool hit_token_limit = true;
    for (int i = 0; i < max_tokens; i++) {
        if (e->abort.load(std::memory_order_relaxed)) {
            __android_log_print(ANDROID_LOG_INFO, TAG, "complete: aborted during generation");
            hit_token_limit = false;
            break;
        }
        const llama_token next = llama_sampler_sample(e->smpl, e->ctx, -1);
        llama_sampler_accept(e->smpl, next);

        if (i == 0) {
            __android_log_print(ANDROID_LOG_INFO, TAG, "complete: first sampled token id=%d eog=%d",
                                (int) next, (int) llama_vocab_is_eog(e->vocab, next));
        }

        // Use is_eog so we cover all end-of-generation tokens the model
        // declares (Qwen exposes both <|im_end|> and <|endoftext|>).
        if (llama_vocab_is_eog(e->vocab, next)) {
            hit_token_limit = false;
            break;
        }

        std::string piece = token_to_piece(e->vocab, next);
        out += piece;
        n_generated++;

        // Stream this piece to Java; if the callback throws or returns false,
        // bail out gracefully (still keep what we already have in `out`).
        if (!stream_piece(piece)) {
            __android_log_print(ANDROID_LOG_INFO, TAG, "complete: callback requested stop at i=%d", i);
            break;
        }

        llama_batch b2 = llama_batch_init(1, 0, 1);
        b2.n_tokens = 1;
        b2.token[0] = next;
        b2.pos[0] = (llama_pos) (tokens.size() + i);
        b2.n_seq_id[0] = 1;
        b2.seq_id[0][0] = 0;
        b2.logits[0] = 1;

        const int32_t rc2 = llama_decode(e->ctx, b2);
        llama_batch_free(b2);
        if (rc2 != 0) {
            __android_log_print(ANDROID_LOG_ERROR, TAG, "llama_decode(gen) rc=%d", rc2);
            hit_token_limit = false;
            break;
        }
    }

    {
        char hex[3 * 32 + 1];
        size_t shown = std::min<size_t>(out.size(), 32);
        for (size_t i = 0; i < shown; i++) {
            snprintf(hex + i * 3, 4, "%02x ", (unsigned char) out[i]);
        }
        hex[shown * 3] = 0;
        __android_log_print(ANDROID_LOG_INFO, TAG,
                            "complete: generated tokens=%d out_bytes=%zu first_hex=%s",
                            n_generated, out.size(), hex);
    }

    if (hit_token_limit && n_generated >= max_tokens) {
        __android_log_print(ANDROID_LOG_WARN, TAG, "complete: stopped at max_tokens=%d (truncated)", max_tokens);
        // Hint in the UI without being too noisy.
        const char * suffix = "\n\n[truncated]";
        out += suffix;
        stream_piece(suffix);
    }

    // Flush any pending UTF-8 bytes (incomplete multi-byte sequence).
    if (!pending_utf8.empty()) {
        invoke_cb(pending_utf8);
        pending_utf8.clear();
    }

    // Use NewString (UTF-16) instead of NewStringUTF: NewStringUTF expects
    // Modified UTF-8 and silently misbehaves on embedded NULs or 4-byte UTF-8
    // sequences, which would surface as an empty assistant bubble.
    return make_jstring(env, out);

    } catch (const std::exception & ex) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "complete: std::exception: %s", ex.what());
        return env->NewStringUTF("[native exception during inference]");
    } catch (...) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "complete: unknown exception");
        return env->NewStringUTF("[unknown native exception]");
    }
}

