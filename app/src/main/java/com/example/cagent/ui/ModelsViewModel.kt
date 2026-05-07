package com.example.cagent.ui

import android.app.Application
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.cagent.storage.BundledModelImporter
import com.example.cagent.storage.ModelStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream

data class ModelsUiState(
    val modelPath: String?,
    val status: String,
    val progressText: String?,
    val progressFraction: Float?,
    val canOpenChat: Boolean,
    val isBusy: Boolean,
)

class ModelsViewModel(app: Application) : AndroidViewModel(app) {
    private val tag = "ModelsViewModel"

    private val _state = MutableStateFlow(
        ModelsUiState(
            modelPath = null,
            status = "No model. Import a .gguf to start.",
            progressText = null,
            progressFraction = null,
            canOpenChat = false,
            isBusy = false,
        )
    )
    val state: StateFlow<ModelsUiState> = _state

    fun refresh() {
        viewModelScope.launch(Dispatchers.IO) {
            val ctx = getApplication<Application>()
            val dest = ModelStore.defaultModelFile(ctx)
            val asset = ModelStore.BUNDLED_ASSET_NAME
            val assetExists = BundledModelImporter.assetExists(ctx, asset)
            val expectedLen = if (assetExists) BundledModelImporter.expectedAssetLength(ctx, asset) else null

            // Reuse already-imported / extracted file if intact.
            if (BundledModelImporter.isDestComplete(dest, expectedLen)) {
                setModelReadyState(dest)
                return@launch
            }

            // Optional: small bundled asset auto-extract path. Large GGUFs cannot be
            // packaged inside an APK due to the Zip32 4 GiB per-entry limit, so in
            // practice users always import via the file picker.
            if (assetExists) {
                _state.value = _state.value.copy(
                    isBusy = true,
                    status = "Extracting bundled model (first launch)…",
                    progressText = null,
                    progressFraction = null,
                    canOpenChat = false,
                )
                try {
                    BundledModelImporter.copyAssetToFile(ctx, asset, dest) { copied, total ->
                        emitCopyProgress(copied, total, copyVerb = "Copied")
                    }
                    setModelReadyState(dest)
                } catch (t: Throwable) {
                    Log.e(tag, "Bundled model extract failed", t)
                    _state.value = _state.value.copy(
                        status = "Asset extract failed: ${t::class.java.simpleName}: ${t.message}",
                        canOpenChat = false,
                        progressText = null,
                        progressFraction = null,
                    )
                } finally {
                    _state.value = _state.value.copy(isBusy = false)
                }
                return@launch
            }

            val exists = dest.exists() && dest.length() > 0
            _state.value = _state.value.copy(
                modelPath = if (exists) dest.absolutePath else null,
                status = if (exists) {
                    "Model file present: ${dest.name} (${dest.length() / (1024 * 1024)} MB)"
                } else {
                    "No model. Import a .gguf to start."
                },
                progressText = null,
                progressFraction = null,
                canOpenChat = exists,
            )
        }
    }

    fun importModelFromUri(uri: Uri) {
        val app = getApplication<Application>()
        val dest = ModelStore.defaultModelFile(app)

        viewModelScope.launch(Dispatchers.IO) {
            _state.value = _state.value.copy(
                isBusy = true,
                status = "Importing model…",
                progressText = null,
                progressFraction = null,
                canOpenChat = false,
            )

            try {
                val resolver = app.contentResolver
                val totalSize = runCatching {
                    resolver.openFileDescriptor(uri, "r")?.use { it.statSize }
                        .takeIf { it != null && it > 0 }
                }.getOrNull()

                dest.parentFile?.mkdirs()
                if (dest.exists()) dest.delete()

                resolver.openInputStream(uri)?.use { input ->
                    FileOutputStream(dest).use { output ->
                        val buf = ByteArray(1024 * 1024)
                        var copied = 0L
                        var lastReported = 0L
                        // Throttle UI updates to ~once per 16 MB so we don't
                        // hammer Compose with 19k state mutations on a 4.9 GB
                        // file (which itself helps avoid main-thread ANR).
                        val reportEvery = 16L * 1024 * 1024
                        while (true) {
                            val n = input.read(buf)
                            if (n <= 0) break
                            output.write(buf, 0, n)
                            copied += n
                            if (copied - lastReported >= reportEvery) {
                                emitCopyProgress(copied, totalSize, copyVerb = "Copied")
                                lastReported = copied
                            }
                        }
                        output.flush()
                        output.fd.sync()
                        emitCopyProgress(copied, totalSize, copyVerb = "Copied")
                    }
                } ?: throw IllegalStateException("Failed to open input stream for uri=$uri")

                if (totalSize != null && dest.length() != totalSize) {
                    val actual = dest.length()
                    dest.delete()
                    throw IllegalStateException(
                        "Imported file is truncated: expected=$totalSize actual=$actual. Re-import."
                    )
                }
                if (dest.length() < 1L * 1024 * 1024) {
                    dest.delete()
                    throw IllegalStateException("Imported file too small to be a GGUF model (${dest.length()} B).")
                }

                Log.i(tag, "import done dest=${dest.absolutePath} size=${dest.length()}")
                setModelReadyState(dest)
            } catch (t: Throwable) {
                Log.e(tag, "Import failed uri=$uri", t)
                _state.value = _state.value.copy(
                    status = "Import failed: ${t::class.java.simpleName}: ${t.message}",
                    canOpenChat = false,
                    progressText = null,
                    progressFraction = null,
                )
            } finally {
                _state.value = _state.value.copy(isBusy = false)
            }
        }
    }

    fun deleteImportedModel() {
        val ctx = getApplication<Application>()
        val dest = ModelStore.defaultModelFile(ctx)
        viewModelScope.launch(Dispatchers.IO) {
            try {
                if (dest.exists()) dest.delete()
            } catch (t: Throwable) {
                Log.w(tag, "delete failed", t)
            }
            _state.value = _state.value.copy(
                modelPath = null,
                status = "No model. Import a .gguf to start.",
                progressText = null,
                progressFraction = null,
                canOpenChat = false,
            )
        }
    }

    private fun setModelReadyState(dest: File) {
        val exists = dest.exists() && dest.length() > 0
        _state.value = _state.value.copy(
            modelPath = if (exists) dest.absolutePath else null,
            status = if (exists) "Model ready: ${dest.name} (${dest.length() / (1024 * 1024)} MB)" else "No model",
            progressText = null,
            progressFraction = null,
            canOpenChat = exists,
        )
    }

    private fun emitCopyProgress(copied: Long, total: Long?, copyVerb: String) {
        val copiedMb = copied / (1024.0 * 1024.0)
        val (fraction, text) = if (total != null && total > 0) {
            val pct = (copied * 100.0 / total).toInt().coerceIn(0, 100)
            val totalMb = total / (1024.0 * 1024.0)
            val frac = (copied.toDouble() / total.toDouble()).toFloat().coerceIn(0f, 1f)
            frac to "${pct}% (${String.format("%.1f", copiedMb)} / ${String.format("%.1f", totalMb)} MB)"
        } else {
            null to "$copyVerb ${String.format("%.1f", copiedMb)} MB"
        }
        _state.value = _state.value.copy(progressText = text, progressFraction = fraction)
    }
}
