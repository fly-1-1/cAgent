package com.example.cagent.storage

import android.content.Context
import java.io.File

object ModelStore {
    private const val MODELS_DIR = "models"

    /** Must match the file under `src/main/assets/`. */
    const val BUNDLED_ASSET_NAME = "Qwen-7B-Chat.Q4_K_M.gguf"

    fun modelsDir(context: Context): File {
        val dir = File(context.filesDir, MODELS_DIR)
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    fun defaultModelFile(context: Context): File {
        return File(modelsDir(context), BUNDLED_ASSET_NAME)
    }
}

