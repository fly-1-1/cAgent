package com.example.cagent.storage

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

object BundledModelImporter {
    private const val TAG = "BundledModelImporter"

    fun assetExists(context: Context, assetName: String): Boolean {
        return try {
            context.assets.open(assetName).use { true }
        } catch (_: IOException) {
            false
        }
    }

    fun expectedAssetLength(context: Context, assetName: String): Long? {
        return try {
            context.assets.openFd(assetName).use { it.length }
        } catch (e: java.io.FileNotFoundException) {
            // Asset not bundled – this is expected for large models.
            null
        } catch (e: Exception) {
            Log.w(TAG, "openFd failed for $assetName (is noCompress 'gguf' set?)", e)
            null
        }
    }

    fun isDestComplete(dest: File, expectedLen: Long?): Boolean {
        if (!dest.exists() || dest.length() <= 0L) return false
        return expectedLen == null || dest.length() == expectedLen
    }

    /**
     * Copies [assetName] from assets to [dest]. Overwrites [dest] if present and size mismatches.
     */
    suspend fun copyAssetToFile(
        context: Context,
        assetName: String,
        dest: File,
        onProgress: (copied: Long, total: Long?) -> Unit,
    ) = withContext(Dispatchers.IO) {
        dest.parentFile?.mkdirs()
        val total = expectedAssetLength(context, assetName)

        if (isDestComplete(dest, total)) {
            onProgress(dest.length(), total)
            return@withContext
        }

        if (dest.exists()) {
            dest.delete()
        }

        context.assets.open(assetName).use { input ->
            FileOutputStream(dest).use { output ->
                val buf = ByteArray(256 * 1024)
                var copied = 0L
                onProgress(0L, total)
                while (true) {
                    val n = input.read(buf)
                    if (n <= 0) break
                    output.write(buf, 0, n)
                    copied += n
                    onProgress(copied, total)
                }
            }
        }

        if (total != null && dest.length() != total) {
            throw IOException("Copy incomplete: got ${dest.length()} expected $total")
        }
    }
}
