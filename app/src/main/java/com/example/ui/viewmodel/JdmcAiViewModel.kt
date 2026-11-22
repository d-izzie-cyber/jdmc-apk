package com.example.ui.viewmodel

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Base64
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.BuildConfig
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.UUID
import java.util.concurrent.TimeUnit

private const val TAG = "JdmcAiViewModel"
private const val GEMINI_MODEL_NAME = "gemini-2.5-flash"
private const val AI_TIMEOUT_MS = 60_000L
private const val MAX_IMAGE_DIM_PX = 1024

private const val SYSTEM_PROMPT =
    "You are the official JDMC Road Concierge AI for Jet Darat Motorcycle Club (JDMC) " +
        "in Los Santos on Satu Mimpi RP (FiveM). Club motto: 'Satu Jalan, Satu Mimpi, Satu Keluarga.' " +
        "Key riders include Gashima (President), Melody (Vice President), Crystal Oda Megumi " +
        "(The Mother of JDMC), Joko Sancez (Charter Veteran), and Duo Tachibana (Tashima & Gonzales), " +
        "riding classic RX-King 2-Tak motorcycles. " +
        "Help riders draft announcements, plan Los Santos / Blaine County convoy routes, " +
        "write member bios, suggest Media Vault photo captions, and give RX-King 2-Tak garage tips. " +
        "Match the user's language (Indonesian or English) naturally, concisely, and with respectful biker brotherhood energy."

sealed interface AiUiState {
    data object Idle : AiUiState
    data object Loading : AiUiState
    data class Streaming(val text: String) : AiUiState
    data class Success(val text: String) : AiUiState
    data class Error(val message: String) : AiUiState
}

data class AiChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val isUser: Boolean,
    val text: String,
    val imageUri: String? = null
)

class JdmcAiViewModel(application: Application) : AndroidViewModel(application) {

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(60, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    private val conversationHistory = mutableListOf<JSONObject>()

    private val _uiState = MutableStateFlow<AiUiState>(AiUiState.Idle)
    val uiState: StateFlow<AiUiState> = _uiState.asStateFlow()

    private val _messages = MutableStateFlow<List<AiChatMessage>>(emptyList())
    val messages: StateFlow<List<AiChatMessage>> = _messages.asStateFlow()

    private val _selectedImageUri = MutableStateFlow<Uri?>(null)
    val selectedImageUri: StateFlow<Uri?> = _selectedImageUri.asStateFlow()

    fun setSelectedImage(uri: Uri?) {
        _selectedImageUri.value = uri
    }

    fun clearConversation() {
        conversationHistory.clear()
        _messages.value = emptyList()
        _selectedImageUri.value = null
        _uiState.value = AiUiState.Idle
    }

    fun sendPrompt(rawPrompt: String) {
        val attachedUri = _selectedImageUri.value
        val prompt = rawPrompt.trim().ifBlank {
            if (attachedUri != null) {
                "Analyze this photo and write a cool JDMC Media Vault title and caption."
            } else {
                ""
            }
        }
        if (prompt.isBlank()) return

        val apiKey = getGeminiApiKey()
        if (apiKey.isBlank()) {
            _uiState.value = AiUiState.Error(
                "GEMINI_API_KEY belum terpasang di APK. Pastikan secret GEMINI_API_KEY sudah diisi di GitHub lalu build ulang."
            )
            return
        }

        val userMessage = AiChatMessage(
            isUser = true,
            text = prompt,
            imageUri = attachedUri?.toString()
        )
        _messages.update { it + userMessage }
        _selectedImageUri.value = null
        _uiState.value = AiUiState.Loading

        viewModelScope.launch {
            try {
                val bitmap = if (attachedUri != null) {
                    decodeScaledBitmap(attachedUri)
                } else {
                    null
                }

                val userParts = JSONArray().apply {
                    if (bitmap != null) {
                        val base64Image = encodeBitmapToBase64(bitmap)
                        put(
                            JSONObject().apply {
                                put(
                                    "inlineData",
                                    JSONObject().apply {
                                        put("mimeType", "image/jpeg")
                                        put("data", base64Image)
                                    }
                                )
                            }
                        )
                    }
                    put(JSONObject().apply { put("text", prompt) })
                }

                val userTurn = JSONObject().apply {
                    put("role", "user")
                    put("parts", userParts)
                }

                val contentsArray = JSONArray().apply {
                    conversationHistory.forEach { put(it) }
                    put(userTurn)
                }

                val requestPayload = JSONObject().apply {
                    put(
                        "systemInstruction",
                        JSONObject().apply {
                            put(
                                "parts",
                                JSONArray().put(
                                    JSONObject().apply { put("text", SYSTEM_PROMPT) }
                                )
                            )
                        }
                    )
                    put("contents", contentsArray)
                }

                val replyText = withTimeout(AI_TIMEOUT_MS) {
                    callGeminiRestApi(apiKey, requestPayload.toString())
                }

                if (replyText.isNotEmpty()) {
                    conversationHistory.add(userTurn)
                    conversationHistory.add(
                        JSONObject().apply {
                            put("role", "model")
                            put(
                                "parts",
                                JSONArray().put(
                                    JSONObject().apply { put("text", replyText) }
                                )
                            )
                        }
                    )
                    val assistantMessage = AiChatMessage(
                        isUser = false,
                        text = replyText
                    )
                    _messages.update { it + assistantMessage }
                    _uiState.value = AiUiState.Success(replyText)
                } else {
                    _uiState.value = AiUiState.Error("Gemini returned an empty response. Please try again.")
                }
            } catch (e: TimeoutCancellationException) {
                Log.e(TAG, "Gemini request timed out after ${AI_TIMEOUT_MS}ms", e)
                _uiState.value = AiUiState.Error(
                    "Request timed out after 60 seconds. Check your connection and try again."
                )
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "Gemini API generation failed", e)
                _uiState.value = AiUiState.Error(
                    e.localizedMessage ?: "Gemini AI request failed. Please try again."
                )
            }
        }
    }

    private suspend fun callGeminiRestApi(apiKey: String, jsonBody: String): String =
        withContext(Dispatchers.IO) {
            val url =
                "https://generativelanguage.googleapis.com/v1beta/models/$GEMINI_MODEL_NAME:generateContent?key=$apiKey"
            val mediaType = "application/json; charset=utf-8".toMediaType()
            val request = Request.Builder()
                .url(url)
                .post(jsonBody.toRequestBody(mediaType))
                .build()

            httpClient.newCall(request).execute().use { response ->
                val bodyString = response.body?.string().orEmpty()
                if (!response.isSuccessful) {
                    val apiErrorMessage = runCatching {
                        JSONObject(bodyString)
                            .optJSONObject("error")
                            ?.optString("message")
                            .orEmpty()
                    }.getOrDefault("")
                    val detail = apiErrorMessage.ifBlank { "HTTP ${response.code}" }
                    throw IllegalStateException("Gemini API Error: $detail")
                }

                val root = JSONObject(bodyString)
                val candidates = root.optJSONArray("candidates")
                val firstCandidate = candidates?.optJSONObject(0)
                val parts = firstCandidate
                    ?.optJSONObject("content")
                    ?.optJSONArray("parts")

                val sb = StringBuilder()
                if (parts != null) {
                    for (i in 0 until parts.length()) {
                        val partText = parts.optJSONObject(i)?.optString("text").orEmpty()
                        if (partText.isNotEmpty()) {
                            sb.append(partText)
                        }
                    }
                }
                sb.toString().trim()
            }
        }

    private fun getGeminiApiKey(): String {
        return runCatching {
            val field = BuildConfig::class.java.getField("GEMINI_API_KEY")
            (field.get(null) as? String).orEmpty().trim()
        }.getOrDefault("")
    }

    private suspend fun decodeScaledBitmap(uri: Uri): Bitmap? = withContext(Dispatchers.IO) {
        runCatching {
            val resolver = getApplication<Application>().contentResolver
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            resolver.openInputStream(uri)?.use { stream ->
                BitmapFactory.decodeStream(stream, null, bounds)
            }
            val sampleSize = calculateInSampleSize(
                bounds.outWidth,
                bounds.outHeight,
                MAX_IMAGE_DIM_PX,
                MAX_IMAGE_DIM_PX
            )
            val decodeOptions = BitmapFactory.Options().apply { inSampleSize = sampleSize }
            resolver.openInputStream(uri)?.use { stream ->
                BitmapFactory.decodeStream(stream, null, decodeOptions)
            }
        }.onFailure { err ->
            Log.e(TAG, "Failed to decode attached image for Gemini multimodal request", err)
        }.getOrNull()
    }

    private fun encodeBitmapToBase64(bitmap: Bitmap): String {
        val outputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 85, outputStream)
        return Base64.encodeToString(outputStream.toByteArray(), Base64.NO_WRAP)
    }

    private fun calculateInSampleSize(width: Int, height: Int, reqWidth: Int, reqHeight: Int): Int {
        var inSampleSize = 1
        if (height > reqHeight || width > reqWidth) {
            val halfHeight = height / 2
            val halfWidth = width / 2
            while (halfHeight / inSampleSize >= reqHeight && halfWidth / inSampleSize >= reqWidth) {
                inSampleSize *= 2
            }
        }
        return inSampleSize
    }
}
