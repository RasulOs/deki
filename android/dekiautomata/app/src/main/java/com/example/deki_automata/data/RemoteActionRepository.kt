package com.example.deki_automata.data

import android.util.Log
import com.example.deki_automata.data.model.ActionRequest
import com.example.deki_automata.data.network.RetrofitInstance
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

interface ActionRepository {
    suspend fun sendActionRequest(prompt: String, screenshotBase64: String): Result<String>
}

class ActionRepositoryImpl : ActionRepository {

    private val apiService = RetrofitInstance.api
    // TODO replace with your backend token or create a token management system
    private val apiToken = "Bearer paste_your_token_here"

    override suspend fun sendActionRequest(prompt: String, screenshotBase64: String): Result<String> {
        return withContext(Dispatchers.IO) {
            try {
//                if (BuildConfig.API_TOKEN.isEmpty()) {
//                    return@withContext Result.failure(IllegalStateException("API Token is not configured in local.properties"))
//                }

                val request = ActionRequest(image = screenshotBase64, prompt = prompt)
                Log.d("ActionRepository", "Sending request: Prompt='${prompt}', Image(size)=${screenshotBase64.length}, Token='${apiToken.take(15)}...'")
                val response = apiService.sendAction(apiToken, request)
                Log.d("ActionRepository", "Received response: ${response.response}")
                Result.success(response.response)
            } catch (e: Exception) {
                Log.e("ActionRepository", "Backend request failed", e)
                Result.failure(RuntimeException("Backend request failed: ${e.message}", e))
            }
        }
    }
}