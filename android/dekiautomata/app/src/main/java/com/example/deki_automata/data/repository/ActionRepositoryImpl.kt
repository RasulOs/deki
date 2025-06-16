package com.example.deki_automata.data.repository

import android.util.Log
import com.example.deki_automata.data.model.ActionRequest
import com.example.deki_automata.data.model.ActionResponse
import com.example.deki_automata.data.network.RetrofitInstance
import com.example.deki_automata.domain.repository.ActionRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class ActionRepositoryImpl : ActionRepository {

    private val apiService = RetrofitInstance.api
    // TODO replace with your backend token or create a token management system
    private val apiToken = "Bearer your_token"

    override suspend fun sendActionRequest(prompt: String, screenshotBase64: String, history: List<String>): Result<ActionResponse> {
        return withContext(Dispatchers.IO) {
            try {
//                if (BuildConfig.API_TOKEN.isEmpty()) {
//                    return@withContext Result.failure(IllegalStateException("API Token is not configured in local.properties"))
//                }

                val request = ActionRequest(image = screenshotBase64, prompt = prompt, history = history)
                Log.d("ActionRepository", "Sending request: Prompt='${prompt}', History(size)=${history.size} , Image(size)=${screenshotBase64.length}, Token='${apiToken.take(15)}...'")
                val response = apiService.sendAction(apiToken, request)
                Log.d("ActionRepository", "Received response: ${response.response}, new History(size)=${response.history.size}")
                Result.success(response)
            } catch (e: Exception) {
                Log.e("ActionRepository", "Backend request failed", e)
                Result.failure(RuntimeException("Backend request failed: ${e.message}", e))
            }
        }
    }
}
