package com.example.deki_automata.domain.repository

import com.example.deki_automata.data.model.ActionResponse

interface ActionRepository {
    suspend fun sendActionRequest(prompt: String, screenshotBase64: String, history: List<String>): Result<ActionResponse>
}
