package com.example.deki_automata.domain.usecase

import com.example.deki_automata.data.model.ActionResponse

interface CommandGenerator {
    suspend fun getNextAction(
        prompt: String,
        screenshotBase64: String,
        history: List<String>,
    ): Result<ActionResponse>
}
