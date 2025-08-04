package com.example.deki_automata.domain.usecase

import com.example.deki_automata.data.model.ActionResponse
import com.example.deki_automata.domain.repository.ActionRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class RemoteCommandGenerator(private val repository: ActionRepository) : CommandGenerator {
    override suspend fun getNextAction(
        prompt: String,
        screenshotBase64: String,
        history: List<String>,
    ): Result<ActionResponse> = withContext(Dispatchers.IO) {
        repository.sendActionRequest(prompt, screenshotBase64, history)
    }
}
