package com.example.deki_automata.data.model

import kotlinx.serialization.Serializable

// {"response": "command"}
@Serializable
data class ActionResponse(
    val response: String,
    // History of previous commands
    val history: List<String>,
)