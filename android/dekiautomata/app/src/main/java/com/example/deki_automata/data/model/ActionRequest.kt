package com.example.deki_automata.data.model

import kotlinx.serialization.Serializable

@Serializable
data class ActionRequest(
    val image: String, // Base64 encoded image
    val prompt: String,
    val history: List<String> = emptyList(), // List of previous commands or messages
)
