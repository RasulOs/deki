package com.example.deki_automata.data.model
import kotlinx.serialization.Serializable

@Serializable
data class DetailedAction(
    val reason: String,
    val action: String,
)
