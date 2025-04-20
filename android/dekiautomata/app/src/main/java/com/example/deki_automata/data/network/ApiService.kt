package com.example.deki_automata.data.network

import com.example.deki_automata.data.model.ActionRequest
import com.example.deki_automata.data.model.ActionResponse
import retrofit2.http.Body
import retrofit2.http.Header
import retrofit2.http.POST

interface ApiService {

    @POST("action")
    suspend fun sendAction(
        @Header("Authorization") token: String,
        @Body request: ActionRequest
    ): ActionResponse
}