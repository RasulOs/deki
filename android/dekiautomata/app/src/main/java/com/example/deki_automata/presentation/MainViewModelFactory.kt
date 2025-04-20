package com.example.deki_automata.presentation

import android.app.Application
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.deki_automata.data.ActionRepository
import com.example.deki_automata.data.ActionRepositoryImpl
import com.example.deki_automata.domain.ExecuteAutomationUseCase

// TODO update
class MainViewModelFactory(
    private val repository: ActionRepository,
    private val executeAutomation: ExecuteAutomationUseCase
) : ViewModelProvider.Factory {

    override fun <T : ViewModel> create(modelClass: Class<T>): T =
        when {
            modelClass.isAssignableFrom(MainViewModel::class.java) ->
                MainViewModel(repository, executeAutomation) as T
            else -> throw IllegalArgumentException(
                "MainViewModelFactory can only create MainViewModel, requested: ${modelClass.name}"
            )
        }
}
