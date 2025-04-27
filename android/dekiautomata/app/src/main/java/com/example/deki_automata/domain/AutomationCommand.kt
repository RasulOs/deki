package com.example.deki_automata.domain

sealed class AutomationCommand {
    data class Swipe(val direction: Direction, val startX: Int, val startY: Int) : AutomationCommand()
    data class Tap(val x: Int, val y: Int) : AutomationCommand()
    data class InsertText(val x: Int, val y: Int, val text: String) : AutomationCommand()
    data class OpenApp(val packageName: String) : AutomationCommand()
    data object GoHome : AutomationCommand()
    data object GoBack : AutomationCommand()
    data object RetryCapture : AutomationCommand()
    data class ShowMessage(val message: String) : AutomationCommand()
}

enum class Direction { LEFT, RIGHT, UP, DOWN }
