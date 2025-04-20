package com.example.deki_automata.service

import com.example.deki_automata.domain.Direction

interface DeviceController {
    fun captureScreenBase64(): String?
    fun swipe(direction: Direction, startX: Int, startY: Int): Boolean
    fun tap(x: Int, y: Int): Boolean
    fun insertText(x: Int, y: Int, text: String): Boolean
    fun openApp(packageName: String): Boolean
    fun pressHome(): Boolean
    fun goBack(): Boolean
    fun returnToApp(): Boolean
    fun waitForIdle()
}