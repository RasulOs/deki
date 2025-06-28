package com.example.deki_automata.domain.usecase

import android.content.Context
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.util.Log
import com.example.deki_automata.domain.ActionParser
import com.example.deki_automata.domain.model.AutomationCommand
import com.example.deki_automata.domain.repository.ActionRepository
import com.example.deki_automata.service.DeviceController
import kotlinx.coroutines.delay
import java.security.MessageDigest

class ExecuteAutomationUseCase(
    private val repository: ActionRepository,
    private val context: Context,
) {

    companion object {
        private const val TAG = "ExecuteAutomationUC"
        private const val MAX_STEPS = 20
        private const val MAX_CONSECUTIVE_LOADING_STATES = 2
        private const val MAX_CAPTURE_ATTEMPTS = 2
        private const val DEFAULT_DELAY = 500L
        private const val DEFAULT_DELAY_AFTER_SCREEN_CAPTURE = 1000L
    }

    private fun getScreenshotHash(screenshot: String): String {
        return MessageDigest.getInstance("MD5")
            .digest(screenshot.toByteArray())
            .joinToString("") { "%02x".format(it) }
    }

    suspend fun runAutomation(prompt: String, device: DeviceController): String {
        val allInstalledPackages by lazy { getInstalledPackages() }
        var currentHistory: List<String> = emptyList()
        var lastScreenshotHash: String? = null
        var finalMessage = ""
        var lastCommandFailed = false
        var consecutiveLoadingStates = 0

        for (step in 1..MAX_STEPS) {
            Log.d(TAG, "Step $step: Capturing screen")

            var screenshot: String? = null
            var promptForThisRequest: String

            var captureAttempts = 0
            while (captureAttempts < MAX_CAPTURE_ATTEMPTS) {
                screenshot = device.captureScreenBase64()
                if (screenshot != null)
                    break

                captureAttempts++
                Log.e(TAG, "Screen capture failed. Retrying (Attempt ${captureAttempts}/${MAX_CAPTURE_ATTEMPTS})")
                delay(DEFAULT_DELAY_AFTER_SCREEN_CAPTURE)
            }

            if (screenshot == null) {
                Log.e(TAG, "Screen capture failed after $MAX_CAPTURE_ATTEMPTS attempts. Terminating automation")
                device.returnToApp()
                delay(DEFAULT_DELAY)
                return "Automation Failed: Could not capture the screen"
            }

            val currentScreenshotHash = getScreenshotHash(screenshot)

            // TODO add installed packages to the prompt if the last command failed or screen has not changed also
            if (lastCommandFailed) {
                Log.w(TAG, "Last command failed to execute. Informing the agent")
                promptForThisRequest = buildString {
                    appendLine("The previous command could not be executed by the device controller")
                    appendLine("Please analyze the screen and provide a different command to continue the task")
                    appendLine("Original user prompt: \"$prompt\"")
                }
                lastCommandFailed = false
            } else if (lastScreenshotHash != null && lastScreenshotHash == currentScreenshotHash) {
                Log.w(TAG, "Screen has not changed. Informing the agent")
                promptForThisRequest = buildString {
                    appendLine("The previous command had no effect on the screen")
                    appendLine("Please analyze the screen again and provide a different command to continue the task")
                    appendLine("Original user prompt: \"$prompt\"")
                }
            } else {
                promptForThisRequest = if (step == 1) buildPrompt(prompt, allInstalledPackages) else prompt
            }
            lastScreenshotHash = currentScreenshotHash
            val result = repository.sendActionRequest(promptForThisRequest, screenshot, currentHistory)

            val response = result.getOrElse { e ->
                Log.e(TAG, "Action request failed", e)
                return "Automation Failed: Network request failed - ${e.message}"
            }

            currentHistory = response.history
            val parsedResult = ActionParser.parseResponse(response.response)

            parsedResult.message?.let {
                if (finalMessage.isNotEmpty()) finalMessage += "\n"
                finalMessage += it
            }

            val command = parsedResult.command

            if (command is AutomationCommand.RetryCapture) {
                consecutiveLoadingStates++
                Log.i(TAG, "Backend reports loading state. Attempt $consecutiveLoadingStates/$MAX_CONSECUTIVE_LOADING_STATES")

                if (consecutiveLoadingStates >= MAX_CONSECUTIVE_LOADING_STATES) {
                    Log.w(TAG, "Stuck in loading state loop. Forcing a different action")
                    lastCommandFailed = true
                }

                lastScreenshotHash = null
                device.waitForIdle()
                continue
            } else {
                consecutiveLoadingStates = 0
            }

            if (command is AutomationCommand.ShowMessage) {
                val finalMessageToShow = (finalMessage + "\n" + command.message).trim()
                Log.i(TAG, "Terminal ShowMessage command received. Task is complete")
                device.returnToApp()
                delay(DEFAULT_DELAY)
                return finalMessageToShow
            }

            Log.i(TAG, "Executing command: $command")
            // TODO add zoom in/out, volume up/down, turn on/off commands, long press
            val success = when (command) {
                is AutomationCommand.Swipe -> device.swipe(command.direction, command.startX, command.startY) // TODO replace to swipe from to
                is AutomationCommand.Tap -> device.tap(command.x, command.y)
                is AutomationCommand.InsertText -> device.insertText(command.x, command.y, command.text)
                is AutomationCommand.OpenApp -> device.openApp(command.packageName)
                is AutomationCommand.GoHome -> device.pressHome()
                is AutomationCommand.GoBack -> device.goBack()
                else -> true
            }

            if (!success) {
                Log.e(TAG, "Command execution failed: $command")
                lastCommandFailed = true
            }

            if (currentHistory.isEmpty()) {
                Log.w(TAG, "History was cleared by the server. Finishing")
                device.returnToApp()
                return finalMessage.ifEmpty { "Automation finished: Server ended the session" }
            }

            device.waitForIdle()
        }

        Log.w(TAG, "Max steps reached")
        device.returnToApp()
        return finalMessage.ifEmpty { "Automation Stopped: Maximum steps reached" }
    }

    private fun buildPrompt(userPrompt: String, installedPackages: List<String>): String {
        return buildString {
            appendLine("User Prompt: \"$userPrompt\"")
            appendLine()
            appendLine("Installed Non‑System Packages:")
            installedPackages.forEach(::appendLine)
        }
    }

    private fun getInstalledPackages(): List<String> {
        val pm = context.packageManager
        return try {
            pm.getInstalledApplications(PackageManager.GET_META_DATA)
                .filter { (it.flags and ApplicationInfo.FLAG_SYSTEM) == 0 }
                .map { it.packageName }
                .sorted()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get installed packages", e)
            emptyList()
        }
    }
}
