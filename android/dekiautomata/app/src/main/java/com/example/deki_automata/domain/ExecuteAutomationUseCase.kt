package com.example.deki_automata.domain

import android.content.Context
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.util.Log
import com.example.deki_automata.data.ActionRepository
import com.example.deki_automata.service.DeviceController

// TODO Extract context
class ExecuteAutomationUseCase(
    private val repository: ActionRepository,
    private val context: Context // for PackageManager
) {

    companion object {
        private const val TAG = "ExecuteAutomationUC"
    }

    private fun getInstalledPackages(): List<String> {
        val pm = context.packageManager
        return try {
            val packages =
                pm.getInstalledApplications(PackageManager.GET_META_DATA) // TODO update the method to the latest one
            packages.filter { (it.flags and ApplicationInfo.FLAG_SYSTEM) == 0 }
                .map { it.packageName }
                .sorted()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get installed packages", e)
            emptyList()
        }
    }

    // TODO remove maxSteps from client and move it to backend
    suspend fun runAutomation(prompt: String, device: DeviceController): String {
        val maxSteps = 20
        val allInstalledPackages by lazy { getInstalledPackages() }

        for (step in 1..maxSteps) {
            Log.d(TAG, "Step $step: Capturing screen")
            val screenshot = device.captureScreenBase64()
                ?: return "Automation Failed: Screen capture error"

            val promptForThisRequest = buildPrompt(prompt, step, allInstalledPackages)
            Log.d(TAG, "Sending action request with prompt:\n$promptForThisRequest")

            val result = repository.sendActionRequest(
                prompt = promptForThisRequest,
                screenshotBase64 = screenshot,
            )

            val responseString = result.getOrElse { e ->
                Log.e(TAG, "Action request failed", e)
                return "Automation Failed: Network request failed - ${e.message}"
            }

            Log.d(TAG, "Parsing response: $responseString")
            val command = ActionParser.parseResponse(responseString)
            if (command is AutomationCommand.RetryCapture) {
                Log.i(TAG, "Backend reports loading screen - sending a fresh capture")
                device.waitForIdle()
                continue
            }

            if (command is AutomationCommand.ShowMessage) {
                Log.i(TAG, "Automation finished. Message: ${command.message}")
                if (command.message.equals("Finished", true)) {
                    device.returnToApp()
                }
                return command.message
            }

            Log.i(TAG, "Executing command: $command")
            val success = when (command) {
                is AutomationCommand.Swipe -> device.swipe(command.direction, command.startX, command.startY)
                is AutomationCommand.Tap -> device.tap(command.x, command.y)
                is AutomationCommand.InsertText -> device.insertText(command.x, command.y, command.text)
                is AutomationCommand.OpenApp -> device.openApp(command.packageName)
                is AutomationCommand.GoHome -> device.pressHome()
                is AutomationCommand.GoBack -> device.goBack()
                else -> true
            }

            if (!success) {
                Log.e(TAG, "Command execution failed: $command")
                device.returnToApp()
                return "Automation Failed: Could not execute command $command"
            }

            device.waitForIdle()
        }

        Log.w(TAG, "Max steps reached")
        device.returnToApp()
        return "Automation Stopped: Maximum steps reached"
    }

    private fun buildPrompt(
        userPrompt: String,
        step: Int,
        installedPackages: List<String>
    ): String {
        return if (step == 1) {
            buildString {
                appendLine("User Prompt: \"$userPrompt\"")
                appendLine()
                appendLine("Installed Non‑System Packages:")
                installedPackages.forEach(::appendLine)
            }
        } else {
            userPrompt
        }
    }
}
