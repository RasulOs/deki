package com.example.deki_automata.domain

import android.util.Log
import com.example.deki_automata.domain.model.AutomationCommand
import com.example.deki_automata.domain.model.Direction
import com.example.deki_automata.domain.model.ParsedResult

object ActionParser {
    private const val TAG = "ActionParser"
    private val commandRegex = Regex("(?i)\"(Swipe|Tap|Insert text|Open|Go home|Go back|Finished|Can't proceed|Screen is in a loading state)\".*")
    private val numberRegex = Regex("\\b\\d+\\b")
    private const val ANSWER_KEYWORD = "Answer:"
    private const val DEFAULT_CENTER_X = 540
    private const val DEFAULT_CENTER_Y = 1160
    private const val NUMBER_OF_AXIS = 2

    fun parseResponse(response: String): ParsedResult {
        Log.d(TAG, "Parsing raw response: '$response'")

        if (response.trim().startsWith(ANSWER_KEYWORD, ignoreCase = true)) {
            val message = response.trim().substringAfter(ANSWER_KEYWORD).trim()
            Log.d(TAG, "Detected '$ANSWER_KEYWORD' pattern. Treating as terminal ShowMessage")
            return ParsedResult(message = null, command = AutomationCommand.ShowMessage(message))
        }

        val commandMatch = commandRegex.find(response)

        val message: String?
        val commandString: String

        if (commandMatch != null) {
            message = response.substring(0, commandMatch.range.first).trim().takeIf { it.isNotEmpty() }
            commandString = commandMatch.value
        } else {
            message = null
            commandString = response
        }

        Log.d(TAG, "Extracted Message: '$message'")
        Log.d(TAG, "Extracted Command String: '$commandString'")

        val automationCommand = parseCommandString(commandString)

        return ParsedResult(message, automationCommand)
    }

    private fun parseCommandString(commandString: String): AutomationCommand {
        val cleaned = commandString.trim().removeSurrounding("\"")
        val lower = cleaned.lowercase()

        return when {
            lower == "go home" -> AutomationCommand.GoHome
            lower == "go back" -> AutomationCommand.GoBack
            cleaned.startsWith("Swipe", ignoreCase = true) -> parseSwipeCommand(cleaned)
            cleaned.startsWith("Tap", ignoreCase = true) -> parseTapCommand(cleaned)
            cleaned.startsWith("Insert text", ignoreCase = true) -> parseInsertTextCommand(cleaned)
            cleaned.startsWith("Open", ignoreCase = true) -> parseOpenAppCommand(cleaned)
            cleaned.startsWith("Screen is in a loading state", ignoreCase = true) -> AutomationCommand.RetryCapture
            lower == "finished" || lower == "can't proceed" || lower == "step limit is reached" -> AutomationCommand.ShowMessage(cleaned)
            else -> AutomationCommand.ShowMessage(cleaned) // Fallback
        }
    }


    private fun parseSwipeCommand(cleaned: String): AutomationCommand {
        val direction = when {
            "left" in cleaned.lowercase() -> Direction.LEFT
            "right" in cleaned.lowercase() -> Direction.RIGHT
            "up" in cleaned.lowercase() || "top" in cleaned.lowercase() -> Direction.UP
            "down" in cleaned.lowercase() || "bottom" in cleaned.lowercase() -> Direction.DOWN
            else -> Direction.DOWN
        }
        val nums = numberRegex.findAll(cleaned).mapNotNull { it.value.toIntOrNull() }.toList()
        val x = nums.getOrNull(0) ?: DEFAULT_CENTER_X
        val y = nums.getOrNull(1) ?: DEFAULT_CENTER_Y
        return AutomationCommand.Swipe(direction, x, y)
    }

    private fun parseTapCommand(cleaned: String): AutomationCommand {
        val nums = numberRegex.findAll(cleaned).mapNotNull { it.value.toIntOrNull() }.toList()
        if (nums.size < NUMBER_OF_AXIS) return showParseError(cleaned, "Tap command missing coordinates")
        return AutomationCommand.Tap(nums[0], nums[1])
    }

    private fun parseInsertTextCommand(cleaned: String): AutomationCommand {
        val idx = cleaned.indexOf(':')
        if (idx == -1) return showParseError(cleaned, "InsertText command missing ':'")

        val coords = cleaned.substringBefore(':')
        val text = cleaned.substringAfter(':').trim()
        val nums = numberRegex.findAll(coords).mapNotNull { it.value.toIntOrNull() }.toList()
        if (nums.size < NUMBER_OF_AXIS) return showParseError(cleaned, "InsertText missing coordinates")
        return AutomationCommand.InsertText(nums[0], nums[1], text)
    }

    private fun parseOpenAppCommand(cleaned: String): AutomationCommand {
        val pkg = cleaned.split(Regex("\\s+"), limit = 2).getOrNull(1)?.trim().orEmpty()
        if (!pkg.contains('.')) return showParseError(cleaned, "OpenApp invalid package name")
        return AutomationCommand.OpenApp(pkg)
    }

    private fun showParseError(response: String, msg: String): AutomationCommand {
        Log.e(TAG, "$msg: '$response'")
        return AutomationCommand.ShowMessage("Error parsing command: $msg")
    }
}
