package com.example.deki_automata.domain

import android.util.Log

object ActionParser {
    private const val TAG = "ActionParser"

    private val numberRegex = Regex("\\b\\d+\\b")
    private const val DEFAULT_SCREEN_WIDTH = 1080
    private const val DEFAULT_SCREEN_HEIGHT = 2316
    private const val DEFAULT_CENTER_X = DEFAULT_SCREEN_WIDTH / 2
    private const val DEFAULT_CENTER_Y = DEFAULT_SCREEN_HEIGHT / 2

    private val TERMINAL_STATUSES = setOf(
        "finished",
        "can't proceed",
        "step limit is reached"
    )

    fun parseResponse(response: String): AutomationCommand {
        val cleaned = response.trim().removeSurrounding("\"")
        Log.d(TAG, "Parsing cleaned response: '$cleaned'")
        val lower = cleaned.lowercase()

        return when {
            lower == "go home" -> AutomationCommand.GoHome.also { Log.d(TAG, "Parsed: GoHome") }
            lower == "go back" -> AutomationCommand.GoBack.also { Log.d(TAG, "Parsed: GoBack") }
            cleaned.startsWith("Swipe", ignoreCase = true) -> parseSwipeCommand(cleaned)
            cleaned.startsWith("Tap", ignoreCase = true) -> parseTapCommand(cleaned)
            cleaned.startsWith("Insert text", ignoreCase = true) -> parseInsertTextCommand(cleaned)
            cleaned.startsWith("Open", ignoreCase = true) -> parseOpenAppCommand(cleaned)
            cleaned.startsWith("Answer:", ignoreCase = true) -> {
                val msg = cleaned.substringAfter("Answer:").trim()
                AutomationCommand.ShowMessage(msg).also { Log.d(TAG, "Parsed: ShowMessage '$msg'") }
            }
            lower in TERMINAL_STATUSES -> AutomationCommand.ShowMessage(cleaned)
                .also { Log.d(TAG, "Parsed: ShowMessage (Status) '$cleaned'") }
            else -> AutomationCommand.ShowMessage(cleaned).also {
                Log.w(TAG, "Unknown command, treating as ShowMessage: '$cleaned'")
            }
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
        val nums = numberRegex.findAll(cleaned)
            .mapNotNull { it.value.toIntOrNull() }
            .toList()
        val x = nums.getOrNull(0) ?: DEFAULT_CENTER_X
        val y = nums.getOrNull(1) ?: DEFAULT_CENTER_Y
        Log.d(TAG, "Parsed: Swipe $direction at ($x, $y)")
        return AutomationCommand.Swipe(direction, x, y)
    }

    private fun parseTapCommand(cleaned: String): AutomationCommand {
        val nums = numberRegex.findAll(cleaned)
            .mapNotNull { it.value.toIntOrNull() }
            .toList()
        if (nums.size < 2) {
            Log.e(TAG, "Tap command missing coordinates: '$cleaned'")
            return AutomationCommand.ShowMessage("Error parsing Tap command (missing coordinates)")
        }
        val x = nums[0]
        val y = nums[1]
        Log.d(TAG, "Parsed: Tap at ($x, $y)")
        return AutomationCommand.Tap(x, y)
    }

    private fun parseInsertTextCommand(cleaned: String): AutomationCommand {
        val idx = cleaned.indexOf(':')
        if (idx == -1) return showParseError(cleaned, "InsertText command missing ':'")

        val coords = cleaned.substringBefore(':')
        val text = cleaned.substringAfter(':').trim()
        val nums = numberRegex.findAll(coords)
            .mapNotNull { it.value.toIntOrNull() }
            .toList()
        if (nums.size < 2) return showParseError(cleaned, "InsertText missing coordinates")

        val x = nums[0]
        val y = nums[1]
        Log.d(TAG, "Parsed: InsertText '$text' at ($x, $y)")
        return AutomationCommand.InsertText(x, y, text)
    }

    private fun parseOpenAppCommand(cleaned: String): AutomationCommand {
        val parts = cleaned.split(Regex("\\s+"), limit = 2)
        val pkg = parts.getOrNull(1)?.trim().orEmpty()
        if (!pkg.contains('.')) return showParseError(cleaned, "OpenApp invalid package name")

        Log.d(TAG, "Parsed: OpenApp '$pkg'")
        return AutomationCommand.OpenApp(pkg)
    }

    private fun showParseError(response: String, msg: String): AutomationCommand {
        Log.e(TAG, "$msg: '$response'")
        return AutomationCommand.ShowMessage("Error parsing command: $msg")
    }
}