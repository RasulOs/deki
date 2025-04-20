package com.example.deki_automata.presentation

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.deki_automata.ui.theme.DekiAutomataTheme

@OptIn(ExperimentalMaterial3Api::class, ExperimentalComposeUiApi::class)
@Composable
fun MainScreen(
    state: MainUiState,
    onSendClicked: () -> Unit,
    onTextChanged: (String) -> Unit,
    onVoiceClicked: () -> Unit,
    onRequestAccessibility: () -> Unit,
    onRequestRecordAudio: () -> Unit,
    onRequestMediaProjection: () -> Unit,
    onResetResult: () -> Unit,
    onDismissPermissionGuidance: () -> Unit
) {
    val keyboardController = LocalSoftwareKeyboardController.current

    Scaffold(topBar = {
        TopAppBar(title = { Text("deki automata") })
    }) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {

            PermissionStatusSection(
                state = state,
                onRequestAccessibility = onRequestAccessibility,
                onRequestRecordAudio = onRequestRecordAudio,
                onRequestMediaProjection = onRequestMediaProjection
            )

            Spacer(modifier = Modifier.height(16.dp))

            OutlinedTextField(
                value = state.promptText,
                onValueChange = onTextChanged,
                label = { Text("Your prompt") },
                placeholder = { Text("Enter or speak your request") },
                modifier = Modifier.fillMaxWidth(),
                enabled = !state.isProcessing && state.isAccessibilityEnabled && state.isScreenCaptureReady,
                singleLine = true,
                keyboardOptions = KeyboardOptions.Default.copy(imeAction = ImeAction.Send),
                keyboardActions = KeyboardActions(
                    onSend = {
                        if (!state.isProcessing &&
                            state.promptText.isNotBlank() &&
                            state.isAccessibilityEnabled &&
                            state.isScreenCaptureReady
                        ) {
                            keyboardController?.hide()
                            onSendClicked()
                        }
                    }
                )
            )

            Spacer(modifier = Modifier.height(8.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Button(
                    onClick = {
                        keyboardController?.hide()
                        onSendClicked()
                    },
                    enabled = !state.isProcessing && state.promptText.isNotBlank() && state.isAccessibilityEnabled && state.isScreenCaptureReady
                ) {
                    Icon(Icons.Default.Send, contentDescription = "Send Prompt")
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Send")
                }

                IconButton(
                    onClick = {
                        keyboardController?.hide()
                        onVoiceClicked()
                    },
                    enabled = !state.isProcessing && !state.isListening && state.isAccessibilityEnabled && state.isScreenCaptureReady && state.isRecordAudioGranted
                ) {
                    Icon(
                        Icons.Default.Mic,
                        contentDescription = "Voice Input",
                        tint = if (state.isListening) MaterialTheme.colorScheme.primary else LocalContentColor.current
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            if (state.isProcessing) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center,
                    modifier = Modifier.padding(vertical = 8.dp)
                ) {
                    CircularProgressIndicator(modifier = Modifier.size(24.dp))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Performing actions...", style = MaterialTheme.typography.bodyMedium)
                }
            }

            state.resultMessage?.let { message ->
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 16.dp),
                    elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            text = "Result:",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = message,
                            style = MaterialTheme.typography.bodyLarge,
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Button(onClick = onResetResult, modifier = Modifier.align(Alignment.End)) {
                            Text("Clear")
                        }
                    }
                }
            }

            if (state.showPermissionGuidance && (state.needsAccessibilityPermission || state.needsRecordAudioPermission || state.needsMediaProjectionPermission)) {
                PermissionGuidanceCard(
                    state = state,
                    onRequestAccessibility = onRequestAccessibility,
                    onRequestRecordAudio = onRequestRecordAudio,
                    onRequestMediaProjection = onRequestMediaProjection,
                    onDismiss = onDismissPermissionGuidance
                )
            }
        }
    }
}

@Composable
fun PermissionStatusSection(
    state: MainUiState,
    onRequestAccessibility: () -> Unit,
    onRequestRecordAudio: () -> Unit,
    onRequestMediaProjection: () -> Unit
) {
    Column(modifier = Modifier.fillMaxWidth()) {
        Text("Status:", style = MaterialTheme.typography.titleMedium)
        Spacer(modifier = Modifier.height(4.dp))

        StatusItem(
            label = "Accessibility Service",
            isGranted = state.isAccessibilityEnabled,
            isRequired = true,
            onRequestPermission = onRequestAccessibility,
            icon = { Icon(Icons.Default.Settings, contentDescription = null, modifier = Modifier.size(18.dp)) }
        )
        StatusItem(
            label = "Microphone Access",
            isGranted = state.isRecordAudioGranted,
            isRequired = true,
            onRequestPermission = onRequestRecordAudio,
            icon = { Icon(Icons.Default.Mic, contentDescription = null, modifier = Modifier.size(18.dp)) }
        )
        StatusItem(
            label = "Screen Capture",
            isGranted = state.isScreenCaptureReady,
            isRequired = true,
            onRequestPermission = onRequestMediaProjection,
            icon = { Icon(Icons.Default.Videocam, contentDescription = null, modifier = Modifier.size(18.dp)) }
        )
    }
}

@Composable
fun StatusItem(
    label: String,
    isGranted: Boolean,
    isRequired: Boolean,
    onRequestPermission: () -> Unit,
    icon: @Composable () -> Unit
) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp)
    ) {
        icon()
        Spacer(Modifier.width(8.dp))
        Text(
            text = "$label:",
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.weight(1f)
        )
        if (isGranted) {
            Text(
                "Enabled",
                color = Color(0xFF2E7D32),
                fontWeight = FontWeight.Bold,
                style = MaterialTheme.typography.bodyMedium
            )
        } else if (isRequired) {
            TextButton(onClick = onRequestPermission, contentPadding = PaddingValues(horizontal = 4.dp, vertical = 0.dp)) {
                Text("Enable", fontWeight = FontWeight.Bold, style = MaterialTheme.typography.bodyMedium)
            }
        } else {
            Text("Disabled", color = Color.Gray, style = MaterialTheme.typography.bodyMedium)
        }
    }
}

@Composable
fun PermissionGuidanceCard(
    state: MainUiState,
    onRequestAccessibility: () -> Unit,
    onRequestRecordAudio: () -> Unit,
    onRequestMediaProjection: () -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Permissions Required") },
        text = {
            Column {
                Text("This app needs the following permissions/settings to function:")
                Spacer(Modifier.height(8.dp))
                if (state.needsAccessibilityPermission) {
                    Text("- Accessibility Service: To perform taps and swipes.")
                }
                if (state.needsRecordAudioPermission) {
                    Text("- Microphone Access: To use voice input.")
                }
                if (state.needsMediaProjectionPermission) {
                    Text("- Screen Capture: To see the screen content.")
                }
            }
        },
        confirmButton = {
            Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                if (state.needsAccessibilityPermission) {
                    TextButton(onClick = {
                        onRequestAccessibility()
                        onDismiss()
                    }) { Text("Accessibility") }
                }
                if (state.needsRecordAudioPermission) {
                    TextButton(onClick = {
                        onRequestRecordAudio()
                        onDismiss()
                    }) { Text("Microphone") }
                }
                if (state.needsMediaProjectionPermission) {
                    TextButton(onClick = {
                        onRequestMediaProjection()
                        onDismiss()
                    }) { Text("Screen Capture") }
                }
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("Later") }
        }
    )
}


@Preview(showBackground = true, widthDp = 360)
@Composable
fun MainScreenPreview_AllDisabled() {
    DekiAutomataTheme {
        val previewState = MainUiState(
            promptText = "",
            isProcessing = false,
            resultMessage = null,
            isAccessibilityEnabled = false,
            isScreenCaptureReady = false,
            isRecordAudioGranted = false,
            showPermissionGuidance = false,
            serviceStatusMessage = "Service: Disabled"
        )
        MainScreen(
            state = previewState,
            onSendClicked = {}, onTextChanged = {}, onVoiceClicked = {},
            onRequestAccessibility = {}, onRequestRecordAudio = {}, onRequestMediaProjection = {},
            onResetResult = {}, onDismissPermissionGuidance = {}
        )
    }
}

@Preview(showBackground = true, widthDp = 360)
@Composable
fun MainScreenPreview_Ready() {
    DekiAutomataTheme {
        val previewState = MainUiState(
            promptText = "Send a message",
            isProcessing = false,
            resultMessage = null,
            isAccessibilityEnabled = true,
            isScreenCaptureReady = true,
            isRecordAudioGranted = true,
            showPermissionGuidance = false,
            serviceStatusMessage = "Service: Connected"
        )
        MainScreen(
            state = previewState,
            onSendClicked = {}, onTextChanged = {}, onVoiceClicked = {},
            onRequestAccessibility = {}, onRequestRecordAudio = {}, onRequestMediaProjection = {},
            onResetResult = {}, onDismissPermissionGuidance = {}
        )
    }
}

@Preview(showBackground = true, widthDp = 360)
@Composable
fun MainScreenPreview_Processing() {
    DekiAutomataTheme {
        val previewState = MainUiState(
            promptText = "Send a message",
            isProcessing = true,
            resultMessage = null,
            isAccessibilityEnabled = true,
            isScreenCaptureReady = true,
            isRecordAudioGranted = true,
            showPermissionGuidance = false,
            serviceStatusMessage = "Service: Connected"
        )
        MainScreen(
            state = previewState,
            onSendClicked = {}, onTextChanged = {}, onVoiceClicked = {},
            onRequestAccessibility = {}, onRequestRecordAudio = {}, onRequestMediaProjection = {},
            onResetResult = {}, onDismissPermissionGuidance = {}
        )
    }
}

@Preview(showBackground = true, widthDp = 360)
@Composable
fun MainScreenPreview_Result() {
    DekiAutomataTheme {
        val previewState = MainUiState(
            promptText = "",
            isProcessing = false,
            resultMessage = "Task finished successfully.",
            isAccessibilityEnabled = true,
            isScreenCaptureReady = true,
            isRecordAudioGranted = true,
            showPermissionGuidance = false,
            serviceStatusMessage = "Service: Connected"
        )
        MainScreen(
            state = previewState,
            onSendClicked = {}, onTextChanged = {}, onVoiceClicked = {},
            onRequestAccessibility = {}, onRequestRecordAudio = {}, onRequestMediaProjection = {},
            onResetResult = {}, onDismissPermissionGuidance = {}
        )
    }
}