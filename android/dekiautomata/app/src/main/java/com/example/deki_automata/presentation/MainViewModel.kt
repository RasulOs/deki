package com.example.deki_automata.presentation

import android.accessibilityservice.AccessibilityServiceInfo
import android.content.Context
import android.util.Log
import android.view.accessibility.AccessibilityManager
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.deki_automata.data.ActionRepository
import com.example.deki_automata.data.ActionRepositoryImpl
import com.example.deki_automata.domain.ExecuteAutomationUseCase
import com.example.deki_automata.service.AutomataAccessibilityService
import com.example.deki_automata.service.ServiceInternalState
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

data class MainUiState(
    val promptText: String = "",
    val isListening: Boolean = false,
    val isProcessing: Boolean = false,
    val resultMessage: String? = null,
    val serviceStatusMessage: String = "Checking service status...",
    val isAccessibilityEnabled: Boolean = false,
    val isScreenCaptureReady: Boolean = false,
    val isRecordAudioGranted: Boolean = false,
    val showPermissionGuidance: Boolean = false
) {
    val needsAccessibilityPermission get() = !isAccessibilityEnabled
    val needsRecordAudioPermission get() = !isRecordAudioGranted
    val needsMediaProjectionPermission get() = isAccessibilityEnabled && !isScreenCaptureReady
}

sealed class AutomationIntent {
    data object RequestVoiceInput : AutomationIntent()
    data class PromptTextChanged(val text: String) : AutomationIntent()
    data object SubmitPrompt : AutomationIntent()
    data object ResetResult : AutomationIntent()
    data object CheckPermissionsAndServices : AutomationIntent()
    data class RecordAudioPermissionResult(val granted: Boolean) : AutomationIntent()
    data class MediaProjectionSetResult(val success: Boolean) : AutomationIntent()
    data object RequestAccessibilitySettings : AutomationIntent()
    data object RequestRecordAudioPermission : AutomationIntent()
    data object RequestMediaProjectionPermission : AutomationIntent()
    data object DismissPermissionGuidance : AutomationIntent()
}

class MainViewModel(
    private val repository: ActionRepository,
    private val useCase: ExecuteAutomationUseCase
) : ViewModel() {

    private val _uiState = MutableStateFlow(MainUiState())
    val uiState: StateFlow<MainUiState> = _uiState.asStateFlow()

    private var serviceListenerJob: Job? = null
    private var automationJob: Job? = null
    private var latestServiceState = ServiceInternalState(null, false)

    init {
        listenForServiceState()
    }

    private fun listenForServiceState() {
        serviceListenerJob?.cancel()
        serviceListenerJob = viewModelScope.launch {
            AutomataAccessibilityService.serviceInternalStateFlow.collect { serviceState ->
                val wasAvailable = latestServiceState.controller != null
                val wasCaptureReady = latestServiceState.isCaptureReady
                latestServiceState = serviceState

                val isNowAvailable = serviceState.controller != null
                val isNowCaptureReady = serviceState.isCaptureReady

                _uiState.update { current ->
                    current.copy(
                        isAccessibilityEnabled = isNowAvailable,
                        isScreenCaptureReady = isNowCaptureReady,
                        serviceStatusMessage =
                        if (isNowAvailable)
                            "Service: Connected" +
                                    if (isNowCaptureReady) "/Capture Ready" else "/Capture Not Ready"
                        else "Service: Disconnected",
                        showPermissionGuidance = current.showPermissionGuidance ||
                                (wasAvailable && !isNowAvailable) ||
                                (wasCaptureReady && !isNowCaptureReady)
                    )
                }

                if ((wasAvailable && !isNowAvailable) ||
                    (wasCaptureReady && !isNowCaptureReady)
                ) {
                    if (automationJob?.isActive == true) {
                        automationJob?.cancelAndJoin()
                        _uiState.update {
                            it.copy(
                                isProcessing = false,
                                resultMessage = "Automation stopped: Service/Capture unavailable",
                            )
                        }
                    }
                }
            }
        }
    }

    fun processIntent(intent: AutomationIntent, context: Context? = null) {
        when (intent) {
            AutomationIntent.RequestVoiceInput ->
                _uiState.update { it.copy(isListening = true, promptText = "", resultMessage = null) }

            is AutomationIntent.PromptTextChanged ->
                _uiState.update { it.copy(promptText = intent.text, isListening = false) }

            AutomationIntent.SubmitPrompt -> handleSubmitPrompt()

            AutomationIntent.ResetResult -> {
                _uiState.update {
                    it.copy(
                        isProcessing = false,
                        resultMessage = null,
                        promptText = "",
                        isListening = false,
                    )
                }
                viewModelScope.launch { automationJob?.cancelAndJoin() }
            }

            AutomationIntent.CheckPermissionsAndServices ->
                context?.let { checkCurrentState(it) }

            is AutomationIntent.RecordAudioPermissionResult -> {
                _uiState.update { curr ->
                    val needsOther = curr.needsAccessibilityPermission || curr.needsMediaProjectionPermission
                    curr.copy(
                        isRecordAudioGranted = intent.granted,
                        showPermissionGuidance =
                        if (intent.granted) curr.showPermissionGuidance && needsOther else true
                    )
                }
            }

            is AutomationIntent.MediaProjectionSetResult -> {
                _uiState.update { curr ->
                    val needsOther = curr.needsAccessibilityPermission || curr.needsRecordAudioPermission
                    curr.copy(
                        showPermissionGuidance =
                        if (intent.success) curr.showPermissionGuidance && needsOther else true
                    )
                }
            }

            AutomationIntent.RequestAccessibilitySettings ->
                _uiState.update { it.copy(showPermissionGuidance = true) }

            AutomationIntent.RequestRecordAudioPermission ->
                _uiState.update { it.copy(showPermissionGuidance = true) }

            AutomationIntent.RequestMediaProjectionPermission ->
                _uiState.update { it.copy(showPermissionGuidance = true) }

            AutomationIntent.DismissPermissionGuidance ->
                _uiState.update { it.copy(showPermissionGuidance = false) }
        }
    }

    private fun checkCurrentState(context: Context) {
        viewModelScope.launch {
            val isAccessibilityEnabled = isAccessibilityServiceEnabled(context)
            val currentServiceValue = latestServiceState
            val needsMic = !_uiState.value.isRecordAudioGranted
            val needsCapture = !currentServiceValue.isCaptureReady && isAccessibilityEnabled

            _uiState.update {
                it.copy(
                    isAccessibilityEnabled = isAccessibilityEnabled,
                    isScreenCaptureReady = currentServiceValue.isCaptureReady,
                    isRecordAudioGranted = !needsMic,
                    serviceStatusMessage =
                    if (isAccessibilityEnabled)
                        "Service: Connected" +
                                if (currentServiceValue.isCaptureReady) "/Capture Ready" else "/Capture Not Ready"
                    else "Service: Disconnected",
                    showPermissionGuidance = needsMic || needsCapture || !isAccessibilityEnabled
                )
            }
        }
    }

    private fun handleSubmitPrompt() {
        val state = _uiState.value
        val service = latestServiceState
        val prompt = state.promptText.trim()

        if (prompt.isEmpty()) {
            _uiState.update { it.copy(resultMessage = "Prompt cannot be empty") }
            return
        }
        if (service.controller == null) {
            _uiState.update {
                it.copy(
                    resultMessage = "Error: Accessibility Service not connected",
                    showPermissionGuidance = true,
                )
            }
            return
        }
        if (!service.isCaptureReady) {
            _uiState.update {
                it.copy(
                    resultMessage = "Error: Screen Capture not ready",
                    showPermissionGuidance = true,
                )
            }
            return
        }
        if (state.isProcessing) return

        _uiState.update { it.copy(isProcessing = true, resultMessage = null, isListening = false) }

        viewModelScope.launch {
            automationJob?.cancelAndJoin()
            automationJob = launch {
                val outcome = runCatching { useCase.runAutomation(prompt, service.controller) }
                outcome.onSuccess { result ->
                    if (isActive) {
                        _uiState.update {
                            it.copy(isProcessing = false, resultMessage = result, promptText = "")
                        }
                    }
                }.onFailure { e ->
                    if (isActive) {
                        _uiState.update {
                            it.copy(isProcessing = false, resultMessage = "Failed: ${e.message}")
                        }
                    }
                }
            }
        }
    }

    private fun isAccessibilityServiceEnabled(context: Context): Boolean {
        val am = context.getSystemService(Context.ACCESSIBILITY_SERVICE) as? AccessibilityManager
            ?: return false.also { Log.e("VM", "Acc Manager null") }
        val enabledServices = try {
            am.getEnabledAccessibilityServiceList(AccessibilityServiceInfo.FEEDBACK_ALL_MASK) ?: emptyList()
        } catch (e: Exception) {
            emptyList<AccessibilityServiceInfo>().also { Log.e("VM", "getEnabled failed", e) }
        }
        val expectedServiceName = "${context.packageName}/.service.AutomataAccessibilityService"
        val serviceEnabled = enabledServices.any { it.id.equals(expectedServiceName, ignoreCase = true) }
        Log.d("ViewModel", "Accessibility Service Check: ${if (serviceEnabled) "ENABLED" else "DISABLED"}")
        return serviceEnabled
    }

    override fun onCleared() {
        super.onCleared()
        serviceListenerJob?.cancel()
        viewModelScope.launch { automationJob?.cancelAndJoin() }
    }
}