package com.example.deki_automata.presentation

import android.Manifest
import android.app.Activity
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ResolveInfo
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.speech.RecognizerIntent
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.example.deki_automata.R
import com.example.deki_automata.data.repository.ActionRepositoryImpl
import com.example.deki_automata.domain.usecase.ExecuteAutomationUseCase
import com.example.deki_automata.service.AutomataAccessibilityService
import com.example.deki_automata.ui.theme.DekiAutomataTheme
import com.example.deki_automata.util.ResultEventBus
import kotlinx.coroutines.flow.collectLatest

class MainActivity : ComponentActivity() {

    private companion object {
        private const val TAG = "MainActivity"
    }

    private val viewModel: MainViewModel by viewModels {
        MainViewModelFactory(ActionRepositoryImpl(), this.application)
    }

    private lateinit var mediaProjectionManager: MediaProjectionManager

    private val requestRecordAudioPermissionLauncher: ActivityResultLauncher<String> =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            viewModel.processIntent(AutomationIntent.RecordAudioPermissionResult(granted = isGranted))
            if (!isGranted) Log.w(TAG, "Record Audio permission denied")
            else Log.i(TAG, "Record Audio permission granted")
        }

    private val speechLauncher: ActivityResultLauncher<Intent> =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            var recognizedText: String? = null
            if (result.resultCode == Activity.RESULT_OK) {
                val data = result.data
                val matches = data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
                if (!matches.isNullOrEmpty()) {
                    recognizedText = matches[0]
                    Log.i(TAG, "Speech recognized: $recognizedText")
                    viewModel.processIntent(AutomationIntent.PromptTextChanged(recognizedText))
                } else {
                    Log.w(TAG, "Speech recognition returned no matches")
                    viewModel.processIntent(AutomationIntent.ResetResult)
                }
            } else {
                Log.w(TAG, "Speech recognition failed or was cancelled. Result code: ${result.resultCode}")
                viewModel.processIntent(AutomationIntent.ResetResult)
            }
        }

    private val requestMediaProjectionLauncher: ActivityResultLauncher<Intent> =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK && result.data != null) {
                Log.i(TAG, "Media Projection permission GRANTED by user")

                val serviceIntent = Intent(this, AutomataAccessibilityService::class.java).apply {
                    action = AutomataAccessibilityService.ACTION_START_MEDIA_PROJECTION
                    putExtra(AutomataAccessibilityService.EXTRA_RESULT_CODE, result.resultCode)
                    putExtra(AutomataAccessibilityService.EXTRA_RESULT_DATA, result.data)
                }

                ContextCompat.startForegroundService(this, serviceIntent)

                viewModel.processIntent(AutomationIntent.MediaProjectionSetResult(success = true))
                Log.d(TAG, "Foreground service start requested for Media Projection")

            } else {
                Log.w(TAG, "Media Projection permission DENIED or failed")
                viewModel.processIntent(AutomationIntent.MediaProjectionSetResult(success = false))
            }
        }

    private val openAccessibilitySettingsLauncher: ActivityResultLauncher<Intent> =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) {
            Log.d(TAG, "Returned from Accessibility Settings")
        }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        mediaProjectionManager = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager

        createNotificationChannel()

        setContent {
            DekiAutomataTheme {
                val uiState by viewModel.uiState.collectAsState()

                LaunchedEffect(key1 = viewModel) {
                    ResultEventBus.events.collectLatest { message ->
                        viewModel.setResultMessage(message)
                    }
                }

                LaunchedEffect(Unit) {
                    Log.d(TAG, "LaunchedEffect: Checking permissions and requesting Audio")
                    viewModel.processIntent(AutomationIntent.CheckPermissionsAndServices)
                    requestRecordAudioPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                }

                MainScreen(
                    state = uiState,
                    onSendClicked = { viewModel.processIntent(AutomationIntent.SubmitPrompt) },
                    onTextChanged = { viewModel.processIntent(AutomationIntent.PromptTextChanged(it)) },
                    onVoiceClicked = {
                        if (uiState.isRecordAudioGranted) {
                            launchSpeechRecognizer()
                        } else {
                            Log.w(TAG, "Voice clicked but Record Audio permission not granted")
                            viewModel.processIntent(AutomationIntent.RequestRecordAudioPermission)
                        }
                    },
                    onRequestAccessibility = ::openAccessibilitySettings,
                    onRequestRecordAudio = { requestRecordAudioPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO) },
                    onRequestMediaProjection = { requestMediaProjectionLauncher.launch(mediaProjectionManager.createScreenCaptureIntent()) },
                    onResetResult = { viewModel.clearResultMessage() },
                    onDismissPermissionGuidance = { viewModel.processIntent(AutomationIntent.DismissPermissionGuidance) },
                    onModeToggled = { isEnabled ->
                        viewModel.processIntent(AutomationIntent.ModeToggled(isEnabled))
                    },
                )
            }
        }
    }

    override fun onResume() {
        super.onResume()
        Log.d(TAG, "onResume: Re-checking permissions and services")
        viewModel.processIntent(AutomationIntent.CheckPermissionsAndServices)
    }

    override fun onStop() {
        super.onStop()
        Log.d(TAG, "onStop called")
    }

    private fun launchSpeechRecognizer() {
        val recognizerIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
        val activities: List<ResolveInfo> = packageManager.queryIntentActivities(
            recognizerIntent,
            PackageManager.MATCH_DEFAULT_ONLY
        )

        if (activities.isEmpty()) {
            Log.e(TAG, "Speech recognition not available on this device")
            viewModel.processIntent(AutomationIntent.ResetResult)
            // TODO show error message to user (snackbar or updating state)
            return
        }

        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_PROMPT, "Speak your request")
            // TODO language ? Get from user settings
            // putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US")
        }
        try {
            viewModel.processIntent(AutomationIntent.ResetResult)
            viewModel.processIntent(AutomationIntent.RequestVoiceInput)
            speechLauncher.launch(intent)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to launch speech recognizer", e)
            viewModel.processIntent(AutomationIntent.ResetResult)
        }
    }

    private fun openAccessibilitySettings() {
        Log.d(TAG, "Opening Accessibility Settings")
        val intent = Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
        try {
            openAccessibilitySettingsLauncher.launch(intent)
        } catch (e: Exception) {
            Log.e(TAG, "Could not open accessibility settings directly", e)
            try {
                startActivity(Intent(Settings.ACTION_SETTINGS))
            } catch (e2: Exception) {
                Log.e(TAG, "Could not open general settings either", e2)
                // TODO show error message to user
            }
        }
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val name = getString(R.string.media_projection_notification_channel_name)
        val descriptionText = getString(R.string.media_projection_notification_channel_description)
        val importance = NotificationManager.IMPORTANCE_DEFAULT
        val channel = NotificationChannel(AutomataAccessibilityService.NOTIFICATION_CHANNEL_ID, name, importance).apply {
                description = descriptionText
            }
        val notificationManager: NotificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.createNotificationChannel(channel)
        Log.d(TAG, "Notification channel created")
    }
}