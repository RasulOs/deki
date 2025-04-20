package com.example.deki_automata.service

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.AccessibilityServiceInfo
import android.accessibilityservice.GestureDescription
import android.app.Activity
import android.app.Notification
import android.app.PendingIntent
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.content.res.Resources
import android.graphics.Bitmap
import android.graphics.Path
import android.graphics.PixelFormat
import android.graphics.Rect
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Base64
import android.util.DisplayMetrics
import android.util.Log
import android.view.Display
import android.view.WindowManager
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import com.example.deki_automata.R
import com.example.deki_automata.domain.Direction
import com.example.deki_automata.presentation.MainActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.ByteArrayOutputStream
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.collections.ArrayDeque

// TODO refactor everything

data class ServiceInternalState(
    val controller: DeviceController?,
    val isCaptureReady: Boolean,
)

class AutomataAccessibilityService : AccessibilityService(), DeviceController {

    private val serviceJob = SupervisorJob()
    private val serviceScope = CoroutineScope(Dispatchers.Main + serviceJob)

    private lateinit var mediaProjectionManager: MediaProjectionManager

    private var currentMediaProjection: MediaProjection? = null
    private var imageReader: ImageReader? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var screenWidth: Int = 0
    private var screenHeight: Int = 0
    private var screenDensity: Int = 0
    private var isForegroundServiceRunning = false
    private val hasCaptureStartedProducingFrames = AtomicBoolean(false)
    private val imageListenerHandler = Handler(Looper.getMainLooper())
    private var statusBarHeightPx: Int = 0


    companion object {
        private const val TAG = "AutomataService"
        private val _serviceInternalState = MutableStateFlow(ServiceInternalState(null, false))
        val serviceInternalStateFlow: StateFlow<ServiceInternalState> = _serviceInternalState.asStateFlow()
        const val ACTION_START_MEDIA_PROJECTION = "com.example.deki_automata.action.START_PROJECTION"
        const val ACTION_STOP_MEDIA_PROJECTION = "com.example.deki_automata.action.STOP_PROJECTION"
        const val EXTRA_RESULT_CODE = "com.example.deki_automata.extra.RESULT_CODE"
        const val EXTRA_RESULT_DATA = "com.example.deki_automata.extra.RESULT_DATA"
        const val NOTIFICATION_CHANNEL_ID = "DekiAutomataMediaProjectionChannel"
        const val NOTIFICATION_ID = 1001
        private var instance: AutomataAccessibilityService? = null
        fun stopProjection(context: Context) {
            Log.d(TAG, "Static stopProjection called")
            val intent = Intent(context, AutomataAccessibilityService::class.java)
            intent.action = ACTION_STOP_MEDIA_PROJECTION
            try {
                context.startService(intent)
            } catch (e: Exception) {
                Log.e(TAG, "Error starting service to stop projection", e)
            }
        }
    }

    private fun updateInternalState(controller: DeviceController? = instance) {
        val isCaptureActuallyReady = hasCaptureStartedProducingFrames.get()
        val newState = ServiceInternalState(controller, isCaptureActuallyReady)
        if (_serviceInternalState.value != newState) {
            Log.d(
                TAG,
                "Updating internal state: Controller=${newState.controller != null}, CaptureReady=${newState.isCaptureReady}"
            )
            _serviceInternalState.value = newState
        }
    }

    override fun onCreate() {
        super.onCreate()
        mediaProjectionManager = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        Log.d(TAG, "Service onCreate")
    }

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
        Log.i(TAG, "Accessibility Service Connected")
        updateStatusBarHeight()
        updateScreenMetrics()
        updateInternalState(controller = this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val action = intent?.action
        Log.d(TAG, "onStartCommand received: Action=$action, StartId=$startId")

        // TODO update
        when (action) {
            ACTION_START_MEDIA_PROJECTION -> {
                val resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, Activity.RESULT_CANCELED)
                val resultData: Intent? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    intent.getParcelableExtra(EXTRA_RESULT_DATA, Intent::class.java)
                } else {
                    @Suppress("DEPRECATION")
                    intent.getParcelableExtra(EXTRA_RESULT_DATA)
                }
                if (resultCode == Activity.RESULT_OK && resultData != null) {
                    if (!isForegroundServiceRunning || currentMediaProjection == null) {
                        startMediaProjectionForegroundService(resultCode, resultData)
                    } else {
                        Log.w(TAG, "Start projection requested but already running")
                    }
                } else {
                    Log.e(TAG, "Invalid result code ($resultCode) or data for media projection start")
                    if (!isForegroundServiceRunning) stopSelf(startId)
                }
            }

            ACTION_STOP_MEDIA_PROJECTION -> {
                Log.i(TAG, "Stop projection action received")
                stopProjectionCapture()
                stopSelf(startId)
            }

            else -> {
                Log.w(TAG, "Received unhandled action ($action) or null intent")
                if (!isForegroundServiceRunning) stopSelf(startId)
            }
        }
        return START_NOT_STICKY
    }

    override fun onInterrupt() {
        Log.w(TAG, "Accessibility Service Interrupted")
        stopProjectionCapture()
        instance = null
        updateInternalState(controller = null)
        serviceJob.cancel()
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.w(TAG, "Accessibility Service Destroyed")
        stopProjectionCapture()
        instance = null
        updateInternalState(controller = null)
        serviceJob.cancel()
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        event?.recycle()
    }

    private fun startMediaProjectionForegroundService(resultCode: Int, resultData: Intent) {
        Log.i(TAG, "Attempting to start FGS and get Media Projection")
        val notification = createNotification()
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION)
            } else {
                startForeground(NOTIFICATION_ID, notification)
            }
            isForegroundServiceRunning = true
            Log.d(TAG, "Service promoted to foreground")
            val projection = mediaProjectionManager.getMediaProjection(resultCode, resultData)
            if (projection != null) {
                Log.i(TAG, "MP obtained")
                handleMediaProjectionSet(projection)
            } else {
                Log.e(TAG, "getMediaProjection null")
                stopProjectionCapture()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Exception starting FGS/getting MP", e)
            isForegroundServiceRunning = false
            try {
                ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
            } catch (_: Exception) {
            }
            updateInternalState()
        }
    }

    private fun createNotification(): Notification {
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        }
        val flags =
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE else PendingIntent.FLAG_UPDATE_CURRENT
        val pi = PendingIntent.getActivity(this, 0, intent, flags)
        return NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setContentTitle(getString(R.string.media_projection_notification_title))
            .setContentText(getString(R.string.media_projection_notification_text))
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentIntent(pi)
            .setOngoing(true)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun stopProjectionCapture() {
        Log.i(TAG, "Stopping projection capture and foreground service state")
        cleanupMediaProjection()
        if (isForegroundServiceRunning) {
            try {
                ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
                Log.d(
                    TAG,
                    "Service stopped FGS"
                )
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping FGS", e)
            } finally {
                isForegroundServiceRunning = false
            }
        }
    }

    private fun handleMediaProjectionSet(projection: MediaProjection?) {
        if (projection != null) {
            Log.d(TAG, "Handling set of MediaProjection: $projection")
            if (currentMediaProjection != null && currentMediaProjection != projection) {
                cleanupMediaProjection()
            }
            if (currentMediaProjection == null) {
                hasCaptureStartedProducingFrames.set(false)
                initScreenCapture(projection)
            } else {
                Log.w(TAG, "Projection already set")
            }
        } else {
            Log.d(TAG, "MP handle set to null. Cleaning up")
            cleanupMediaProjection()
        }
    }

    private fun initScreenCapture(projection: MediaProjection) {
        if (currentMediaProjection === projection && virtualDisplay != null) {
            Log.w(TAG, "initScreenCapture skipping: same projection")
            return
        }
        if (currentMediaProjection != null || imageReader != null || virtualDisplay != null) {
            cleanupMediaProjection()
        }
        if (screenWidth <= 0 || screenHeight <= 0) {
            updateScreenMetrics()
            if (screenWidth <= 0 || screenHeight <= 0) {
                if (isForegroundServiceRunning) stopProjectionCapture()
                updateInternalState()
                return
            }
        }

        Log.i(TAG, "Initializing screen capture resources for projection: $projection")
        currentMediaProjection = projection
        hasCaptureStartedProducingFrames.set(false)
        updateInternalState()

        try {
            currentMediaProjection?.registerCallback(mediaProjectionCallback, null)
            imageReader = ImageReader.newInstance(screenWidth, screenHeight, PixelFormat.RGBA_8888, 2).apply {
                setOnImageAvailableListener({ reader ->
                    try {
                        reader?.acquireLatestImage()?.close()
                        if (hasCaptureStartedProducingFrames.compareAndSet(
                                false,
                                true
                            )
                        ) {
                            Log.i(TAG, "First image. Signaling ready")
                            updateInternalState()
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Error in ImageListener", e)
                    }
                }, imageListenerHandler)
            }
            Log.d(TAG, "ImageReader created ($screenWidth x $screenHeight)")
            virtualDisplay = currentMediaProjection?.createVirtualDisplay(
                "DekiAutomataCapture", // TODO get names from string resources or const val
                screenWidth,
                screenHeight,
                screenDensity,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                imageReader?.surface,
                null,
                null,
            )
            if (virtualDisplay != null) {
                Log.i(TAG, "VD created. Waiting for first image")
            } else {
                Log.e(TAG, "Failed VD create")
                cleanupMediaProjection()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Exception initializing IR/VD", e)
            cleanupMediaProjection()
        }
    }

    private fun cleanupMediaProjection() {
        Log.i(TAG, "Cleaning up media projection internal resources")
        val needsStateUpdate =
            (currentMediaProjection != null || imageReader != null || virtualDisplay != null || hasCaptureStartedProducingFrames.get())
        val mp = currentMediaProjection
        val vd = virtualDisplay
        val ir = imageReader
        currentMediaProjection = null
        virtualDisplay = null
        imageReader = null
        hasCaptureStartedProducingFrames.set(false)
        mp?.let { proj ->
            try {
                proj.unregisterCallback(mediaProjectionCallback)
            } catch (_: Exception) {
            }
            try {
                proj.stop()
            } catch (_: Exception) {
            }
        }
        vd?.let { display ->
            try {
                display.release()
            } catch (_: Exception) {
            }
        }
        ir?.let { reader ->
            try {
                reader.close()
            } catch (_: Exception) {
            }
        }
        Log.d(TAG, "Internal media projection resource cleanup finished")
        if (needsStateUpdate) updateInternalState()
    }

    private val mediaProjectionCallback = object : MediaProjection.Callback() {
        override fun onStop() {
            Log.w(TAG, "MP Callback onStop()")
            stopProjectionCapture()
        }
    }

    private fun updateStatusBarHeight() {
        // TODO TBD if needed
    }

    private fun updateScreenMetrics() {
        updateStatusBarHeight()
        val windowManager =
            getSystemService(Context.WINDOW_SERVICE) as? WindowManager ?: return Unit.also { Log.e(TAG, "WM null") }
        // TODO update
        @Suppress("DEPRECATION")
        val display = windowManager.defaultDisplay ?: return Unit.also { Log.e(TAG, "Display null") }
        val metrics = DisplayMetrics()
        try {
            display.getRealMetrics(metrics)
            val oldW = screenWidth
            val oldH = screenHeight
            screenWidth = metrics.widthPixels
            screenHeight = metrics.heightPixels
            screenDensity = metrics.densityDpi
            Log.i(
                TAG,
                "Metrics Updated: $screenWidth x $screenHeight"
            )
            // TODO update
            if (currentMediaProjection != null && (oldW != screenWidth || oldH != screenHeight)) {
                Log.w(TAG, "Resizing capture")
                initScreenCapture(currentMediaProjection!!)
            }
        } catch (e: Exception) {
            Log.e(TAG, "getRealMetrics failed", e)
            screenWidth = 0
            screenHeight = 0
        }
    }

    // TODO put numbers in const vals
    override fun captureScreenBase64(): String? {
        if (!hasCaptureStartedProducingFrames.get() || imageReader == null || currentMediaProjection == null) {
            Log.e(
                TAG,
                "Capture components not ready. HasFrame=${hasCaptureStartedProducingFrames.get()}, IR=${imageReader != null}, MP=${currentMediaProjection != null}"
            )
            return null
        }
        val reader = imageReader!!
        var image: Image? = null
        var attempt = 0
        val maxAttempts = 3
        val retryDelayMs = 50L
        while (image == null && attempt < maxAttempts) {
            attempt++
            try {
                image = reader.acquireLatestImage()
                if (image == null && attempt < maxAttempts) {
                    Thread.sleep(retryDelayMs)
                }
            } catch (e: Exception) {
                if (attempt == maxAttempts) return null
                Thread.sleep(retryDelayMs)
            }
        }
        if (image == null) {
            Log.e(TAG, "Failed acquire after $maxAttempts attempts")
            return null
        }
        try {
            val width = image.width
            val height = image.height
            val planes = image.planes
            if (width <= 0 || height <= 0 || planes.isEmpty()) {
                Log.e(TAG, "Inv dims ${width}x$height")
                return null
            }
            val buffer = planes[0].buffer
            val pS = planes[0].pixelStride
            val rS = planes[0].rowStride
            val rP = rS - pS * width
            if (pS <= 0 || rS <= 0 || buffer == null || rP < 0) {
                Log.e(TAG, "Inv buf params")
                return null
            }
            val cap = rS * height
            if (buffer.remaining() < cap) {
                Log.e(TAG, "Buf cap low ${buffer.remaining()}<$cap")
                return null
            }
            val bmpW = if (rP > 0) width + rP / pS else width
            if (bmpW <= 0) {
                Log.e(TAG, "Inv bmp W")
                return null
            }
            var bmp: Bitmap? = null
            var crp: Bitmap? = null
            var b64: String? = null
            try {
                bmp = Bitmap.createBitmap(bmpW, height, Bitmap.Config.ARGB_8888)
                bmp.copyPixelsFromBuffer(buffer)
                crp = if (rP > 0) {
                    Bitmap.createBitmap(bmp, 0, 0, width, height)
                } else {
                    bmp
                }
                val o = ByteArrayOutputStream()
                crp.compress(Bitmap.CompressFormat.WEBP, 85, o) // TODO experiment with quality
                b64 = Base64.encodeToString(o.toByteArray(), Base64.NO_WRAP)
            } catch (e: Exception) {
                Log.e(TAG, "Bmp/Enc Exc", e)
                b64 = null
            } finally {
                if (crp != null && crp !== bmp) {
                    crp.recycle()
                }
                bmp?.recycle()
            }
            if (b64 != null) Log.d(TAG, "Capture OK (Atmpt $attempt), len: ${b64.length}")
            else Log.e(TAG, "Capture Fail processing (Atmpt $attempt)")
            return b64
        } catch (e: Exception) {
            Log.e(TAG, "Capture Generic Exc", e)
            return null
        } finally {
            try {
                image.close()
            } catch (e: Exception) {
                Log.e(TAG, "Error closing image", e)
            }
        }
    }

    override fun swipe(direction: Direction, startX: Int, startY: Int): Boolean {
        if (screenWidth <= 0 || screenHeight <= 0) {
            Log.e(TAG, "Swipe invalid screen dimensions ($screenWidth, $screenHeight)")
            return false
        }

        // TODO put constants in const vals
        val horizontalMargin = (screenWidth * 0.1f).toInt().coerceAtLeast(30)
        val verticalMargin = (screenHeight * 0.1f).toInt().coerceAtLeast(50)
        val verticalSwipeDistance = (screenHeight * 0.70f).toInt().coerceAtLeast(150)
        val horizontalSwipeDistance = (screenWidth * 0.80f).toInt().coerceAtLeast(150)

        // TODO const val
        val durationMs = 500L

        val clampedStartX: Int
        val clampedStartY: Int
        val finalEndX: Int
        val finalEndY: Int

        when (direction) {
            Direction.LEFT -> {
                clampedStartY = startY.coerceIn(verticalMargin, screenHeight - 1 - verticalMargin)
                clampedStartX = (screenWidth - horizontalMargin).coerceIn(0, screenWidth - 1)
                finalEndX = (clampedStartX - horizontalSwipeDistance).coerceIn(0, screenWidth - 1)
                finalEndY = clampedStartY
                Log.d(
                    TAG,
                    "Swipe LEFT: Orig($startX,$startY), Start($clampedStartX,$clampedStartY), End($finalEndX,$finalEndY)"
                )
            }

            // TODO update
            Direction.RIGHT -> {
                clampedStartY = startY.coerceIn(verticalMargin, screenHeight - 1 - verticalMargin)
                clampedStartX = horizontalMargin
                finalEndX = (clampedStartX + horizontalSwipeDistance).coerceIn(0, screenWidth - 1)
                finalEndY = clampedStartY
                Log.d(
                    TAG,
                    "Swipe RIGHT: Orig($startX,$startY), Start($clampedStartX,$clampedStartY), End($finalEndX,$finalEndY)"
                )
            }

            Direction.UP -> {
                clampedStartX = startX.coerceIn(horizontalMargin, screenWidth - 1 - horizontalMargin)
                clampedStartY = startY.coerceIn(verticalMargin, screenHeight - 1 - verticalMargin)
                finalEndY = (clampedStartY - verticalSwipeDistance).coerceIn(0, screenHeight - 1)
                finalEndX = clampedStartX
                Log.d(
                    TAG,
                    "Swipe UP: Orig($startX,$startY), Start($clampedStartX,$clampedStartY), End($finalEndX,$finalEndY)"
                )
            }

            Direction.DOWN -> {
                clampedStartX = startX.coerceIn(horizontalMargin, screenWidth - 1 - horizontalMargin)
                clampedStartY = startY.coerceIn(verticalMargin, screenHeight - 1 - verticalMargin)
                finalEndY = (clampedStartY + verticalSwipeDistance).coerceIn(0, screenHeight - 1)
                finalEndX = clampedStartX
                Log.d(
                    TAG,
                    "Swipe DOWN: Orig($startX,$startY), Start($clampedStartX,$clampedStartY), End($finalEndX,$finalEndY)"
                )
            }
        }

        if (abs(finalEndX - clampedStartX) < 10 && abs(finalEndY - clampedStartY) < 10) {
            Log.w(
                TAG,
                "Swipe near-zero distance after calculations. Start=($clampedStartX,$clampedStartY), End=($finalEndX,$finalEndY)"
            )
            return false
        }

        Log.i(
            TAG,
            "Swiping $direction from ($clampedStartX, $clampedStartY) to ($finalEndX, $finalEndY) over ${durationMs}ms"
        )
        val path = Path().apply {
            moveTo(clampedStartX.toFloat(), clampedStartY.toFloat())
            lineTo(finalEndX.toFloat(), finalEndY.toFloat())
        }
        val gesture = GestureDescription.Builder()
            .addStroke(GestureDescription.StrokeDescription(path, 0L, durationMs))
            .build()

        val success = dispatchGesture(gesture, null, null)
        Log.d(TAG, "dispatchGesture result for swipe $direction: $success")
        if (!success) Log.e(TAG, "Swipe $direction dispatchGesture failed")

        if (success) {
            try {
                Thread.sleep(300)
            } catch (_: InterruptedException) {
            }
        }

        return success
    }

    override fun tap(x: Int, y: Int): Boolean {
        if (screenWidth <= 0 || screenHeight <= 0) {
            Log.e(TAG, "Tap invalid dims")
            return false
        }
        val adjustedY = y - statusBarHeightPx
        Log.d(TAG, "Tap: OrigY=$y, SBH=$statusBarHeightPx, AdjY=$adjustedY")
        val clampedX = x.coerceIn(0, screenWidth - 1)
        val clampedY = adjustedY.coerceIn(0, screenHeight - 1)
        Log.d(TAG, "Tapping at final coords ($clampedX, $clampedY)")
        val path = Path().apply { moveTo(clampedX.toFloat(), clampedY.toFloat()) }
        val tapDuration = 50L
        val gesture = GestureDescription.Builder()
            .addStroke(GestureDescription.StrokeDescription(path, 0L, tapDuration))
            .build()
        val success = dispatchGesture(gesture, null, null)

        Log.d(TAG, "dispatchGesture result for tap at ($clampedX, $clampedY): $success")
        if (!success) Log.e(TAG, "Tap dispatchGesture failed")

        return success
    }

    override fun insertText(x: Int, y: Int, text: String): Boolean {
        if (!tap(x, y)) {
            Log.e(TAG, "InsertText failed: Initial tap failed at ($x, $y)")
            return false
        }

        // TODO const val
        val waitDelayMs = 600L
        Log.d(TAG, "InsertText: Waiting ${waitDelayMs}ms")
        try {
            Thread.sleep(waitDelayMs)
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
            return false
        }

        val rootNode = rootInActiveWindow ?: run {
            Log.e(TAG, "InsertText failed: rootInActiveWindow is null after tap and wait")
            return false
        }

        var targetNode: AccessibilityNodeInfo? = null
        var insertionSuccess = false

        try {
            val focusedNode = rootNode.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)
            if (focusedNode != null) {
                Log.d(
                    TAG,
                    "InsertText: Found focused node: ID=${focusedNode.viewIdResourceName ?: "N/A"}, Cls=${focusedNode.className}, Edit=${focusedNode.isEditable}, Enab=${focusedNode.isEnabled}"
                )
                if (focusedNode.isEditable && focusedNode.isEnabled) {
                    targetNode = focusedNode
                } else {
                    Log.w(TAG, "InsertText: Focused node not suitable")
                    focusedNode.recycle()
                }
            } else {
                Log.d(TAG, "InsertText: No node has input focus after tap/wait")
            }

            if (targetNode == null) {
                Log.d(TAG, "InsertText: Searching for editable/enabled node containing ($x, $y)")
                targetNode = findEditableNodeAt(rootNode, x, y)
                if (targetNode != null) {
                    Log.d(
                        TAG,
                        "InsertText: Found node by location: ID=${targetNode.viewIdResourceName ?: "N/A"}, Cls=${targetNode.className}"
                    )
                    if (targetNode.isFocusable) {
                        Log.d(TAG, "InsertText: Attempting ACTION_FOCUS on node found by location")
                        targetNode.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
                        try {
                            Thread.sleep(150)
                        } catch (_: InterruptedException) {
                        }
                    }
                } else {
                    Log.w(TAG, "InsertText: No suitable editable/enabled node found by focus or location")
                }
            }

            if (targetNode != null) {
                if (!targetNode.refresh() || !targetNode.isEditable || !targetNode.isEnabled) {
                    Log.e(TAG, "InsertText: Target node became stale/invalid before action")
                    insertionSuccess = false
                } else {
                    Log.d(TAG, "InsertText: Attempting ACTION_PASTE")
                    val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
                    if (clipboard != null) {
                        val clip = ClipData.newPlainText("deki_automata_text", text)
                        clipboard.setPrimaryClip(clip)
                        try {
                            Thread.sleep(50) // TODO experiment with all delays to find optimal values
                        } catch (_: InterruptedException) {
                        }
                        insertionSuccess = targetNode.performAction(AccessibilityNodeInfo.ACTION_PASTE)
                        if (insertionSuccess) Log.i(TAG, "InsertText: PASTE OK")
                        else Log.w(TAG, "InsertText: PASTE failed")
                    } else {
                        Log.e(TAG, "InsertText: ClipboardManager null")
                    }

                    if (!insertionSuccess) {
                        Log.d(TAG, "InsertText: Falling back to SET_TEXT")
                        val args = Bundle().apply {
                            putCharSequence(
                                AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE,
                                text
                            )
                        }
                        insertionSuccess = targetNode.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)
                        if (insertionSuccess) Log.i(TAG, "InsertText: SET_TEXT OK")
                        else Log.e(TAG, "InsertText: SET_TEXT failed")
                    }
                    // TODO risky operation, TBD and then update
                    if (insertionSuccess) {
                        Log.d(TAG, "InsertText: Text inserted. Hiding keyboard via GLOBAL_ACTION_BACK")
                        try {
                            Thread.sleep(100)
                            val backSuccess = performGlobalAction(GLOBAL_ACTION_BACK)
                            Log.d(TAG, "InsertText: GLOBAL_ACTION_BACK result: $backSuccess")
                            Thread.sleep(350)
                        } catch (e: Exception) {
                            Log.e(TAG, "InsertText: Exception during GLOBAL_ACTION_BACK", e)
                        }
                    }
                }
            } else {
                Log.w(TAG, "InsertText: No target node found to insert text into")
                insertionSuccess = false
            }
        } catch (e: Exception) {
            Log.e(TAG, "InsertText: Exception during node search or action", e)
            insertionSuccess = false
        } finally {
            try {
                targetNode?.recycle()
            } catch (e: Exception) {
                Log.e(TAG, "Error recycling targetNode", e)
            }
            if (rootNode !== targetNode) {
                try {
                    rootNode?.recycle()
                } catch (e: Exception) {
                    Log.e(TAG, "Error recycling rootNode", e)
                }
            } else if (targetNode == null) { // TODO update
                try {
                    rootNode?.recycle()
                } catch (e: Exception) {
                    Log.e(TAG, "Error recycling rootNode target null", e)
                }
            }
        }

        return insertionSuccess
    }

    private fun findEditableNodeAt(rootNode: AccessibilityNodeInfo?, x: Int, y: Int): AccessibilityNodeInfo? {
        if (rootNode == null) return null
        val q = ArrayDeque<AccessibilityNodeInfo>()
        q.addLast(rootNode)
        while (q.isNotEmpty()) {
            val node = q.removeFirst()
            val bounds = Rect()
            var nodeInfo: String? = null
            try {
                if (!node.refresh()) {
                    try {
                        node.recycle()
                    } catch (_: Exception) {
                    }
                    continue
                }
                nodeInfo = "ID=${node.viewIdResourceName ?: "N/A"}, Cls=${node.className}"
                node.getBoundsInScreen(bounds)
                val isMatch = node.isEditable && node.isEnabled && bounds.contains(x, y)
                if (isMatch) {
                    Log.d(TAG, "findEditable: Match $nodeInfo")
                    q.forEach { rn ->
                        try {
                            rn.recycle()
                        } catch (_: Exception) {
                        }
                    }
                    return node
                } else {
                    for (i in 0 until node.childCount) {
                        node.getChild(i)?.let { q.addLast(it) }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "findEditable Exception: $nodeInfo", e)
            } finally {
                try {
                    node.recycle()
                } catch (e: Exception) {
                }
            }
        }
        Log.d(TAG, "findEditable: No match at ($x, $y)")
        return null
    }

    override fun openApp(packageName: String): Boolean {
        return try {
            val i = packageManager.getLaunchIntentForPackage(packageName)
                ?: return false
            i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_RESET_TASK_IF_NEEDED)
            startActivity(i)
            Thread.sleep(500) // TODO find optimal value
            Log.i(TAG, "Started $packageName")
            true
        } catch (e: Exception) {
            Log.e(TAG, "openApp failed $packageName", e)
            false
        }
    }

    override fun pressHome(): Boolean {
        val s = performGlobalAction(GLOBAL_ACTION_HOME)
        if (s) Thread.sleep(300) // TODO find optimal value
        else Log.e(TAG, "GLOBAL_ACTION_HOME failed")
        return s
    }

    override fun goBack(): Boolean {
        Log.i(TAG, "Performing global action: BACK")
        val success = performGlobalAction(GLOBAL_ACTION_BACK)
        if (success) {
            try {
                Thread.sleep(300) // TODO find optimal value
            } catch (_: InterruptedException) {
            }
        } else {
            Log.e(TAG, "GLOBAL_ACTION_BACK failed")
        }
        return success
    }

    override fun returnToApp(): Boolean {
        val p = packageName
        return try {
            val i = packageManager.getLaunchIntentForPackage(p)
                ?: return false
            i.addFlags(Intent.FLAG_ACTIVITY_REORDER_TO_FRONT or Intent.FLAG_ACTIVITY_NEW_TASK)
            startActivity(i)
            Thread.sleep(300)
            true
        } catch (e: Exception) {
            Log.e(TAG, "returnToApp failed $p", e)
            false
        }
    }

    // TODO find optimum and move to const val
    override fun waitForIdle() {
        val t = 500L
        Log.d(TAG, "Wait $t ms")
        Thread.sleep(t)
    }
}