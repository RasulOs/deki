package com.example.deki_automata.domain.usecase

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.RectF
import android.util.Base64
import android.util.Log
import com.example.deki_automata.data.model.ActionResponse
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.google.mediapipe.tasks.genai.llminference.GraphOptions
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.common.FileUtil
import java.lang.Exception
import kotlin.system.measureTimeMillis
import com.google.mediapipe.framework.image.BitmapImageBuilder
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.File
import java.util.PriorityQueue
import kotlin.math.max
import kotlin.math.min
import androidx.core.graphics.scale
import com.example.deki_automata.domain.model.DetectionResult
import com.example.deki_automata.util.ImageDebugUtils

class LocalCommandGenerator(private val context: Context) : CommandGenerator {
    private companion object {
        const val TAG = "LocalCommandGenerator"
        const val YOLO_MODEL_FILE = "best_float32.tflite"
//        const val GEMMA_TASK_FILE = "gemma-3n-2b-it-int4.task"
        const val GEMMA_TASK_FILE = "gemma-3n-4b-it-int4.task"
        const val YOLO_INPUT_SIZE = 640
        const val CONFIDENCE_THRESHOLD = 0.3f
        const val MAX_IMAGE_DIMENSION = 640
        const val IOU_THRESHOLD = 0.5f
        const val MAX_YOLO_RESULTS = 50
    }

    private val labels = listOf("View", "ImageView", "Text", "Line")

    private val llmInference: LlmInference by lazy {
        Log.d(TAG, "Initializing LlmInference Engine...")
        val modelFile = File(context.cacheDir, GEMMA_TASK_FILE)
        val modelPath = modelFile.absolutePath
        val options = LlmInference.LlmInferenceOptions.builder()
            .setModelPath(modelPath)
            .setMaxNumImages(1)
            .setMaxTokens(4096)
            .setPreferredBackend(LlmInference.Backend.GPU)
            .build()
        LlmInference.createFromOptions(context, options)
    }

    private val yoloInterpreter: Interpreter by lazy {
        Log.d(TAG, "Initializing TFLite Interpreter for YOLO...")
        val modelBuffer = FileUtil.loadMappedFile(context, YOLO_MODEL_FILE)
        val options = Interpreter.Options()
        Interpreter(modelBuffer, options)
    }

    private val ocrClient by lazy {
        Log.d(TAG, "Initializing OCR Client...")
        TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    }

    override suspend fun getNextAction(
        prompt: String,
        screenshotBase64: String,
        history: List<String>,
    ): Result<ActionResponse> = withContext(Dispatchers.IO) {
        try {
            val imageBytes = Base64.decode(screenshotBase64, Base64.DEFAULT)
            val originalBitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

            val detectedObjects = runYoloInference(originalBitmap)
            Log.d(TAG, "YOLO Interpreter (final) detected ${detectedObjects.size} objects.")

            val markedUpBitmap = ImageDebugUtils.drawDetectionsWithIds(originalBitmap, detectedObjects)

            // To save the image with YOLO bounding boxes uncomment the code
//            val timestamp = System.currentTimeMillis()
//            ImageDebugUtils.saveBitmapToPictures(context, markedUpBitmap, "yolo_final_output_$timestamp.png")
//            Log.d(TAG, "Saved FINAL debug image to Pictures/yolo_final_output_$timestamp.png")

            val imageDescriptionJson = buildJsonScreenDescription(detectedObjects, originalBitmap)

            val gemmaPrompt = buildActionPrompt(prompt, imageDescriptionJson, history)
            Log.d(TAG, "\n\ngemmaPrompt: $gemmaPrompt \n\n")

            var gemmaResponse = ""
            val llmTime = measureTimeMillis {
                val sessionOptions = LlmInferenceSession.LlmInferenceSessionOptions.builder()
//                    .setTemperature(0.2f)
                    .setTopK(64)
                    .setTopP(0.95f)
                    .setGraphOptions(
                        GraphOptions.builder().setEnableVisionModality(true).build()
                    )
                    .build()

                LlmInferenceSession.createFromOptions(llmInference, sessionOptions).use { session ->
                    val mpImage = BitmapImageBuilder(markedUpBitmap).build()
                    session.addQueryChunk(gemmaPrompt)
                    session.addImage(mpImage)
                    gemmaResponse = session.generateResponse()
                }

                gemmaResponse = gemmaResponse.substringAfter("```json").substringBefore("```").trim()
            }
            Log.d(TAG, "Gemma multimodal inference took $llmTime ms. Response: $gemmaResponse")

            val newHistory = history + gemmaResponse
            val response = ActionResponse(response = gemmaResponse, history = newHistory)
            Result.success(response)

        } catch (e: Exception) {
            Log.e(TAG, "Local command generation failed spectacularly", e)
            Result.failure(e)
        }
    }

    private fun runYoloInference(originalBitmap: Bitmap): List<DetectionResult> {
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0.0f, 255.0f))
            .build()

        var tensorImage = TensorImage.fromBitmap(originalBitmap)
        tensorImage = imageProcessor.process(tensorImage)

        val outputBuffer = Array(1) { Array(300) { FloatArray(6) } }

        yoloInterpreter.run(tensorImage.buffer, outputBuffer)

        val allDetections = mutableListOf<DetectionResult>()
        val detections = outputBuffer[0]

        for (detection in detections) {
            val confidence = detection[4]

            if (confidence >= CONFIDENCE_THRESHOLD) {
                val classId = detection[5].toInt()
                val label = labels.getOrElse(classId) { "Unknown" }

                val x1 = detection[0] * originalBitmap.width
                val y1 = detection[1] * originalBitmap.height
                val x2 = detection[2] * originalBitmap.width
                val y2 = detection[3] * originalBitmap.height

                val rect = RectF(x1, y1, x2, y2)
                allDetections.add(DetectionResult(rect, label, confidence))
            }
        }

        return performNms(allDetections)
    }

    private fun performNms(detections: List<DetectionResult>): List<DetectionResult> {
        val detectionsByClass = detections.groupBy { it.label }
        val finalDetections = mutableListOf<DetectionResult>()

        detectionsByClass.forEach { (_, classDetections) ->
            val pq = PriorityQueue<DetectionResult>(classDetections.size, compareByDescending { it.confidence })
            pq.addAll(classDetections)

            while (pq.isNotEmpty()) {
                val bestDetection = pq.poll() ?: continue
                finalDetections.add(bestDetection)

                val remainingDetections = pq.toList()
                pq.clear()

                for (detection in remainingDetections) {
                    val iou = calculateIoU(bestDetection.boundingBox, detection.boundingBox)
                    if (iou < IOU_THRESHOLD)
                        pq.add(detection)
                }
            }
        }

        finalDetections.sortByDescending { it.confidence }

        val limitedDetections = if (finalDetections.size > MAX_YOLO_RESULTS)
            finalDetections.subList(0, MAX_YOLO_RESULTS)
        else
            finalDetections

        limitedDetections.sortBy { it.boundingBox.top }

        return limitedDetections
    }

    private fun calculateIoU(box1: RectF, box2: RectF): Float {
        val xA = max(box1.left, box2.left)
        val yA = max(box1.top, box2.top)
        val xB = min(box1.right, box2.right)
        val yB = min(box1.bottom, box2.bottom)

        val intersectionArea = max(0f, xB - xA) * max(0f, yB - yA)
        val box1Area = box1.width() * box1.height()
        val box2Area = box2.width() * box2.height()
        val unionArea = box1Area + box2Area - intersectionArea

        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }

    private suspend fun buildJsonScreenDescription(
        detections: List<DetectionResult>,
        fullBitmap: Bitmap,
    ): String {
        val root = JSONObject()
        root.put("image_size", "[${fullBitmap.width},${fullBitmap.height}]")
        root.put("bbox_format", "center_x, center_y, width, height")
        val elementsArray = JSONArray()

        detections.forEachIndexed { index, detection ->
            val element = JSONObject()
            val box = detection.boundingBox
            val category = detection.label.lowercase().replace("imageview", "image")

            val width = box.width().toInt()
            val height = box.height().toInt()
            val centerX = box.centerX().toInt()
            val centerY = box.centerY().toInt()

            element.put("id", "${category}_${index + 1}")
            element.put("bbox", "[$centerX,$centerY,$width,$height]")

            if (category == "text") {
                try {
                    val safeLeft = max(0f, box.left).toInt()
                    val safeTop = max(0f, box.top).toInt()
                    val safeWidth = min(fullBitmap.width - safeLeft, box.width().toInt())
                    val safeHeight = min(fullBitmap.height - safeTop, box.height().toInt())

                    if (safeWidth > 0 && safeHeight > 0) {
                        var croppedBmp = Bitmap.createBitmap(fullBitmap, safeLeft, safeTop, safeWidth, safeHeight)

                        // Check if the cropped bitmap is smaller than ML Kit's requirement (32)
                        if (croppedBmp.width < 32 || croppedBmp.height < 32) {
                            val scaleFactor = 4
                            val newWidth = croppedBmp.width * scaleFactor
                            val newHeight = croppedBmp.height * scaleFactor
                            croppedBmp = croppedBmp.scale(newWidth, newHeight)
                        }

                        val inputImage = InputImage.fromBitmap(croppedBmp, 0)
                        val ocrResult = Tasks.await(ocrClient.process(inputImage))
                        val recognizedText = ocrResult.text.replace("\n", " ").trim()
                        if (recognizedText.isNotEmpty()) element.put("text", recognizedText)

                    }
                } catch (e: Exception) {
                    Log.e(TAG, "OCR failed for element $index", e)
                }
            }
            elementsArray.put(element)
        }
        root.put("elements", elementsArray)
        return root.toString(2)
    }

    private fun buildActionPrompt(
        userPrompt: String,
        imageDescription: String,
        history: List<String>
    ): String {
        val currentStep = history.size + 1
        val previousStepsText = if (history.isEmpty()) {
            "This is the first step."
        } else {
            "Previous Actions and Reasoning:\n" + history.mapIndexed { index, step ->
                "- Step ${index + 1}: $step"
            }.joinToString("\n")
        }

        return """
        You are an AI agent controlling a mobile device.     
        Your task is to analyze the provided context and screen to output a single action in a specific JSON format.
        Do not add any text or explanations outside of the JSON object.
        
        <CONTEXT>
            <USER_GOAL>
                $userPrompt
            </USER_GOAL>
            <CURRENT_STEP>
                $currentStep
            </CURRENT_STEP>
            <HISTORY>
                $previousStepsText
            </HISTORY>
        </CONTEXT>

        <SCREEN_DESCRIPTION>
            $imageDescription
        </SCREEN_DESCRIPTION>

        <AVAILABLE_ACTIONS>
            1. "Swipe left. From start coordinates 300, 400" (or other coordinates)
            2. "Swipe right. From start coordinates 500, 650" (or other coordinates)
            3. "Swipe top. From start coordinates 600, 510" (or other coordinates)
            4. "Swipe bottom. From start coordinates 640, 500" (or other coordinates)
            5. "Go home"
            6. "Go back"
            7. "Open com.linkedin.android" (or other package name)
            9. "Tap coordinates 160, 820" (or other coordinates)
            10. "Insert text 210, 820:Hello world" (or other coordinates and text)
            11. "Screen is in a loading state. Try again" (send image again)
            12. "Answer: There are no new important mails today" (or other answer)
            13. "Finished" (task is finished)
            14. "Can't proceed" (can't understand what to do or image has problem etc.)
        </AVAILABLE_ACTIONS>

        <INSTRUCTIONS>
        Your response MUST be a single, valid JSON object with two keys: "reason" and "action".
        For the "reason", first observe the screen, then state the user's goal, then formulate a plan.
        In the first screen you will see 'Send' button, you don't need to use it, it is already clicked.
        </INSTRUCTIONS>
        
        <EXAMPLE>
            <USER_GOAL>Sign up in the application</USER_GOAL>
            <SCREEN_DESCRIPTION>
            { "elements": [ 
                    { 
                        "id": "text_21", 
                        "bbox": "[540,1850,200,50]", 
                        "text": "Continue" 
                    } 
                ] 
            }
            </SCREEN_DESCRIPTION>
            <RESPONSE>
            {
              "reason": "Observation: The screen has a 'Continue' button.\nGoal: The user wants to sign up.\nPlan: I will tap the 'Continue' button.",
              "action": "Tap coordinates 540, 1850"
            }
            </RESPONSE>
        </EXAMPLE>
        <EXAMPLE>
            <USER_GOAL>Open whatsapp and finish</USER_GOAL>
            <SCREEN_DESCRIPTION>
            { "elements": [ 
                    { 
                        "id": "text_10", 
                        "bbox": "[340,450,140,40]", 
                        "text": "Send" 
                    } 
                ] 
            }
            </SCREEN_DESCRIPTION>
            <RESPONSE>
            {
              "reason": "Observation: The screen has a 'Send' button.\nGoal: The user wants to open whatsapp and finish the task.\nPlan: I will return command 'Open com.whatsapp' and in the next step 'Finished'.",
              "action": "Open com.whatsapp"
            }
            </RESPONSE>
        </EXAMPLE>
        ```json""".trimIndent() // ```json is added to force the LLM to output JSON format from this point on
    }
}
