package com.example.deki_automata.util

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Environment
import com.example.deki_automata.domain.model.DetectionResult
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream

object ImageDebugUtils {

    fun letterbox(bitmap: Bitmap, targetSize: Int = 640): Bitmap {
        val width = bitmap.width
        val height = bitmap.height

        val scale = if (width > height) targetSize.toFloat() / width
        else targetSize.toFloat() / height

        val newWidth = (width * scale).toInt()
        val newHeight = (height * scale).toInt()

        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
        val resultBitmap = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(resultBitmap)
        canvas.drawColor(Color.rgb(114, 114, 114))

        val left = (targetSize - newWidth) / 2
        val top = (targetSize - newHeight) / 2

        canvas.drawBitmap(scaledBitmap, left.toFloat(), top.toFloat(), null)
        scaledBitmap.recycle()
        return resultBitmap
    }

    /**
     * Draws bounding boxes and their ID numbers (1, 2, 3 ...) on the bitmap.
     * This is for the "Set of Marks" image sent to the LLM and for debugging.
     * Check README to see examples.
     */
    fun drawDetectionsWithIds(bitmap: Bitmap, detections: List<DetectionResult>): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)

        val boxPaint = Paint().apply {
            color = Color.GREEN
            style = Paint.Style.STROKE
            strokeWidth = 4.0f
        }

        val textPaint = Paint().apply {
            color = Color.rgb(255, 0, 0)
            textSize = 32.0f
            style = Paint.Style.FILL
        }

        detections.forEachIndexed { index, result ->
            canvas.drawRect(result.boundingBox, boxPaint)
            canvas.drawText((index + 1).toString(), result.boundingBox.left, result.boundingBox.top - 5, textPaint)
        }
        return mutableBitmap
    }

    fun saveBitmapToPictures(context: Context, bitmap: Bitmap, filename: String) {
        val imagesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
        val imageFile = File(imagesDir, filename)
        try {
            val fos: OutputStream = FileOutputStream(imageFile)
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos)
            fos.flush()
            fos.close()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}
