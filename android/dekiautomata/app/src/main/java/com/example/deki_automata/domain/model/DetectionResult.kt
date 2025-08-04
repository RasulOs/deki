package com.example.deki_automata.domain.model

import android.graphics.RectF

data class DetectionResult(val boundingBox: RectF, val label: String, val confidence: Float)
