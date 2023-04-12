package com.example.kt.utils

import android.graphics.RectF
import com.example.kt.ui.layout.CircleLayout
import com.example.kt.ui.layout.CrossHairLayout
import com.example.kt.ui.layout.RectOverlay
import org.opencv.core.Mat
import kotlin.math.sqrt

class DrawUtils {
    companion object {
        fun plotCrossHair(ch: CrossHairLayout, x: Float, y: Float, d: Float){
            val crossHairBegin = mutableListOf(mutableListOf(x - d, y, x + d, y),
                mutableListOf(x, y - d, x, y + d))
            ch.post{ch.drawCrossHair(crossHairBegin)}
        }

        fun plotRect(ro: RectOverlay, x: Float, y: Float, d: Float){
            val focusRects = listOf(RectF(x - d, y - d, x + d, y + d))
            ro.post { ro.drawRectBounds(focusRects) }
        }

        fun plotCircle(cl: CircleLayout, x: Float, y: Float, r: Float){
            val circle = mutableListOf(mutableListOf(x, y, r))
            cl.post { cl.drawCircle(circle) }
        }
    }
}