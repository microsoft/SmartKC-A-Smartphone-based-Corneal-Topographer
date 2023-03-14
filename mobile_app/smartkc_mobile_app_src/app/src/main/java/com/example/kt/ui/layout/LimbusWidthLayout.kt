package com.example.kt.ui.layout

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat

class LimbusWidthLayout constructor(context: Context?, attributeSet: AttributeSet?) :
    View(context, attributeSet) {

    private val limbusWidth: MutableList<MutableList<Float>> = mutableListOf()
    private val paint = Paint().apply {
        style = Paint.Style.STROKE
        color = ContextCompat.getColor(context!!, android.R.color.holo_orange_light)
        strokeWidth = 5f
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        // Pass it a list of Float (crossHairs)
        limbusWidth.forEach{ canvas.drawLines(it.toFloatArray(), paint) }
    }

    fun drawLimbus(limbusPoints: MutableList<MutableList<Float>>) {
        this.limbusWidth.clear()
        this.limbusWidth.addAll(limbusPoints)
        invalidate()
    }
}