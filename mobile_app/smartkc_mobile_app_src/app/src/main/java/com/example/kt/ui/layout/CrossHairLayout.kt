package com.example.kt.ui.layout

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat

class CrossHairLayout constructor(context: Context?, attributeSet: AttributeSet?) :
    View(context, attributeSet) {

    private val crossHair: MutableList<MutableList<Float>> = mutableListOf()
    private val paint = Paint().apply {
        style = Paint.Style.STROKE
        color = ContextCompat.getColor(context!!, android.R.color.holo_red_light)
        strokeWidth = 5f
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        // Pass it a list of Float (crossHairs)
        crossHair.forEach{ canvas.drawLines(it.toFloatArray(), paint) }
    }

    fun drawCrossHair(crossHair: MutableList<MutableList<Float>>) {
        this.crossHair.clear()
        this.crossHair.addAll(crossHair)
        invalidate()
    }
}