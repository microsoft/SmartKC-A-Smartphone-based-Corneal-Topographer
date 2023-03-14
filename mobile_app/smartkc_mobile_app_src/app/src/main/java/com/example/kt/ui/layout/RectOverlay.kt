package com.example.kt.ui.layout

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat

class RectOverlay constructor(context: Context?, attributeSet: AttributeSet?) :
    View(context, attributeSet) {

    private val rectBounds: MutableList<RectF> = mutableListOf()
    private var paint = Paint().apply {
        style = Paint.Style.STROKE
        color = ContextCompat.getColor(context!!, android.R.color.holo_green_light)
        strokeWidth = 5f
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        // Pass it a list of RectF (rectBounds)
        rectBounds.forEach { canvas.drawRect(it, paint) }
    }

    fun drawRectBounds(rectBounds: List<RectF>, flag: Int = 0) {
        // flag: 0 for default, flag: 1 for green big, flag: 2 for red big
        if (flag == 1){
            this.paint = Paint().apply {
                style = Paint.Style.STROKE
                color = ContextCompat.getColor(context!!, android.R.color.holo_green_light)
                strokeWidth = 50f
            }
        }
        else if (flag == 2){
            this.paint = Paint().apply {
                style = Paint.Style.STROKE
                color = ContextCompat.getColor(context!!, android.R.color.holo_red_light)
                strokeWidth = 50f
            }
        }
        this.rectBounds.clear()
        this.rectBounds.addAll(rectBounds)
        invalidate()
    }
}