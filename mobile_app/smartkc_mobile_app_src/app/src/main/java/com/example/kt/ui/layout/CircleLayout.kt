package com.example.kt.ui.layout

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat

class CircleLayout constructor(context: Context?, attributeSet: AttributeSet?) :
    View(context, attributeSet) {

    private val circle: MutableList<MutableList<Float>> = mutableListOf()
    private val paint = Paint().apply {
        style = Paint.Style.STROKE
        color = ContextCompat.getColor(context!!, android.R.color.holo_blue_light)
        strokeWidth = 5f
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        // Pass it a list of Float (crossHairs)
        circle.forEach{canvas.drawCircle(it[0], it[1], it[2], paint)}
    }

    fun drawCircle(crossHair: MutableList<MutableList<Float>>) {
        this.circle.clear()
        this.circle.addAll(crossHair)
        invalidate()
    }
}