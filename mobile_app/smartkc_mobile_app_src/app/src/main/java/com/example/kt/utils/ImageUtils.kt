package com.example.kt.utils

import com.example.kt.ExposureResult
import com.example.kt.ImageCheck
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc
import kotlin.math.sqrt

class ImageUtils(var image: Mat) {

    init {
        // Operate on copy of the image
        image = image.clone()
    }

    val imageChecker = ImageCheck()

    /**
     * Function detects cross hair, the returned coordinates can be used to plot it on the preview:
     * Return values contain a pair of (-1, -1) if the function fails to detect a cross hair within the central region
     */
    fun detectCrossHair(dp: Double, minDist: Double, minR: Int, maxR: Int) : Array<Float> {

        // output points
        var output_points = arrayOf<Float>(-1.0F, -1.0F, -1.0F)
        // sharpen image
        val image_sharpened = imageChecker.sharpen(image) // sharpen image
        //Imgproc.medianBlur(image_sharpened, image_sharpened, 5)
        // detect circles
        val detected_circles = imageChecker.detectCircles(image_sharpened, dp, minDist, minR, maxR)
        val xx = detected_circles[0]; val yy = detected_circles[1]; val rad = detected_circles[2]; // get circle coordinates

        // only consider if circle is in central region
        val image_xx = image.cols().toFloat() / 2; val image_yy = image.rows().toFloat() / 2
        if(xx > image_xx-50  && xx < image_xx+50 && yy > image_yy - 50 && yy < image_yy + 50) {
            output_points[0] = xx; output_points[1] = yy // get output points
            //populate radius
            output_points[2] = rad
            return output_points

        }
        return output_points
    }

    fun detectMireCenter(dp: Double, minDist: Double, minR: Int, maxR: Int, jump: Int) :Array<Float>{
        val output_points = arrayOf<Float>(-1.0F, -1.0F) // output points
        var xx = 0F; var yy = 0F; var rCount = 0
        for(currRadius in minR..maxR step jump){
            val detected_circles = imageChecker.detectCircles(image.clone(), dp, minDist, currRadius, currRadius + jump)
            if (detected_circles[2] > 0) {
                xx += detected_circles[0]; yy += detected_circles[1]
                rCount += 1
            }
        }
        xx /= rCount; yy /= rCount
        //val image = image_checker.autoCanny(image, 0.33)
        //Core.circle(image, Point(Math.round(xx).toDouble(), Math.round(yy).toDouble()), 10,
        //        Scalar(255.0, 255.0, 255.0), -1, 8, 0)

        output_points[0] = xx; output_points[1] = yy // get output points
        return output_points
    }

    /**
     * Check the image for quality. Returns the result as a Triple where:
     * 1. First Element: Result for center quality check
     * 2. Second Element: Result for exposure quality check
     * 3. Third Element: Result for sharpness quality check
     */
    fun qualityCheck(crosshair_center: Array<Float>, mire_center: Array<Float>, normFactor: Double, correct_cutoff: Double, zoom_factor: Double): Triple<Boolean, ExposureResult?, Boolean> {
        // get crop_center
        val crop_center = Point(crosshair_center[0].toDouble(), crosshair_center[1].toDouble())

        // check 1: if crosshair_center and mire center are within a threshold
        val check1 = sqrt((crosshair_center[0] - mire_center[0]) * (crosshair_center[0] - mire_center[0])
                + (crosshair_center[1] - mire_center[1]) * (crosshair_center[1] - mire_center[1])) <= (correct_cutoff*zoom_factor/normFactor)

        // check 2: image is neither over-exposed or under-exposed
        var gray = image.clone()
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_RGB2GRAY) // converting image to grayscale
        gray = imageChecker.cropResultWindow(gray, (500 * zoom_factor / normFactor).toInt(), crop_center)
        val exposure = imageChecker.checkExposure(gray.clone())

        // check 3: image is sharp and non blurred
        val check3 = imageChecker.checkSharpness(gray)

        return Triple(check1, exposure, check3)

    }

}