package com.example.kt

import android.content.Context
import android.graphics.BitmapFactory
import android.util.Log
import android.view.Surface
import androidx.camera.core.ImageProxy
import androidx.camera.view.PreviewView
import org.opencv.core.*
import org.opencv.core.Core.meanStdDev
import org.opencv.imgproc.Imgproc
import org.opencv.android.Utils
import org.opencv.imgproc.Imgproc.*
import java.lang.Math.pow
import java.util.*

enum class ExposureResult {
    UNDER_EXPOSED, NORMAL, OVER_EXPOSED
}

class ImageCheck() {

    // Overall image quality thresholds
    val SHARPNESS_GAUSSIAN_BLUR_WINDOW = 5
    var SHARPNESS_THRESHOLD = 0.8
    var UNDER_EXPOSURE_THRESHOLD = 128.0
    var OVER_EXPOSURE_THRESHOLD = 200.0
    var OVER_EXPOSURE_WHITE_COUNT = 0.2
    var GLARE_WHITE_VALUE = 235
    var GLARE_WHITE_RATIO = 0.00

    fun getMatFromImage(image: ImageProxy): Mat {
        /* https://stackoverflow.com/questions/30510928/convert-android-camera2-api-yuv-420-888-to-rgb */
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer[nv21, 0, ySize]
        vBuffer[nv21, ySize, vSize]
        uBuffer[nv21, ySize + vSize, uSize]
        val yuv = Mat(image.height + image.height / 2, image.width, CvType.CV_8UC1)
        yuv.put(0, 0, nv21)
        val mat = Mat()
        Imgproc.cvtColor(yuv, mat, Imgproc.COLOR_YUV2RGB_NV21, 3)
        return mat
    }

    fun fixMatRotation(matOrg: Mat, myPreviewView: PreviewView?): Mat {
        val mat: Mat
        when (myPreviewView?.getDisplay()?.getRotation()) {
            Surface.ROTATION_0 -> {
                mat = Mat(matOrg.cols(), matOrg.rows(), matOrg.type())
                Core.transpose(matOrg, mat)
                Core.flip(mat, mat, 1)
            }
            Surface.ROTATION_90 -> mat = matOrg
            Surface.ROTATION_270 -> {
                mat = matOrg
                Core.flip(mat, mat, -1)
            }
            else -> {
                mat = Mat(matOrg.cols(), matOrg.rows(), matOrg.type())
                Core.transpose(matOrg, mat)
                Core.flip(mat, mat, 1)
            }
        }
        return mat
    }

    fun sharpen(image: Mat) : Mat {
        val kernel: Mat = object : Mat(3, 3, CvType.CV_32F) {
            init {
                put(0, 0, -1.0)
                put(0, 1, -1.0)
                put(0, 2, -1.0)
                put(1, 0, -1.0)
                put(1, 1, 9.0)
                put(1, 2, -1.0)
                put(2, 0, -1.0)
                put(2, 1, -1.0)
                put(2, 2, -1.0)
            }
        }

        Imgproc.filter2D(image, image, -1, kernel)
        return image
    }

    fun autoCanny(image: Mat, sigma: Double): Mat {
        Imgproc.cvtColor(image, image, Imgproc.COLOR_RGB2GRAY)
        //  compute the median of the single channel pixel intensities
        var mean = MatOfDouble()
        var stddev = MatOfDouble()
        meanStdDev(image, mean, stddev)
        // apply automatic Canny edge detection using the computed median
        val lower = maxOf(0.0, (1.0 - sigma) * mean!!.get(0, 0)[0])
        val upper = minOf(255.0, (1.0 + sigma) * mean!!.get(0, 0)[0])

        val edged: Mat = image.clone()
        //Canny(edged, edged, lower, upper)
        Canny(edged, edged, 50.0, 100.0)
        return edged
    }

    fun detectCircles(image: Mat, dp: Double, minDist: Double, minR: Int, maxR: Int) : Array<Float> {
        // get edge and dilate
        //val edge = autoCanny(image.clone(), 0.33) // getting edge image
        //val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        //Imgproc.dilate(edge, edge, kernel) // dilate
        // convert to grayscale

        var gray = image.clone()
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_RGB2GRAY)
//        val reducedNoise = gray.clone()
//        GaussianBlur(gray, reducedNoise,  Size(9.0, 9.0), 2.0, 2.0)
        // detecting circles
        val circles = Mat()
        //Imgproc.HoughCircles(gray, circles, Imgproc.CV_HOUGH_GRADIENT, dp, minDist, 85.0, 100.0, minR, maxR)
        Imgproc.HoughCircles(gray, circles, Imgproc.CV_HOUGH_GRADIENT, dp, minDist, 100.0, 100.0, minR, maxR)

        //Log.w("DETECT_CIRCLES", circles.cols().toString() + "")
        var xx = -1.0F; var yy = -1.0F; var rad = -1.0F
        var counts = 0
        for (x in 0 until circles.cols()) {
            val vCircle = circles[0, x]
            val center = Point(Math.round(vCircle[0]).toDouble(), Math.round(vCircle[1]).toDouble())
            val radius = Math.round(vCircle[2]).toInt()
            // get radius and center of circle
            rad = radius.toFloat(); xx = vCircle[0].toFloat(); yy = vCircle[1].toFloat()
            // draw the circle center
            //Core.circle(image, center, 5, Scalar(0.0, 255.0, 0.0), -1, 8, 0)
            // draw the circle outline
            //Core.circle(image, center, radius, Scalar(255.0, 0.0, 255.0), 3, 8, 0)
            counts += 1
            break // break after a single circle is detected
        }
        //Log.e("DETECT_CIRCLES", "COUNTS " + counts)
        // garbage collection
        //edge.release()
        //kernel.release()
        //gray.release()
        //image.release()

        return arrayOf<Float>(xx, yy, rad)
    }


    // The below code has been referenced from: https://github.com/cjpark87/rdt-scan

    /*
       * Calculates the brightness histogram of the candidate video frame
       * @param inputMat: the candidate video frame (in grayscale)
       * @return a 256-element histogram that quantifies the number of pixels at each
       * brightness level for the inputMat
       */
   fun measureExposure(inputMat: Mat): FloatArray {
        // Setup the histogram calculation
        val mHistSizeNum = 256
        val mHistSize = MatOfInt(mHistSizeNum)
        val hist = Mat()
        val mBuff = FloatArray(mHistSizeNum)
        val mBuff2 = FloatArray(mHistSizeNum)
        val histogramRanges = MatOfFloat(0f, 256f)
        val mChannels = arrayOf(MatOfInt(0))
        val sizeRgba = inputMat.size()

        // Calculate the grayscale histogram
        calcHist(Arrays.asList(inputMat), mChannels[0], Mat(), hist,
                mHistSize, histogramRanges)
        hist[0, 0, mBuff2]
        Core.divide(hist, Scalar(sizeRgba.area()), hist)
        hist[0, 0, mBuff]
        mChannels[0].release()

        // Garbage collection
        mHistSize.release()
        histogramRanges.release()
        hist.release()
        return mBuff
    }

    /**
     * Determines whether the candidate video frame has sufficient lighting without being too bright
     * @param inputMat: the candidate video frame (in grayscale)
     * @return ExposureResult enum for whether the candidate video frame has a reasonable brightness
     */
    fun checkExposure(inputMat: Mat): ExposureResult? {
        // Calculate the brightness histogram
        val histograms = measureExposure(inputMat)

        // Identify the highest brightness level in the histogram
        // and the amount at the highest brightness
        var maxWhite = 0
        var whiteCount = 0f
        for (i in histograms.indices) {
            if (histograms[i] > 0) maxWhite = i
            if (i == histograms.size - 1) whiteCount = histograms[i]
        }

        // Assess the brightness relative to thresholds
        return if (maxWhite >= OVER_EXPOSURE_THRESHOLD && whiteCount > OVER_EXPOSURE_WHITE_COUNT) {
            ExposureResult.OVER_EXPOSED
        } else if (maxWhite < UNDER_EXPOSURE_THRESHOLD) {
            ExposureResult.UNDER_EXPOSED
        } else {
            ExposureResult.NORMAL
        }
    }

    /**
     * Calculates the Laplacian variance of the candidate video frame as a metric for sharpness
     * @param inputMat: the candidate video frame (in grayscale)
     * @return the Laplacian variance of the candidate video frame
     */
   fun measureSharpness(inputMat: Mat): Double {
        // Calculate the Laplacian
        val des = Mat()
        Laplacian(inputMat, des, CvType.CV_64F)

        // Calculate the mean and std
        val mean = MatOfDouble()
        val std = MatOfDouble()
        meanStdDev(des, mean, std)

        // Calculate variance
        val sharpness: Double = pow(std[0, 0][0], 2.0)

        // Garbage collection
        des.release()
        return sharpness
    }
    fun getRefImageSharpness(context: Context?): Array<Double>{
        val refImageBitmap = BitmapFactory.decodeResource(context?.getResources(),R.drawable.reference_img)
        var refImg = Mat()
        Utils.bitmapToMat(refImageBitmap, refImg)
        val refImgWidth = refImg.cols().toDouble()
        cvtColor(refImg, refImg, COLOR_RGB2GRAY)
        val refImgSharpness = measureSharpness(refImg)

        //Log.e("TEST", "REF_IMAGE_DIMS "+refImg.cols()+","+refImg.rows())

        // gargabe collection
        refImg.release()
        return arrayOf<Double>(refImgSharpness, refImgWidth)
    }
    /**
     * Determines whether the candidate video frame is focused
     * @param inputMat: the candidate video frame (in grayscale)
     * @return whether the candidate video frame has a reasonable sharpness
     */
    fun checkSharpness(inputMat: Mat,  refImgSharpness: Double = 0.96): Boolean {
        // Resize the image to the scale of the reference
        //val resized = Mat()
        //val scale: Double = refImgWidth / inputMat.cols()
        //resize(inputMat, resized, Size(inputMat.size().width * scale, inputMat.size().height * scale))

        // Calculate sharpness and assess relative to thresholds
        val sharpness = measureSharpness(inputMat)
        val isSharp: Boolean = sharpness > refImgSharpness * SHARPNESS_THRESHOLD

        //Log.e("TEST", "FRAME_SHARPNESS "+sharpness)

        // Garbage collection
        inputMat.release()
        //resized.release()
        return isSharp
    }

    /**
     * Determines if there is glare within the detected RDT's result window (often due to
     * protective covering of the immunoassay)
     * @param inputMat: the candidate video frame (in grayscale)
     * @param boundary: the corners of the bounding box around the detected RDT
     * @return whether there is glare within the detected RDT's result window
     */
    fun checkGlare(inputMat: Mat): Boolean {
        // Convert the image to HLS
        val hls = Mat()
        cvtColor(inputMat, hls, COLOR_BGR2HLS)

        // Calculate brightness histogram across L channel
        val channels: ArrayList<Mat> = ArrayList()
        Core.split(hls, channels)
        val histograms = measureExposure(channels[1])

        // Identify the highest brightness level in the histogram
        // and the amount at the highest brightness
        var maxWhite = 0
        var clippingCount = 0f
        for (i in histograms.indices) {
            if (histograms[i] > 0) maxWhite = i
            if (i == histograms.size - 1) clippingCount = histograms[i]
        }
        //Log.d(TAG, String.format("maxWhite: %d, clippingCount: %.20f", maxWhite, clippingCount))

        // Assess glare relative to thresholds
        return maxWhite >= GLARE_WHITE_VALUE || clippingCount > GLARE_WHITE_RATIO
    }

    /**
     * Crops out the detected RDT's result window as a rectangle
     * @param inputMat: the candidate video frame (in grayscale)
     * @param boundary: the corners of the bounding box around the detected RDT
     * @return the RDT image tightly cropped and de-skewed around the result window
     */
    fun cropResultWindow(inputMat: Mat, cropDims: Int, center: Point): Mat? {
        //Log.e("TAG", "CROP_DIMS " + cropDims + " " + center.x + "x" + center.y)
        if (center.x <= cropDims/2 || center.y <= cropDims/2){
            center.x = inputMat.cols().toDouble()/2; center.y = inputMat.rows().toDouble()/2
        }
        //Log.e("TAG", "CROP_DIMS_AFTER " + cropDims + " " + center.x + "x" + center.y)
        val rectCrop = Rect(center.x.toInt() - cropDims / 2, center.y.toInt() - cropDims / 2,
                cropDims, cropDims)
        val image_output = inputMat.submat(rectCrop)
        //Log.e("TAG", "CROP_DIMS " + cropDims + " " + center.x + "x" + center.y)
        return image_output
    }

}