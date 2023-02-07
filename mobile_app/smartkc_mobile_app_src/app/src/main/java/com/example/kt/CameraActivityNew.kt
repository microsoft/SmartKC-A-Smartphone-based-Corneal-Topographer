package com.example.kt

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.*
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.params.StreamConfigurationMap
import android.media.MediaActionSound
import android.net.Uri
import android.os.Bundle
import android.util.AttributeSet
import android.util.Log
import android.util.Size
import android.view.MotionEvent
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.*
import androidx.camera.core.Camera
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.datastore.preferences.core.stringPreferencesKey
import com.example.kt.data.repo.FileRepository
import com.example.kt.injection.dataStore
import com.example.kt.utils.PreferenceKeys
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.android.synthetic.main.activity_cameranew.*
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc
import java.io.File
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import javax.inject.Inject
import kotlin.collections.LinkedHashSet
import kotlin.math.sqrt

@ExperimentalCamera2Interop @ExperimentalCameraFilter @AndroidEntryPoint
class CameraActivityNew : AppCompatActivity() {

    private var imageCapture: ImageCapture? = null
    private lateinit var outputDirectory: File
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var cameraProvider: ProcessCameraProvider
    private lateinit var camera: Camera // this was a local variable in original code
    // File Repository
    @Inject lateinit var fileRepository: FileRepository

    // public class variables
    var base_dir: String? = null
    var dir_name: String? = null
    var left_right: String? = null
    var hash_map : HashMap<*, *>? = null
    var currentCounts = 0 // initialize current counts as 0
    var maxCounts=-1; // initialized in onCreate()
    var idx=0;
    var rectSize = 100
    var focus_timestamp = -1L
    var time_diff = 2 // time difference b/w tap to focus and capture
    var frames_elapsed = 0
    val trigger_rate = 1
    val zoom_factor = 2.0 // how much to zoom the image by
    var trigger_coords : FloatArray = floatArrayOf(0.0F, 0.0F)
    // lock_button flag
    var lock_button_flag: Boolean = true
    // lock auto_capture
    var lock_auto_capture_flag: Boolean = true
    // correct cut-off 25 pixels
    var correct_cutoff = 45.0
    // reference image => lateinit var refImgMetrics: Array<Double>
    var quality_counter = 0
    var quality_threshold = 2
    var capture_timestamp = -1L
    var capture_time_diff = 2 // time difference b/w last capture and current capture
    var limbus_radius = 250
    /*** Views  */
    private var myPreviewView: PreviewView? = null
    //private var myImageView: ImageView? = null

    /*** For CameraX  */
    private var imageAnalysis: ImageAnalysis? = null
    val image_checker = ImageCheck() // create ImageCheck object

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_cameranew)

        // get app preferences
        var sharedPrefs = this.getSharedPreferences("KT_APP_PREFERENCES", Context.MODE_PRIVATE)
        correct_cutoff *= sharedPrefs.getFloat("CENTER_CUTOFF", 0.5F)
        Log.e(TAG, "CAMERA_ACTIVITY_PREFERENCE "+" "+sharedPrefs.getFloat("CENTER_CUTOFF", 0.5F) + " In Pixels "+correct_cutoff);

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                    this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        // Set up the listener for take photo button
        camera_capture_button.setOnClickListener { takePhoto() }
        // setup lock_button listener
        lock_button.setOnClickListener{
            if(lock_button_flag){
                lock_button_flag = false
                lock_button.setText("Unlock Cross")
            }
            else{
                lock_button_flag = true
                lock_button.setText("Lock Cross")
            }

        }
        // auto_capture lock
        lock_capture.setOnClickListener{
            if(lock_auto_capture_flag){
                lock_auto_capture_flag = false
                lock_capture.setText("Unlock Auto-Click")
            }
            else{
                lock_auto_capture_flag = true
                lock_capture.setText("Lock Auto-Click")
            }
        }

        MainActivity.PACKAGE_NAME = getApplicationInfo().loadLabel(getPackageManager()).toString()
        // Fix this (it was not picking the version from the initial page)
        MainActivity.VERSION = "14"
        base_dir = MainActivity.PACKAGE_NAME

        // retrieve getExtras
        val bundle = intent.extras
        dir_name = bundle!!.getString("dir_name")
        left_right = bundle.getString("left_right")
        maxCounts = bundle.getString("number_of_images")?.toInt() ?: 3
        hash_map = getIntent().getSerializableExtra("hash_map") as HashMap<*, *>

        // get current idx
        val dir = File(getExternalFilesDir(null), MainActivity.PACKAGE_NAME + "/" + dir_name)
        if(dir.listFiles { dir, name -> name.toLowerCase().startsWith(left_right!!) } != null)
            idx = dir.listFiles { dir, name -> name.toLowerCase().startsWith(left_right!!) }.size

        outputDirectory = getOutputDirectory()
        cameraExecutor = Executors.newSingleThreadExecutor()

        if (left_right == "right") {
            supportActionBar!!.setTitle("Right Eye: Capture Image " + (currentCounts + 1).toString() + " of " + (maxCounts).toString())
        }
        else {
            supportActionBar!!.setTitle("Left Eye: Capture Image " + (currentCounts + 1).toString() + " of " + (maxCounts).toString())
        }
        myPreviewView = findViewById(R.id.viewFinder)
    }

    private fun takePhoto() {

        // reset zoom
        //camera.cameraControl.setZoomRatio(1.0F)

        if ((System.currentTimeMillis()-capture_timestamp)/1000 < capture_time_diff){
            quality_counter = 0;
            return;
        }

        // if auto-focus not triggered, re-focus at center
        if (focus_timestamp == -1L || (System.currentTimeMillis()-focus_timestamp)/1000 >= time_diff) {
            focusAtCenter(trigger_coords[0], trigger_coords[1])
            // focusAtCenter((viewFinder.width).toFloat() / 2, (viewFinder.height).toFloat() / 2) // older when triggering at center
            //Log.d(TAG, "Time diff in seconds"+((System.currentTimeMillis()-focus_timestamp)/1000))
        }

        // Get a stable reference of the modifiable image capture use case
        val imageCapture = imageCapture ?: return

        // Create indexed output file to hold the image
        val photoFile = File(
                outputDirectory,
                left_right + "_" + (idx + currentCounts) + ".jpg")

        // Create output options object which contains file + metadata
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        // Set up image capture listener, which is triggered after photo has been taken
        imageCapture.takePicture(
                outputOptions, ContextCompat.getMainExecutor(this), object : ImageCapture.OnImageSavedCallback {
            override fun onError(exc: ImageCaptureException) {
                Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
            }

            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                val savedUri = Uri.fromFile(photoFile)
                val msg = "Photo capture succeeded: $savedUri"
                // Record it in database
                runBlocking {
                    val sharedPrefs = getSharedPreferences("KT_APP_PREFERENCES", MODE_PRIVATE)
                    val center_name = sharedPrefs.getString("CENTER_NAME", "")
                    if (center_name.isNullOrEmpty()) {
                        throw Error("Center name cannot be null")
                    }
                    val fileNameParts = savedUri.toString().split("_")
                    val idx = fileNameParts[fileNameParts.size-1]
                    val fileName = "${center_name}/${dir_name?.split("_")?.get(0)}/${left_right}/${idx}"
                    fileRepository.insertNewFileRecord(savedUri.toString(), fileName)
                }
                Toast.makeText(baseContext, "Counts:" + (currentCounts + 1) + "/" + maxCounts, Toast.LENGTH_SHORT).show()
                Log.d(TAG, msg)
                // play capture sound
                val sound = MediaActionSound()
                sound.play(MediaActionSound.SHUTTER_CLICK)

                //increment counts, and move to next activity after image has been saved
                currentCounts += 1
                //Log.e(TAG, "Current counts: " + currentCounts + " maxCounts " + maxCounts)
                if (currentCounts >= maxCounts) {
                    //cameraProvider.unbindAll()
                    val intent = Intent(this@CameraActivityNew, CheckImages::class.java)
                    // add extras to intent
                    intent.putExtra("dir_name", dir_name)
                    intent.putExtra("left_right", left_right)
                    intent.putExtra("number_of_images", "" + maxCounts)
                    intent.putExtra("hash_map", hash_map)
                    //finish current activity
                    finish()
                    //start the second Activity
                    this@CameraActivityNew.startActivity(intent)
                }

                if (left_right == "right") {
                    supportActionBar!!.setTitle("Right Eye: Capture Image " + (currentCounts + 1).toString() + " of " + (maxCounts).toString())
                } else {
                    supportActionBar!!.setTitle("Left Eye: Capture Image " + (currentCounts + 1).toString() + " of " + (maxCounts).toString())
                }

            }
        })

        // reset counters
        quality_counter = 0 // reset quality_counter
        capture_timestamp = System.currentTimeMillis()

    }

    @SuppressLint("ClickableViewAccessibility")
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(Runnable {
            // Used to bind the lifecycle of cameras to the lifecycle owner
            cameraProvider = cameraProviderFuture.get()
            // Preview
            val preview = Preview.Builder()
                    //.setTargetResolution(Size(640, 480))
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    .build()
                    .also {
                        it.setSurfaceProvider(viewFinder.surfaceProvider)
                    }
            // hard coding the resolution to 6000 x 8000
            imageCapture = ImageCapture.Builder()
                    .setTargetResolution(Size(6000, 8000))
                    .build()

            // image analysis
            imageAnalysis = ImageAnalysis.Builder()
                    //.setTargetResolution(Size(240, 320))
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { it.setAnalyzer(cameraExecutor, MyImageAnalyzer()) }

            var cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            // Get camera selector
            runBlocking {
                val data = dataStore.data.first()
                val selectedCameraKey = stringPreferencesKey(PreferenceKeys.CHOSEN_CAMERA)
                val selectedCamera = data[selectedCameraKey]
                Toast.makeText(applicationContext, selectedCamera, Toast.LENGTH_LONG).show()
                if (selectedCamera != null) cameraSelector = CameraSelector.Builder().addCameraFilter {
                    val filtered = it.filter {
                        val cameraId = Camera2CameraInfo.fromCameraInfo(it.cameraInfo).cameraId
                        return@filter cameraId == selectedCamera }
                    val result = LinkedHashSet<Camera>()
                    filtered.forEach { result.add(it) }
                    result
                }
                .build()
            }

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to (val) camera
                camera = cameraProvider.bindToLifecycle(
                        this, cameraSelector, preview, imageCapture, imageAnalysis)

                //touch to focus listener
                viewFinder.setOnTouchListener { _, event ->
                    if (event.action != MotionEvent.ACTION_UP) {
                        return@setOnTouchListener true
                    }

                    // trigger focus on tapped region & draw rectangle
                    focusAtCenter(event.x, event.y)
                    return@setOnTouchListener true
                }

                // set zoom here
                camera.cameraControl.setZoomRatio(zoom_factor.toFloat())
                // focus at center initially
                focusAtCenter((viewFinder.width).toFloat() / 2, (viewFinder.height).toFloat() / 2)
                //plot_rect((viewFinder.width).toFloat() / 2, (viewFinder.height).toFloat() / 2, rectSize.toFloat())


            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }


        }, ContextCompat.getMainExecutor(this))
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
                baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
            requestCode: Int, permissions: Array<String>, grantResults:
            IntArray) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this,
                        "Permissions not granted by the user.",
                        Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }


    private fun getOutputDirectory(): File {
        // ugly way to do this, fix if needed
        var dir = File(getExternalFilesDir(null), base_dir)
        if (!dir.exists()) {
            dir.mkdirs()
            Log.e(TAG, "Output directory Created")
        }
        dir = File(getExternalFilesDir(null), base_dir + '/' + dir_name)
        if (!dir.exists()) {
            dir.mkdirs()
        }
        return dir
    }

    override fun onResume() {
        super.onResume()
        if(this@CameraActivityNew::camera.isInitialized) {
            camera?.cameraControl.setZoomRatio(zoom_factor.toFloat())
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    private fun stopCamera(){
        cameraProvider.unbindAll()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "CameraActivityNew"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    // image analysis class
    inner private class MyImageAnalyzer : ImageAnalysis.Analyzer {
        @SuppressLint("UnsafeExperimentalUsageError")
        override fun analyze(image: ImageProxy) {

            /* Create cv::mat(RGB888) from image(NV21) */
            //val matOrg = getMatFromImage(image)
            val temp_bmp = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
            val converter = YuvToRgbConverter(this@CameraActivityNew)
            converter.yuvToRgb(image.getImage()!!, temp_bmp)
            val matOrg = Mat()
            Utils.bitmapToMat(temp_bmp, matOrg);

            /* Fix image rotation (it looks image in PreviewView is automatically fixed by CameraX???) */
            val matFixed = image_checker.fixMatRotation(matOrg, myPreviewView)

            /* Do some image processing */
            // add multiplying factor & choosing parameters for detecting circles based on 3000x4000 image
            val basewidth = 3000.0
            val baseheight = 4000.0
            val baseminDist = 1200.0*zoom_factor
            val baseminR = 595.0*zoom_factor
            val basemaxR = 615.0*zoom_factor
            val normfactor = sqrt((basewidth * basewidth + baseheight * baseheight) / (image.width * image.width + image.height * image.height))

            // detect the central cross-hair
            var crosshair_center = arrayOf<Float>(trigger_coords[0] / ((viewFinder.width).toFloat() / matFixed.cols().toFloat()),
                    trigger_coords[1] / ((viewFinder.height).toFloat() / matFixed.rows().toFloat()))
            if (frames_elapsed % trigger_rate == 0 && lock_button_flag) {
                // returns un-normalized mire center in the image not preview
                var updated_center = detectCrossHair(matFixed.clone(), 2.5,
                        baseminDist / normfactor,
                        (baseminR / normfactor).toInt() - 2, (basemaxR / normfactor).toInt() + 3)
                if(updated_center[0]!=-1.0f && updated_center[1]!=-1.0f){
                    crosshair_center = updated_center
                }
            }

            //Log.e(TAG, "CHECKER "+frames_elapsed+" "+(baseminR / normfactor).toInt()+" "+(basemaxR / normfactor).toInt() +" "+matFixed.size())
            // detect the mire center
            val start = (30*zoom_factor/normfactor).toInt(); val end = (100*zoom_factor/normfactor).toInt()
            val jump = (10*zoom_factor/normfactor).toInt()
            val mire_center = detectMireCenter(matFixed.clone(), 2.5, baseminDist / normfactor, start, end, jump, (zoom_factor*viewFinder.width/basewidth).toFloat())

            // image quality check
            qualityCheck(matFixed.clone(), crosshair_center, mire_center, normfactor)

            frames_elapsed += 1
            runOnUiThread(Runnable {
                progressBar.setProgress(100 * quality_counter / quality_threshold)
            })
            if(quality_counter >= quality_threshold && (System.currentTimeMillis()-capture_timestamp)/1000 > capture_time_diff && lock_auto_capture_flag){
                runOnUiThread(Runnable {
                    takePhoto()
                })
            }
            else if(quality_counter >= quality_threshold){
                quality_counter = 0
            }
            /*Close the image otherwise, this function is not called next time*/
            matFixed.release()
            image.close()
        }
        private fun detectCrossHair(image: Mat, dp: Double, minDist: Double, minR: Int, maxR: Int) : Array<Float> {

            // output points
            var output_points = arrayOf<Float>(-1.0F, -1.0F)
            // sharpen image
            val image_sharpened = image_checker.sharpen(image) // sharpen image
            //Imgproc.medianBlur(image_sharpened, image_sharpened, 5)
            // detect circles
            val detected_circles = image_checker.detectCircles(image_sharpened, dp, minDist, minR, maxR)
            val xx = detected_circles[0]; val yy = detected_circles[1]; val rad = detected_circles[2]; // get circle coordinates

            // only consider if circle is in central region
            val image_xx = image.cols().toFloat() / 2; val image_yy = image.rows().toFloat() / 2
            if(xx > image_xx-50  && xx < image_xx+50 && yy > image_yy - 50 && yy < image_yy + 50) {
                output_points[0] = xx; output_points[1] = yy // get output points
                trigger_coords[0] = xx * ((viewFinder.width).toFloat() / image.cols().toFloat())
                trigger_coords[1] = yy * ((viewFinder.height).toFloat() / image.rows().toFloat())
                // plot cross_hair on preview
                plot_cross_hair(trigger_coords[0], trigger_coords[1], 50.0F)
                val normfactor = sqrt((viewFinder.width * viewFinder.width + viewFinder.height * viewFinder.height).toDouble()
                        / (image.cols() * image.cols() + image.rows() * image.rows()).toDouble()).toFloat()
                plot_circle(trigger_coords[0], trigger_coords[1], rad * normfactor)

                // trigger focus and draw rectangle
                /*
                runOnUiThread(Runnable {
                    focusAtCenter(trigger_coords[0], trigger_coords[1])
                })
                 */
//                Log.e(TAG, "CROSS_HAIR "+output_points)
                return output_points

            }
            return output_points
        }

        private fun detectMireCenter(image: Mat, dp: Double, minDist: Double, minR: Int, maxR: Int, jump: Int, scale_factor: Float) :Array<Float>{
            var output_points = arrayOf<Float>(-1.0F, -1.0F) // output points
            var xx = 0F; var yy = 0F; var rCount = 0
            for(currRadius in minR..maxR step jump){
                val detected_circles = image_checker.detectCircles(image.clone(), dp, minDist, currRadius, currRadius + jump)
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
            // overlay cross hair
            xx = xx * ((viewFinder.width).toFloat() / image.cols().toFloat())
            yy = yy * ((viewFinder.height).toFloat() / image.rows().toFloat())
            // draw crossHair for mire
            val crossHairBegin = mutableListOf(mutableListOf(xx - 10, yy, xx + 10, yy),
                    mutableListOf(xx, yy - 10, xx, yy + 10))
            cross_hair_mire.post { cross_hair_mire.drawCrossHair(crossHairBegin) }

            // draw limbus pointer
            val limbusPoints = mutableListOf(mutableListOf(xx - limbus_radius*scale_factor, yy, xx + limbus_radius*scale_factor, yy),
                    mutableListOf(xx - limbus_radius*scale_factor, yy - 10, xx-limbus_radius*scale_factor, yy + 10),
                    mutableListOf(xx + limbus_radius*scale_factor, yy - 10, xx+limbus_radius*scale_factor, yy + 10))
            limbus_width.post { limbus_width.drawLimbus(limbusPoints) }

            return output_points
        }

        // takes as input uncropped color image
        // image quality handler function
        fun qualityCheck(image: Mat, crosshair_center: Array<Float>, mire_center: Array<Float>, normFactor: Double){
            // get crop_center
            val crop_center = Point(crosshair_center[0].toDouble(), crosshair_center[1].toDouble())

            // check 1: if crosshair_center and mire center are within a threshold
            val check1 = sqrt((crosshair_center[0] - mire_center[0]) * (crosshair_center[0] - mire_center[0])
                    + (crosshair_center[1] - mire_center[1]) * (crosshair_center[1] - mire_center[1])) <= (correct_cutoff*zoom_factor/normFactor)

            /*
            val dist = sqrt((crosshair_center[0] - mire_center[0]) * (crosshair_center[0] - mire_center[0])
                    + (crosshair_center[1] - mire_center[1]) * (crosshair_center[1] - mire_center[1]))
            val threshold = (correct_cutoff*zoom_factor/normFactor
             */

            if(check1) {
                runOnUiThread(Runnable {
                    text_center.setText("CENTERED: ✓")
                    text_center.setTextColor(Color.parseColor("#8BC34A"))
                })
            }
            else{
                runOnUiThread(Runnable {
                    text_center.setText("CENTERED: X")
                    text_center.setTextColor(Color.parseColor("#F44336"))
                })
            }

            // check 2: image is neither over-exposed or under-exposed
            var gray = image.clone()
            Imgproc.cvtColor(image, gray, Imgproc.COLOR_RGB2GRAY) // converting image to grayscale
            gray = image_checker.cropResultWindow(gray, (500 * zoom_factor / normFactor).toInt(), crop_center)
            val exposure = image_checker.checkExposure(gray.clone())
            val check2 = exposure == ExposureResult.NORMAL

            if(exposure == ExposureResult.OVER_EXPOSED) {
                runOnUiThread(Runnable {
                    text_over_under.setText("EXPOSURE: X")
                    text_over_under.setTextColor(Color.parseColor("#F44336"))
                })
            }
            else if(exposure == ExposureResult.UNDER_EXPOSED) {
                runOnUiThread(Runnable {
                    text_over_under.setText("EXPOSURE: X")
                    text_over_under.setTextColor(Color.parseColor("#F44336"))
                })
            }
            else{
                runOnUiThread(Runnable {
                    text_over_under.setText("EXPOSURE: ✓")
                    text_over_under.setTextColor(Color.parseColor("#8BC34A"))
                })
            }

            // check 3: image is sharp and non blurred
            val check3 = image_checker.checkSharpness(gray)
            //val check3 = true
            if(check3) {
                runOnUiThread(Runnable {
                    text_sharp.setText("SHARPNESS: ✓")
                    text_sharp.setTextColor(Color.parseColor("#8BC34A"))
                })

            }
            else{
                runOnUiThread(Runnable {
                    text_sharp.setText("SHARPNESS: X")
                    text_sharp.setTextColor(Color.parseColor("#F44336"))
                })
            }

            if(check1 && check2 && check3){
                val focusRects = listOf(RectF(0.0F, 0.0F, viewFinder.width.toFloat(), viewFinder.height.toFloat()))
                rect_overlay_correct.post { rect_overlay_correct.drawRectBounds(focusRects, 1) }
                quality_counter += 1
            }
            else{
                val focusRects = listOf(RectF(0.0F, 0.0F, viewFinder.width.toFloat(), viewFinder.height.toFloat()))
                rect_overlay_correct.post { rect_overlay_correct.drawRectBounds(focusRects, 2) }
                quality_counter = 0
            }

        }
    }

    private fun plot_cross_hair(x: Float, y: Float, d: Float){
        val crossHairBegin = mutableListOf(mutableListOf(x - d, y, x + d, y),
                mutableListOf(x, y - d, x, y + d))
        cross_hair.post{cross_hair.drawCrossHair(crossHairBegin)}
    }

    private fun plot_rect(x: Float, y: Float, d: Float){
        val focusRects = listOf(RectF(x - d, y - d, x + d, y + d))
        rect_overlay.post { rect_overlay.drawRectBounds(focusRects) }
    }

    private fun plot_circle(x: Float, y: Float, r: Float){
        val circle = mutableListOf(mutableListOf(x, y, r))
        circle_overlay.post { circle_overlay.drawCircle(circle) }
    }

    // function to get max resolution for a given camera device
    private fun getMaxRes(id: Int): Size{
        val manager: CameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val cameraId = manager.getCameraIdList().get(id)
        val characteristics: CameraCharacteristics = manager.getCameraCharacteristics(cameraId)
        val map: StreamConfigurationMap = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
        val imageDimension = map.getOutputSizes(SurfaceTexture::class.java).get(0)
        Log.e(TAG, "imageDimension $imageDimension")
        return imageDimension;
    }

    private fun focusAtCenter(x: Float, y: Float){
        // trigger focus on tapped region
        val factory = viewFinder.getMeteringPointFactory()
        val point = factory.createPoint(x, y)
        val action = FocusMeteringAction.Builder(point).build()
        camera.cameraControl.startFocusAndMetering(action)
        // draw rectangle
        //plot_rect(x, y, rectSize.toFloat())
        // store time stamp when focus tapped
        focus_timestamp = System.currentTimeMillis()
        //Log.d(TAG, "Focus Timestamp is "+focus_timestamp)
    }
}

// canvas classes for drawing the rectangle and cross-hair
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

class DrawCrossHair constructor(context: Context?, attributeSet: AttributeSet?) :
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

class DrawLimbusWidth constructor(context: Context?, attributeSet: AttributeSet?) :
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

class DrawCircle constructor(context: Context?, attributeSet: AttributeSet?) :
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
