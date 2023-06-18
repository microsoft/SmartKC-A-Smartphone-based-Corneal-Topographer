package com.example.kt.ui

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.media.MediaActionSound
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.util.SizeF
import android.view.MotionEvent
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
import com.example.kt.*
import com.example.kt.R
import com.example.kt.data.repo.FileRepository
import com.example.kt.injection.dataStore
import com.example.kt.utils.DrawUtils
import com.example.kt.utils.ImageUtils
import com.example.kt.utils.PreferenceKeys
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.android.synthetic.main.activity_cameranew.*
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.opencv.android.Utils
import org.opencv.core.*
import java.io.File
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import javax.inject.Inject
import kotlin.math.sqrt

@ExperimentalCamera2Interop @ExperimentalCameraFilter @AndroidEntryPoint
class NgCameraActivityNew : AppCompatActivity() {

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
    var center_name: String? = null
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
    var lock_button_flag: Boolean = false
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
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        // Set up the listener for take photo button
        camera_capture_button.setOnClickListener { takePhoto() }
        // setup lock_button listener
        unlock_cross_switch.setOnClickListener{
            lock_button_flag = unlock_cross_switch.isChecked
        }
        // auto_capture lock
        unlock_auto_capture_click.setOnClickListener{
            lock_auto_capture_flag = !unlock_auto_capture_click.isChecked
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
        // Get center name
        center_name = sharedPrefs.getString("CENTER_NAME", "")
        if (center_name.isNullOrEmpty()) {
              throw Error("Center name cannot be null")
        }


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

        capture_timestamp = System.currentTimeMillis()

        // if auto-focus not triggered, re-focus at center
        if (focus_timestamp == -1L || (System.currentTimeMillis()-focus_timestamp)/1000 >= time_diff) {
            focusAtCenter(trigger_coords[0], trigger_coords[1])
            // focusAtCenter((viewFinder.width).toFloat() / 2, (viewFinder.height).toFloat() / 2) // older when triggering at center
            //Log.d(TAG, "Time diff in seconds"+((System.currentTimeMillis()-focus_timestamp)/1000))
        }

        // Get a stable reference of the modifiable image capture use case
        val imageCapture = imageCapture ?: return

        // Create indexed output file to hold the image
        val fileName = center_name + "_" + dir_name + "_" +  left_right + "_" + (idx + currentCounts) + ".jpg"
        val photoFile = File(outputDirectory, fileName)

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
                    Log.d(TAG, msg)
                    // Record it in database
                    runBlocking {
                        val blobFileName = "${center_name}/${dir_name?.split("_")?.get(0)}/${left_right}/${fileName}"
                        fileRepository.insertNewFileRecord(savedUri.toString(), blobFileName)
                    }
                    Toast.makeText(baseContext, "Counts:" + (currentCounts + 1) + "/" + maxCounts, Toast.LENGTH_SHORT).show()
                    // play capture sound
                    val sound = MediaActionSound()
                    sound.play(MediaActionSound.SHUTTER_CLICK)

                    //increment counts, and move to next activity after image has been saved
                    currentCounts += 1
                    //Log.e(TAG, "Current counts: " + currentCounts + " maxCounts " + maxCounts)
                    if (currentCounts >= maxCounts) {
                        var cameraPhysicalSize: SizeF? = null
                        var focalSize: FloatArray? = null
                        runBlocking {
                            val data = dataStore.data.first()
                            val selectedCameraKey =
                                stringPreferencesKey(PreferenceKeys.CHOSEN_CAMERA)
                            val selectedCamera = data[selectedCameraKey] ?: "0"
                            val manager = getSystemService(CAMERA_SERVICE) as CameraManager
                            cameraPhysicalSize = manager.getCameraCharacteristics(selectedCamera).get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                            focalSize = manager.getCameraCharacteristics(selectedCamera).get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                        }
                        //cameraProvider.unbindAll()
                        val intent = Intent(this@NgCameraActivityNew, NgCheckImages::class.java)
                        // add extras to intent
                        intent.putExtra("dir_name", dir_name)
                        intent.putExtra("left_right", left_right)
                        intent.putExtra("number_of_images", "" + maxCounts)
                        intent.putExtra("hash_map", hash_map)
                        intent.putExtra("camera_physical_size", cameraPhysicalSize.toString())
                        intent.putExtra("focal_length", focalSize?.get(0))
                        //finish current activity
                        finish()
                        //start the second Activity
                        this@NgCameraActivityNew.startActivity(intent)
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
                if (selectedCamera != null) {
                    cameraSelector = CameraSelector.Builder().addCameraFilter {
                        val filtered = it.filter {
                            val cameraId = Camera2CameraInfo.fromCameraInfo(it.cameraInfo).cameraId
                            return@filter cameraId == selectedCamera }
                        val result = LinkedHashSet<Camera>()
                        filtered.forEach { result.add(it) }
                        result
                    }
                        .build()
                }
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
        if(this@NgCameraActivityNew::camera.isInitialized) {
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
        private const val TAG = "NgCameraActivityNew"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    // image analysis class
    inner private class MyImageAnalyzer : ImageAnalysis.Analyzer {
        val drawUtils = DrawUtils()
        @SuppressLint("UnsafeExperimentalUsageError")
        override fun analyze(image: ImageProxy) {

            /* Create cv::mat(RGB888) from image(NV21) */
            //val matOrg = getMatFromImage(image)
            val temp_bmp = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
            val converter = YuvToRgbConverter(this@NgCameraActivityNew)
            converter.yuvToRgb(image.getImage()!!, temp_bmp)
            val matOrg = Mat()
            Utils.bitmapToMat(temp_bmp, matOrg);

            /* Fix image rotation (it looks image in PreviewView is automatically fixed by CameraX???) */
            val matFixed = image_checker.fixMatRotation(matOrg, myPreviewView)
            val imageUtils = ImageUtils(matFixed)

            /* Do some image processing */
            val basewidth = 3000.0
            val baseheight = 4000.0
            val baseminDist = 1200.0*zoom_factor
            val baseminR = 595.0*zoom_factor
            val basemaxR = 615.0*zoom_factor
            val normfactor = sqrt((basewidth * basewidth + baseheight * baseheight) / (image.width * image.width + image.height * image.height))
            // L2 Norm

            // detect the central cross-hair
            var crosshair_center = arrayOf<Float>(trigger_coords[0] / ((viewFinder.width).toFloat() / matFixed.cols().toFloat()),
                trigger_coords[1] / ((viewFinder.height).toFloat() / matFixed.rows().toFloat()))
            // Try to detect cross hair for every trigger rate if the user has unclocked the cross or it is the first time that
            // we are trying to detect the cross hair (both trigger coordinates are zero)
            if (frames_elapsed % trigger_rate == 0 && ((trigger_coords[0] == 0F && trigger_coords[1] == 0F) || !lock_button_flag)) {
                // returns un-normalized mire center in the image not preview
                var crossHairOutput = imageUtils.detectCrossHair(2.5,
                    baseminDist / normfactor,
                    (baseminR / normfactor).toInt() - 2, (basemaxR / normfactor).toInt() + 3)
                val updated_center = arrayOf<Float>(crossHairOutput[0], crossHairOutput[1])
                if(updated_center[0]!=-1.0f && updated_center[1]!=-1.0f){

                    val xx = crossHairOutput[0]
                    val yy = crossHairOutput[1]
                    val rad = crossHairOutput[2]

                    crosshair_center = updated_center
                    trigger_coords[0] = xx * ((viewFinder.width).toFloat() / imageUtils.image.cols().toFloat())
                    trigger_coords[1] = yy * ((viewFinder.height).toFloat() / imageUtils.image.rows().toFloat())
                    // Draw cross hair if detected
                    DrawUtils.plotCrossHair(cross_hair, trigger_coords[0], trigger_coords[1], 50F)
                    val normfactor = sqrt((viewFinder.width * viewFinder.width + viewFinder.height * viewFinder.height).toDouble()
                            / (imageUtils.image.cols() * imageUtils.image.cols() + imageUtils.image.rows() * imageUtils.image.rows()).toDouble()).toFloat()
                    DrawUtils.plotCircle(circle_overlay, trigger_coords[0], trigger_coords[1], rad * normfactor)
                }
            }

            //Log.e(TAG, "CHECKER "+frames_elapsed+" "+(baseminR / normfactor).toInt()+" "+(basemaxR / normfactor).toInt() +" "+matFixed.size())
            // detect the mire center
            val start = (30*zoom_factor/normfactor).toInt(); val end = (100*zoom_factor/normfactor).toInt()
            val jump = (10*zoom_factor/normfactor).toInt()
            val mire_center = imageUtils.detectMireCenter(2.5, baseminDist / normfactor, start, end, jump)
            // check if the center was detected
            Log.i("mire_center:", "${mire_center[0]} + ${mire_center[1]}")
            if (mire_center[0] != -1F && mire_center[1] != -1F) {
                // overlay cross hair
                val xx = mire_center[0] * ((viewFinder.width).toFloat() / imageUtils.image.cols().toFloat())
                val yy = mire_center[1] * ((viewFinder.height).toFloat() / imageUtils.image.rows().toFloat())
                val scale_factor = (zoom_factor*viewFinder.width/basewidth).toFloat()
                // draw crossHair for mire
                DrawUtils.plotCrossHair(cross_hair_mire, xx, yy, 10F)

                // draw limbus pointer
                val limbusPoints = mutableListOf(mutableListOf(xx - limbus_radius*scale_factor, yy, xx + limbus_radius*scale_factor, yy),
                    mutableListOf(xx - limbus_radius*scale_factor, yy - 10, xx-limbus_radius*scale_factor, yy + 10),
                    mutableListOf(xx + limbus_radius*scale_factor, yy - 10, xx+limbus_radius*scale_factor, yy + 10))
                limbus_width.post { limbus_width.drawLimbus(limbusPoints) }
            }


            // image quality check
            val qualityResults = imageUtils.qualityCheck(crosshair_center, mire_center, normfactor, correct_cutoff, zoom_factor)
            val centerCheck = qualityResults.first
            val exposureQuality = qualityResults.second
            val exposureCheck = exposureQuality == ExposureResult.NORMAL
            val sharpnessCheck = qualityResults.third

            // --- Update UI based on quality results --- //

            // UI update on center check
            if(centerCheck) {
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

            // UI update on exposure check
            if(exposureQuality == ExposureResult.OVER_EXPOSED) {
                runOnUiThread(Runnable {
                    text_over_under.setText("EXPOSURE: X")
                    text_over_under.setTextColor(Color.parseColor("#F44336"))
                })
            }
            else if(exposureQuality == ExposureResult.UNDER_EXPOSED) {
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

            // UI update on sharpness check
            if(sharpnessCheck) {
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
            if(centerCheck && exposureCheck && sharpnessCheck){
                val focusRects = listOf(RectF(0.0F, 0.0F, viewFinder.width.toFloat(), viewFinder.height.toFloat()))
                rect_overlay_correct.post { rect_overlay_correct.drawRectBounds(focusRects, 1) }
                quality_counter += 1
            }
            else{
                val focusRects = listOf(RectF(0.0F, 0.0F, viewFinder.width.toFloat(), viewFinder.height.toFloat()))
                rect_overlay_correct.post { rect_overlay_correct.drawRectBounds(focusRects, 2) }
                quality_counter = 0
            }

            // ------//


            frames_elapsed += 1
            if (lock_auto_capture_flag) {
                runOnUiThread(Runnable {
                    progressBar.progress = 100 * quality_counter / quality_threshold
                })
            }
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
