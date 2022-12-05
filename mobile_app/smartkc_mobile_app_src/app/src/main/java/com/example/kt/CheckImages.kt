package com.example.kt

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import android.os.AsyncTask
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.kt.data.repo.FileRepository
import org.apache.commons.io.comparator.LastModifiedFileComparator
import com.github.chrisbanes.photoview.PhotoView
import com.opencsv.CSVWriter
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.runBlocking
import org.opencv.imgproc.Imgproc
import org.opencv.android.Utils
import org.opencv.core.*
import java.io.File
import java.io.FileNotFoundException
import java.io.FileWriter
import java.io.IOException
import java.lang.Exception
import java.text.SimpleDateFormat
import java.util.*
import javax.inject.Inject

@AndroidEntryPoint
class CheckImages : AppCompatActivity(), View.OnClickListener {

    // File Repository
    @Inject
    lateinit var fileRepository: FileRepository

    val image_checker = ImageCheck()

    /** Called when the activity is first created.  */
    private var image_index = 0
    private var maxCounts = 0
    private var origMaxCounts = 0
    private lateinit var imageFiles: Array<File>

    // offset distance measure
    var offset_distance = -1.0

    //public class variables
    var dir_name: String? = null
    var left_right: String? = null
    var hash_map: HashMap<String, String>? = null
    var centerCutoff = 1.0f
    public override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_checkimages)

        // retrieve getExtras
        val bundle = intent.extras
        dir_name = bundle!!.getString("dir_name")
        left_right = bundle.getString("left_right")
        hash_map = intent.getSerializableExtra("hash_map") as HashMap<String, String>?
        origMaxCounts = bundle.getString("number_of_images")!!.toInt()
        image_index = 0 // initialize image index as 0

        // get app preferences
        val sharedPrefs = getSharedPreferences("KT_APP_PREFERENCES", MODE_PRIVATE)
        centerCutoff = sharedPrefs.getFloat("CENTER_CUTOFF", 0.5f)

        // list images in the directory
        val dir = File(
            getExternalFilesDir(null),
            MainActivity.PACKAGE_NAME + "/" + dir_name
        )
        imageFiles = dir.listFiles { dir, name ->
            name.lowercase(Locale.getDefault()).startsWith(
                left_right!!
            )
        }
        Arrays.sort(imageFiles, LastModifiedFileComparator.LASTMODIFIED_REVERSE)
        // incase there are less number of images than maxCounts
        maxCounts = Math.min(imageFiles.size, origMaxCounts)
        //Log.e("Counts", "maxCounts "+maxCounts);

        // previous
        val btnNo = findViewById<View>(R.id.no_btn) as Button
        btnNo.setOnClickListener(this)

        // next
        val btnYes = findViewById<View>(R.id.yes_btn) as Button
        btnYes.setOnClickListener(this)

        // remove view
        removeView()
        // show image
        //showImage();
        AsyncTask.execute { showImage() }
    }

    private fun showImage() {

        //PhotoView imgView = (PhotoView) findViewById(R.id.myimage);
        //imgView.setImageURI(Uri.fromFile(imageFiles[image_index]));

        // check if file is empty
        val fileEmpty = imageFiles[image_index].exists() && imageFiles[image_index].length() == 0L
        if (fileEmpty) {
            // emulate click
            runOnUiThread { pressNo() }
            return
        } else {
            // Find and display center, check if No then disable Yes
            var check = false
            check = try {
                checkCenter(imageFiles[image_index].toString(), centerCutoff)
            } catch (e: Exception) {
                runOnUiThread { pressNo() }
                e.printStackTrace()
                Log.e("CHECK_IMAGES", "Center not found or empty image!")
                return
            }
            if (!check) {
                runOnUiThread {
                    val btnYes = findViewById<View>(R.id.yes_btn) as Button
                    btnYes.isClickable = false
                    btnYes.visibility = View.GONE
                    val questionView = findViewById<TextView>(R.id.questionView)
                    questionView.text = "The image is not good!"
                }
            }
            runOnUiThread { // set prompt text and Image view
                val textView = findViewById<View>(R.id.textView) as TextView
                if (left_right == "right") textView.text =
                    "Right Eye: Image " + (image_index + 1) + "/" + maxCounts else textView.text =
                    "Left Eye: Image " + (image_index + 1) + "/" + maxCounts
            }
            return
        }
    }

    override fun onClick(v: View) {
        when (v.id) {
            R.id.no_btn -> {
                hash_map!![imageFiles[image_index].name] = "No"
                hash_map!![imageFiles[image_index].name + "_offset"] = "" + offset_distance
                image_index += 1
                if (image_index >= maxCounts) {
                    // if all images clicked NO then REDO
                    val intent = Intent(this, CameraActivityNew::class.java)
                    // add extras to intent
                    intent.putExtra("dir_name", dir_name)
                    intent.putExtra("left_right", left_right)
                    intent.putExtra("number_of_images", "" + origMaxCounts)
                    intent.putExtra("hash_map", hash_map)
                    //finish current activity
                    finish()
                    //start the second Activity
                    this.startActivity(intent)
                } else {
                    removeView()
                    AsyncTask.execute { showImage() }
                }
            }
            R.id.yes_btn -> {
                hash_map!![imageFiles[image_index].name] = "Yes"
                hash_map!![imageFiles[image_index].name + "_offset"] = "" + offset_distance
                var intent = Intent(this, GetGTDataActivity::class.java)
                if (left_right == "right") {
                    intent = Intent(this, PromptActivity::class.java)
                    // add extras to intent
                    intent.putExtra("dir_name", dir_name)
                    intent.putExtra("left_right", "left") // move to left
                    intent.putExtra("number_of_images", "" + origMaxCounts)
                    intent.putExtra("hash_map", hash_map)
                } else if (left_right == "left") {
                    // write the meta data file since both left and right complete
                    write_metadata()
                    // add extras to intent
                    intent.putExtra("dir_name", dir_name)
                }

                //finish current activity
                finish()
                //start the second Activity
                this.startActivity(intent)
            }
        }
    }

    private fun pressNo() {
        val noButtomView = findViewById<View>(R.id.no_btn)
        noButtomView.performClick()
        return
    }

    private fun removeView() {

        // remove photo
        val imgView = findViewById<View>(R.id.myimage) as PhotoView
        imgView.visibility = View.GONE
        // remove Yes button
        val btnYes = findViewById<View>(R.id.yes_btn) as Button
        btnYes.isClickable = false
        btnYes.visibility = View.GONE
        // remove No button
        val btnNo = findViewById<View>(R.id.no_btn) as Button
        btnNo.isClickable = false
        btnNo.visibility = View.GONE
        // set loading text
        val questionView = findViewById<TextView>(R.id.questionView)
        questionView.text = "Analyzing Image " + (image_index + 1) + " ..."
        // remove pinch view
        val pinchView = findViewById<TextView>(R.id.pinchView)
        pinchView.visibility = View.GONE
        // remove text view
        val textView = findViewById<TextView>(R.id.textView)
        textView.visibility = View.GONE
    }

    private fun addView() {

        // remove photo
        val imgView = findViewById<View>(R.id.myimage) as PhotoView
        imgView.visibility = View.VISIBLE
        // remove Yes button
        val btnYes = findViewById<View>(R.id.yes_btn) as Button
        btnYes.isClickable = true
        btnYes.visibility = View.VISIBLE
        // remove No button
        val btnNo = findViewById<View>(R.id.no_btn) as Button
        btnNo.isClickable = true
        btnNo.visibility = View.VISIBLE
        // set loading text
        val questionView = findViewById<TextView>(R.id.questionView)
        questionView.text = "Is the image in focus and sharp?"
        // remove pinch view
        val pinchView = findViewById<TextView>(R.id.pinchView)
        pinchView.visibility = View.VISIBLE
        // remove text view
        val textView = findViewById<TextView>(R.id.textView)
        textView.visibility = View.VISIBLE
    }

    fun write_metadata() {
        val dir = File(
            getExternalFilesDir(null),
            MainActivity.PACKAGE_NAME + "/" + dir_name
        )
        val files = dir.listFiles()
        Arrays.sort(files, LastModifiedFileComparator.LASTMODIFIED_REVERSE)
        Log.e("Finish", "Inside Write Metadata Size: " + files.size)
        val filePath = getExternalFilesDir(null)!!.absolutePath + File.separator +
                MainActivity.PACKAGE_NAME + File.separator + dir_name + File.separator + dir_name + ".csv"
        Log.e("Finish", "Inside Write Metadata filePath $filePath")
        val f = File(filePath)
        var writer: CSVWriter? = null

        // File exist
        if (f.exists() && !f.isDirectory) {
            try {
                val mFileWriter = FileWriter(filePath, true)
                writer = CSVWriter(mFileWriter)
            } catch (e: FileNotFoundException) {
                System.err.print("File not found")
            } catch (e: IOException) {
                System.err.print("Something went wrong")
            }
        } else {
            try {
                writer = CSVWriter(FileWriter(filePath))
            } catch (e: FileNotFoundException) {
                System.err.print("File not found")
            } catch (e: IOException) {
                System.err.print("Something went wrong")
            }
        }
        val data = arrayOf(
            "Patient Id",
            "Age",
            "Gender",
            "App Version",
            "Image Name",
            "Date",
            "Time",
            "Is OK",
            "Offset",
            "Cutoff"
        )
        writer!!.writeNext(data)
        try {

            //List<String> letters = Arrays.asList(string.split(""));
            for (i in files.indices) {
                Log.d("Files", "FileName:" + files[i].name)
                val namesList = dir_name!!.split("_".toRegex()).toTypedArray()
                val id = i + 1
                val patientId = namesList[0]
                val patientAge = namesList[1]
                val patientGender = namesList[2]
                val isOk =
                    if (hash_map!!.containsKey(files[i].name)) hash_map!![files[i].name] else "NA"
                val offset =
                    if (hash_map!!.containsKey(files[i].name + "_offset")) hash_map!![files[i].name + "_offset"] else "NA"
                val sdf = SimpleDateFormat("MM/dd/yyyy, hh:mm:ss aa")
                val date_time =
                    sdf.format(files[i].lastModified()).split(",".toRegex()).toTypedArray()
                val curr_data = arrayOf(
                    patientId,
                    patientAge,
                    patientGender,
                    MainActivity.VERSION,
                    files[i].name,
                    date_time[0],
                    date_time[1],
                    isOk,
                    offset,
                    centerCutoff.toString()
                )
                writer.writeNext(curr_data)
            }
            writer.close()
            val sharedPrefs = getSharedPreferences("KT_APP_PREFERENCES", MODE_PRIVATE)
            val center_name = sharedPrefs.getString("CENTER_NAME", "")
            if (center_name.isNullOrEmpty()) {
                throw Error("Center name cannot be null")
            }
            val fileName = "${center_name}/${dir_name?.split("_")?.get(0)}/meta_data.csv"
            runBlocking { fileRepository.insertNewFileRecord(Uri.fromFile(f).toString(), fileName) }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun checkCenter(currImage: String, centerThresh: Float): Boolean {

        //String imageString = imageFile.toString();
        var bitmap = BitmapFactory.decodeFile(currImage)
        val imageFile = File(currImage)
        // get Exif data for rotation
        try {
            val exif = ExifInterface(imageFile.absolutePath)
            val rotation = exif.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
            val rotationInDegrees = exifToDegrees(rotation)
            val matrix = Matrix()
            if (rotation.toFloat() != 0f) {
                matrix.preRotate(rotationInDegrees.toFloat())
                bitmap =
                    Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        val imageFull = Mat()
        Utils.bitmapToMat(bitmap, imageFull)
        val image = Mat()
        Imgproc.resize(imageFull, image, Size((480 * 4).toDouble(), (640 * 4).toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        val zoom_factor = 2
        // add multiplying factor & choosing parameters for detecting circles based on 3000x4000 image
        val basewidth = 3000.0f
        val baseheight = 4000.0f
        val baseminDist = 1200.0f * zoom_factor
        val baseminR = 595.0f * zoom_factor
        val basemaxR = 615.0f * zoom_factor
        val normfactor = Math.sqrt(
            ((basewidth * basewidth + baseheight * baseheight) /
                    (image.cols() * image.cols() + image.rows() * image.rows())).toDouble()
        )
        val crosshair_center = detectCrossHair(
            image.clone(),
            2.5,
            baseminDist / normfactor,
            (baseminR / normfactor).toInt() - 5,
            (basemaxR / normfactor).toInt() + 5
        )
        val start = (30 * zoom_factor / normfactor).toInt()
        val end = (100 * zoom_factor / normfactor).toInt()
        val jump = (10 * zoom_factor / normfactor).toInt()
        val mire_center =
            detectMireCenter(image.clone(), 2.5, baseminDist / normfactor, start, end, jump)

        // draw cross_hair
        try {
            Core.line(
                image, Point(
                    (crosshair_center[0] - 25).toDouble(), crosshair_center[1].toDouble()
                ), Point(
                    (crosshair_center[0] + 25).toDouble(), crosshair_center[1].toDouble()
                ), Scalar(0.0, 0.0, 255.0), 4
            )
            Core.line(
                image, Point(
                    crosshair_center[0].toDouble(), (crosshair_center[1] - 25).toDouble()
                ), Point(
                    crosshair_center[0].toDouble(), (crosshair_center[1] + 25).toDouble()
                ), Scalar(0.0, 0.0, 255.0), 4
            )
        } catch (e: Exception) {
            e.printStackTrace()
        }
        try {
            Core.circle(
                image, Point(
                    mire_center[0].toDouble(), mire_center[1].toDouble()
                ), 15, Scalar(0.0, 255.0, 0.0), -1, 8, 0
            )
        } catch (e: Exception) {
            e.printStackTrace()
        }

        // distance b/w mireCenter and crossHair
        val dist = Math.sqrt(
            Math.pow(
                (crosshair_center[0] - mire_center[0]).toDouble(),
                2.0
            ) + Math.pow((crosshair_center[1] - mire_center[1]).toDouble(), 2.0)
        )
        //boolean check1 = dist < (centerThresh*zoom_factor/normfactor); // check 1: if the offset is within threshold
        offset_distance =
            dist * 30 / (2 * crosshair_center[2] + 0.000001) // update offset distance in mm
        val check1 =
            offset_distance <= centerThresh // check 1: if the offset is within threshold in mm
        //Log.e("Counts", "CENTER_CUTOFF_PREFERENCE "+centerThresh+" Offset "+offset_distance+" dist in pixels "+dist);
        //image = this.image_checker.sharpen(image);
        //image = this.image_checker.autoCanny(image, 0.33);

        // check 2: image is neither over-exposed or under-exposed
        var gray = image.clone()
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_RGB2GRAY) // converting image to grayscale
        gray = image_checker.cropResultWindow(
            gray!!, (500 * zoom_factor / normfactor).toInt(),
            Point(mire_center[0].toDouble(), mire_center[1].toDouble())
        )
        val exposure = image_checker.checkExposure(gray!!.clone())
        val check2 = exposure === ExposureResult.NORMAL
        // check 3: image is sharp enough
        val check3 = image_checker.checkSharpness(gray, 0.96)
        val bitmapSmall = Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(image, bitmapSmall)
        runOnUiThread {
            addView()
            val imgView = findViewById<View>(R.id.myimage) as PhotoView
            imgView.setImageBitmap(bitmapSmall)
        }
        return check1 && check2 && check3
    }

    // this code was auto de-compiled to JAVA from Kotlin (have to re-factor this)
    private fun detectCrossHair(
        image: Mat,
        dp: Double,
        minDist: Double,
        minR: Int,
        maxR: Int
    ): Array<Float> {
        val output_points = arrayOf(-1.0f, -1.0f, 0.0f)
        val image_sharpened = image_checker.sharpen(image)
        //Imgproc.medianBlur(image_sharpened, image_sharpened, 5);
        val detected_circles = image_checker.detectCircles(image_sharpened, dp, minDist, minR, maxR)
        val xx = detected_circles[0]
        val yy = detected_circles[1]
        val rad = detected_circles[2]
        val image_xx = image.cols().toFloat() / 2f
        val image_yy = image.rows().toFloat() / 2f
        if (xx > image_xx - 75f && xx < image_xx + 75f && yy > image_yy - 100f && yy < image_yy + 100f) {
            output_points[0] = xx
            output_points[1] = yy
            output_points[2] = rad
            return output_points
        }
        return output_points
    }

    private fun detectMireCenter(
        image: Mat,
        dp: Double,
        minDist: Double,
        minR: Int,
        maxR: Int,
        jump: Int
    ): Array<Float> {
        val output_points = arrayOf(-1.0f, -1.0f)
        var xx = 0.0f
        var yy = 0.0f
        var rCount = 0
        var currRadius = minR
        while (currRadius <= maxR) {
            val detected_circles = image_checker.detectCircles(
                image.clone(),
                dp,
                minDist,
                currRadius,
                currRadius + jump
            )
            if (detected_circles[2] > 0) {
                xx += detected_circles[0]
                yy += detected_circles[1]
                rCount += 1
            }
            currRadius += jump
        }
        xx /= rCount.toFloat()
        yy /= rCount.toFloat()
        output_points[0] = xx
        output_points[1] = yy
        return output_points
    }

    private fun exifToDegrees(exifOrientation: Int): Int {
        if (exifOrientation == ExifInterface.ORIENTATION_ROTATE_90) {
            return 90
        } else if (exifOrientation == ExifInterface.ORIENTATION_ROTATE_180) {
            return 180
        } else if (exifOrientation == ExifInterface.ORIENTATION_ROTATE_270) {
            return 270
        }
        return 0
    } /*
    class BackgroundTask extends AsyncTask<String, Integer, Boolean> {

        @Override
        protected Boolean doInBackground(String... params) {

            boolean check = checkCenter(params[0], 50.0F);
            return check;
        }
        @Override
        protected void onPostExecute(Boolean result) {
            Log.d("Over Check", " "+result);
            ProgressBar spinner;
            spinner = (ProgressBar)findViewById(R.id.progressBar);
            spinner.setVisibility(View.GONE);

        }
        @Override
        protected void onPreExecute() {

            ProgressBar spinner;
            spinner = (ProgressBar)findViewById(R.id.progressBar);
            spinner.setVisibility(View.VISIBLE);
        }

        @Override
        protected void onProgressUpdate(Integer... values) {
            Log.d("Progress", String.valueOf(values[0]));
        }

    }
     */
}