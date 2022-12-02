package com.example.kt

import android.content.Intent
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.View
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.kt.data.repo.FileRepository
import com.opencsv.CSVWriter
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.runBlocking
import java.io.File
import java.io.FileNotFoundException
import java.io.FileWriter
import java.io.IOException
import java.lang.Exception
import javax.inject.Inject

@AndroidEntryPoint
class GetGTDataActivity : AppCompatActivity(), View.OnClickListener {
    var dir_name: String? = null
    // File Repository
    @Inject
    lateinit var fileRepository: FileRepository
    public override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        //setting activity view
        setContentView(R.layout.activity_gt)

        //get the Button reference
        val buttonClick = findViewById<View>(R.id.FinishTestButton)
        //set event listener
        buttonClick.setOnClickListener(this)

        //skip button reference
        val skipButton = findViewById<View>(R.id.SkipTestButton)
        skipButton.setOnClickListener(this)

        // get extras
        val bundle = intent.extras
        dir_name = bundle!!.getString("dir_name")
    }

    //override the OnClickListener interface method
    override fun onClick(arg0: View) {
        if (arg0.id == R.id.FinishTestButton) {

            // get right eye data
            val editRightSph = findViewById<View>(R.id.EditRightSph) as EditText
            val rightSph = editRightSph.text.toString()
            val editRightCyl = findViewById<View>(R.id.EditRightCyl) as EditText
            val rightCyl = editRightCyl.text.toString()
            val editRightAxis = findViewById<View>(R.id.EditRightAxis) as EditText
            val rightAxis = editRightAxis.text.toString()
            val editRightKC = findViewById<View>(R.id.EditRightKC) as EditText
            val rightKC = editRightKC.text.toString()

            // get left eye data
            val editLeftSph = findViewById<View>(R.id.EditLeftSph) as EditText
            val leftSph = editLeftSph.text.toString()
            val editLeftCyl = findViewById<View>(R.id.EditLeftCyl) as EditText
            val leftCyl = editLeftCyl.text.toString()
            val editLeftAxis = findViewById<View>(R.id.EditLeftAxis) as EditText
            val leftAxis = editLeftAxis.text.toString()
            val editLeftKC = findViewById<View>(R.id.EditLeftKC) as EditText
            val leftKC = editLeftKC.text.toString()
            val check_gt =
                leftSph.length > 0 && leftCyl.length > 0 && leftAxis.length > 0 && leftKC.length > 0 && rightSph.length > 0 && rightCyl.length > 0 && rightAxis.length > 0 && rightKC.length > 0
            val gt_data =
                arrayOf(rightSph, rightCyl, rightAxis, rightKC, leftSph, leftCyl, leftAxis, leftKC)
            Log.e("TEST", "GT_DATA $gt_data")
            if (!check_gt) {
                val toast =
                    Toast.makeText(this, "Enter the Patient Refractive Data", Toast.LENGTH_SHORT)
                toast.setGravity(Gravity.CENTER_VERTICAL, 0, 200)

                // set color for toast
                val view = toast.view
                val text = view!!.findViewById<TextView>(android.R.id.message)
                text.setTextColor(Color.RED)
                toast.show()
            } else {
                // raise toast
                val toast = Toast.makeText(this, "Test Complete!", Toast.LENGTH_SHORT)
                toast.setGravity(Gravity.CENTER_VERTICAL, 0, 200)
                toast.show()
                //define a new Intent for the main Activity
                val intent = Intent(this, MainActivity::class.java)
                //write data
                write_gtdata(gt_data, dir_name)
                //finish current activity
                finish()
                //start the second Activity
                this.startActivity(intent)
            }
        } else if (arg0.id == R.id.SkipTestButton) {
            // raise toast
            val toast = Toast.makeText(this, "Test Complete!", Toast.LENGTH_SHORT)
            toast.setGravity(Gravity.CENTER_VERTICAL, 0, 200)
            toast.show()

            //define a new Intent for the main Activity
            val intent = Intent(this, MainActivity::class.java)
            //finish current activity
            finish()
            //start the second Activity
            this.startActivity(intent)
        }
    }

    fun write_gtdata(gt_data: Array<String>?, dir_name: String?) {
        var dir = File(getExternalFilesDir(null), MainActivity.PACKAGE_NAME)
        if (!dir.exists()) {
            dir.mkdirs()
        }
        dir = File(
            getExternalFilesDir(null),
            MainActivity.PACKAGE_NAME + "/" + dir_name
        )
        if (!dir.exists()) {
            dir.mkdirs()
        }
        val filePath = getExternalFilesDir(null).toString() + File.separator +
                MainActivity.PACKAGE_NAME + File.separator + dir_name + File.separator + dir_name + "_gt_data.csv"
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
            "Right_Sph",
            "Right_Cyl",
            "Right_Axis",
            "Right_KC",
            "Left_Sph",
            "Left_Cyl",
            "Left_Axis",
            "Left_KC"
        )
        writer!!.writeNext(data)
        try {
            writer.writeNext(gt_data)
            writer.close()
            val sharedPrefs = getSharedPreferences("KT_APP_PREFERENCES", MODE_PRIVATE)
            val center_name = sharedPrefs.getString("CENTER_NAME", "")
            if (center_name.isNullOrEmpty()) {
                throw Error("Center name cannot be null")
            }
            val fileName = "${center_name}/${dir_name?.split("_")?.get(0)}/gt_data.csv"
            runBlocking { fileRepository.insertNewFileRecord(Uri.fromFile(f).toString(), fileName) }

        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}