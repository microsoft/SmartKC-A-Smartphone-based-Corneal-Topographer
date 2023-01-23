package com.example.kt

import android.Manifest
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.lifecycle.lifecycleScope
import androidx.work.*
import com.balsikandar.crashreporter.CrashReporter
import com.example.kt.data.repo.FileRepository
import com.example.kt.utils.PreferenceKeys
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import java.io.File
import javax.inject.Inject

//implement the OnClickListener interface
@AndroidEntryPoint
class MainActivity : AppCompatActivity(), View.OnClickListener {
    companion object {
        /** Called when the activity is first created.  */
        private const val REQUEST_CAMERA_PERMISSION = 200

        @JvmField
        var PACKAGE_NAME: String? = null

        @JvmField
        var VERSION: String? = null

        init {
            System.loadLibrary("opencv_java")
        }
    }

    private val UNIQUE_SYNC_WORK_NAME = "UPLOAD_FILE"

    // File Repository
    @Inject
    lateinit var fileRepository: FileRepository

    // Datastore
    @Inject
    lateinit var dataStore: DataStore<Preferences>

    private var centerName: String? = null
    private lateinit var uploadFileStatusTv: TextView
    private lateinit var uploadFileErrorTv: TextView
    private lateinit var uploadBtn: Button

    public override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        //setting activity view
        setContentView(R.layout.activity_main)
        PACKAGE_NAME = applicationInfo.loadLabel(packageManager).toString()
        VERSION = "14"

        // setting app preferences
        setPreferences()

        // Add permission for camera and let user grant the permission
        // Checking for Camera, Write / Read Permissions
        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
            ) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.READ_EXTERNAL_STORAGE
            ) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.READ_EXTERNAL_STORAGE
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(
                    Manifest.permission.CAMERA,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE,
                    Manifest.permission.READ_EXTERNAL_STORAGE,
                    Manifest.permission.READ_MEDIA_IMAGES
                ),
                REQUEST_CAMERA_PERMISSION
            )
        }

        //get the Button reference
        val buttonClick = findViewById<View>(R.id.new_button)
        buttonClick.setOnClickListener(this) //set event listener

        //get view_records button reference
        val viewRecords = findViewById<View>(R.id.view_records)
        viewRecords.setOnClickListener(this)

        //get update preferences button reference
        val updatePrefs = findViewById<View>(R.id.update_preferences)
        updatePrefs.setOnClickListener(this)

        // get upload button reference
        uploadBtn = findViewById(R.id.upload_files_btn)

        // get center set button reference
        val saveCenterBtn = findViewById<Button>(R.id.save_center_btn)

        // center edittext
        val centerEt = findViewById<EditText>(R.id.center_name_editText)

        // Upload file status textview
        uploadFileStatusTv = findViewById(R.id.upload_file_status_tv)

        // Upload file error textview
        uploadFileErrorTv = findViewById(R.id.upload_file_error_tv)
        // Make error tv invisible
        uploadFileErrorTv.visibility = View.INVISIBLE

        val sharedPrefs = getSharedPreferences("KT_APP_PREFERENCES", MODE_PRIVATE)

        centerName = sharedPrefs.getString("CENTER_NAME", "")
        centerEt.setText(centerName)

        // set file upload status
        lifecycleScope.launch {
            val unUploadedFiles = fileRepository.getUnUploadedFileRecords()
            uploadFileStatusTv.text = "Files to Upload: ${unUploadedFiles.size}"
        }

        saveCenterBtn.setOnClickListener {
            centerName = centerEt.text.toString()
            if (centerName.isNullOrEmpty()) {
                Toast.makeText(applicationContext, "Please enter a center name", Toast.LENGTH_LONG)
                    .show()
            } else {
                val ed = sharedPrefs.edit()
                ed.putString("CENTER_NAME", centerName)
                ed.commit()
                Toast.makeText(applicationContext, "Center name saved", Toast.LENGTH_LONG).show()
            }
        }

        uploadBtn.setOnClickListener {
            Toast.makeText(applicationContext, "Uploading files", Toast.LENGTH_LONG).show()
            val constraints = Constraints.Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                .build()

            val syncWorkRequest = OneTimeWorkRequestBuilder<UploadWorker>()
                .setConstraints(constraints)
                .build()
            WorkManager.getInstance(applicationContext)
                .enqueueUniqueWork(UNIQUE_SYNC_WORK_NAME, ExistingWorkPolicy.KEEP, syncWorkRequest)

        }

        WorkManager.getInstance(this)
            .getWorkInfosForUniqueWorkLiveData(UNIQUE_SYNC_WORK_NAME)
            .observe(this) { workInfos ->
                lifecycleScope.launch {
                    // Determine if upload is available
                    val data = dataStore.data.first()
                    val uploadEnabledKey = booleanPreferencesKey(PreferenceKeys.UPLOAD_ENABLED)
                    val uploadEnabled = data[uploadEnabledKey] ?: BuildConfig.UPLOAD_ENABLED
                    if (!uploadEnabled) return@launch //Return if upload is not enabled
                    if (workInfos.size == 0) return@launch // Return if the workInfo List is empty
                    val workInfo = workInfos[0] // Picking the first workInfo
                    if (workInfo != null && workInfo.state == WorkInfo.State.SUCCEEDED) {
                        val errorMsg = workInfo.outputData.keyValueMap["errorMsg"] as String?
                        val unUploadedFilesCount =
                            workInfo.outputData.keyValueMap["unUploadedFilesCount"] as String?
                        if ((unUploadedFilesCount ?: "0").toInt() > 0) {
                            uploadFileErrorTv.text =
                                "Cannot upload $unUploadedFilesCount files \n error: $errorMsg"
                            uploadFileErrorTv.visibility = View.VISIBLE
                        }
                        val unUploadedFiles = fileRepository.getUnUploadedFileRecords()
                        uploadFileStatusTv.text = "Files to Upload: ${unUploadedFiles.size}"
                    }
                    if (workInfo != null && workInfo.state == WorkInfo.State.RUNNING) {
                        val fileIndex = workInfo.progress.getInt("progress", -1)
                        if (fileIndex > -1) {
                            uploadFileStatusTv.text = "Uploading file ${fileIndex + 1} "
                        }
                        uploadFileErrorTv.visibility = View.INVISIBLE
                    }
                    if (workInfo != null && workInfo.state == WorkInfo.State.FAILED) {
                        Toast.makeText(applicationContext, "FAILED TO UPLOAD SOME FILES", Toast.LENGTH_LONG)
                            .show()
                        val errorMsg = workInfo.outputData.getString("errorMsg")
                        uploadFileErrorTv.text = errorMsg
                        uploadFileErrorTv.visibility = View.VISIBLE
                    }
                }
            }

        var dir = File(getExternalFilesDir(null), PACKAGE_NAME)
        if (!dir.exists()) {
            dir.mkdirs()
        }
        // create directory for testing crash logs
        dir =
            File(getExternalFilesDir(null), PACKAGE_NAME + "/" + "test_crash_logs")
        if (!dir.exists()) {
            dir.mkdirs()
        }
        // this is to log crash reports (used external lib)
        CrashReporter.initialize(this, dir.toString())
    }

    override fun onResume() {
        super.onResume()
        // Hide upload button and error tv if upload is deactivated
        lifecycleScope.launch {
            val data = dataStore.data.first()
            val uploadEnabledKey = booleanPreferencesKey(PreferenceKeys.UPLOAD_ENABLED)
            val uploadEnabled = data[uploadEnabledKey] ?: BuildConfig.UPLOAD_ENABLED
            if (!uploadEnabled) {
                uploadBtn.visibility = View.GONE
                uploadFileErrorTv.visibility = View.GONE
                uploadFileStatusTv.visibility = View.GONE
            } else {
                uploadBtn.visibility = View.VISIBLE
                uploadFileErrorTv.visibility = View.VISIBLE
                uploadFileStatusTv.visibility = View.VISIBLE
            }
        }
    }

    //override the OnClickListener interface method
    override fun onClick(arg0: View) {
        if (arg0.id == R.id.new_button) {
            if (centerName.isNullOrEmpty()) {
                Toast.makeText(applicationContext, "Please enter a center name", Toast.LENGTH_LONG)
                    .show()
                return
            }
            //define a new Intent for the second Activity
            val intent = Intent(this, DataActivity::class.java)
            //finish current activity
            finish()
            //start the second Activity
            this.startActivity(intent)
        } else if (arg0.id == R.id.view_records) {

            //define a new Intent for the second Activity
            val intent = Intent(this, ViewRecordActivity::class.java)
            //finish current activity
            finish()
            //start the second Activity
            this.startActivity(intent)
        } else if (arg0.id == R.id.update_preferences) {

            //define a new Intent for the second Activity
            val intent = Intent(this, UpdatePreferences::class.java)
            //start the second Activity
            this.startActivity(intent)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
                // close the app
                Toast.makeText(
                    this,
                    "Sorry!!!, you can't use this app without granting permission",
                    Toast.LENGTH_LONG
                ).show()
                finish()
            }
        }
    }

    fun setPreferences() {
        // setting app preferences
        val sharedPrefs = getSharedPreferences("KT_APP_PREFERENCES", MODE_PRIVATE)
        // initialize for the first time
        val ed: SharedPreferences.Editor
        if (!sharedPrefs.contains("INITIALIZED")) {
            Log.e("MAIN", "INITIALIZING PREFERENCES")
            ed = sharedPrefs.edit()
            //Indicate that the default shared prefs have been set
            ed.putBoolean("INITIALIZED", true)
            //Set some default shared pref
            ed.putFloat("CENTER_CUTOFF", 0.5f)
            ed.commit()
        }
    }
}