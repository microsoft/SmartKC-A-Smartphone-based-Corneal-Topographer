package com.example.kt

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.*
import android.widget.AdapterView.OnItemSelectedListener
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.*
import androidx.lifecycle.lifecycleScope
import com.example.kt.utils.PreferenceKeys
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject

@AndroidEntryPoint
class UpdatePreferences : AppCompatActivity(), View.OnClickListener {

    // DEFAULT VALUES
    val DEFAULT_CENTER_CUTOFF = 0.5f

    lateinit var centerCutoffEt: EditText
    lateinit var uploadUrlEt: EditText
    lateinit var uploadSecretEt: EditText
    lateinit var uploadSwitch: SwitchCompat
    lateinit var spinner: Spinner

    @Inject
    lateinit var dataStore: DataStore<Preferences>

    //get preference keys
    val centerCutoffKey = floatPreferencesKey(PreferenceKeys.CENTER_CUTOFF)
    val uploadUrlKey = stringPreferencesKey(PreferenceKeys.UPLOAD_URL)
    val uploadSecretKey = stringPreferencesKey(PreferenceKeys.UPLOAD_SECRET)
    val uploadEnabledKey = booleanPreferencesKey(PreferenceKeys.UPLOAD_ENABLED)
    val selectedCameraKey = stringPreferencesKey(PreferenceKeys.CHOSEN_CAMERA)

    public override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        //setting activity view
        setContentView(R.layout.activity_update)

        // Get view references
        centerCutoffEt = findViewById(R.id.update_center_cutoff_et)
        uploadUrlEt = findViewById(R.id.update_upload_url_et)
        uploadSecretEt = findViewById(R.id.update_upload_secret_et)
        uploadSwitch = findViewById(R.id.upload_switch)
        spinner = findViewById(R.id.camera_list_spinner)

        // Get list of logical cameras
        val cameraManager: CameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val cameraIds = cameraManager.cameraIdList.filter { id ->
            cameraManager.getCameraCharacteristics(id).get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
        }

        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, cameraIds)
        spinner.adapter = adapter

        //get the Button reference
        val buttonClick = findViewById<View>(R.id.SavePreference)
        //set event listener
        buttonClick.setOnClickListener(this)
        //set event listener for the switch
        uploadSwitch.setOnCheckedChangeListener { _, checked ->
            modifyUploadView(checked)
        }

        // update EditText with the current cutoff and upload URL value
        lifecycleScope.launch {
            val data = dataStore.data.first()
            val centerCutOff = data[centerCutoffKey] ?: DEFAULT_CENTER_CUTOFF
            val uploadUrl = data[uploadUrlKey] ?: BuildConfig.UPLOAD_URL
            val uploadSecret = data[uploadSecretKey] ?: ""
            val uploadEnabled = data[uploadEnabledKey] ?: BuildConfig.UPLOAD_ENABLED
            val selectedCamera = data[selectedCameraKey]

            centerCutoffEt.setText(centerCutOff.toString())
            uploadUrlEt.setText(uploadUrl)
            uploadSecretEt.setText(uploadSecret)
            uploadSwitch.isChecked = uploadEnabled
            if(selectedCamera != null) spinner.setSelection(adapter.getPosition(selectedCamera))
            modifyUploadView(uploadEnabled)
        }
    }

    // Manage view based on the uploadEnabled value
    private fun modifyUploadView(uploadEnabled: Boolean) {
        // If unchecked hide URL and secret inputs
        if (!uploadEnabled) {
            // Hide URL input
            uploadUrlEt.visibility = View.GONE
            findViewById<TextView>(R.id.update_upload_url_label).visibility = View.GONE
            // Hide upload secret input
            uploadSecretEt.visibility = View.GONE
            findViewById<TextView>(R.id.update_upload_secret_label).visibility = View.GONE
        } else {
            // Show URL input
            uploadUrlEt.visibility = View.VISIBLE
            findViewById<TextView>(R.id.update_upload_url_label).visibility = View.VISIBLE
            // Show upload secret input
            uploadSecretEt.visibility = View.VISIBLE
            findViewById<TextView>(R.id.update_upload_secret_label).visibility = View.VISIBLE
        }
    }

    //override the OnClickListener interface method
    override fun onClick(arg0: View) {
        if (arg0.id == R.id.SavePreference) {

            lifecycleScope.launch {
                val centerCutOffInput = centerCutoffEt.text.toString()
                val uploadUrlInput = uploadUrlEt.text.toString()
                val uploadSecret = uploadSecretEt.text.toString()
                val uploadEnabled = uploadSwitch.isChecked
                val selectedCamera = spinner.selectedItem as String

                if (centerCutOffInput.isNullOrEmpty() || uploadEnabled && (uploadUrlInput.isNullOrEmpty() || uploadSecret.isNullOrEmpty())) {
                    Toast.makeText(applicationContext, "Please provide input in all the fields", Toast.LENGTH_SHORT).show()
                    return@launch
                }
                var centerCutOff = centerCutOffInput.toFloat()
                // center cutoff check
                if (centerCutOff > 1.0f) centerCutOff = 1.0f else if (centerCutOff < 0.0) centerCutOff =
                    0.5f

                // Save in datastore
                dataStore.edit { prefs -> prefs[centerCutoffKey] = centerCutOff }
                dataStore.edit { prefs -> prefs[uploadUrlKey] = uploadUrlInput }
                dataStore.edit { prefs -> prefs[uploadSecretKey] = uploadSecret }
                dataStore.edit { prefs -> prefs[uploadEnabledKey] = uploadEnabled }
                dataStore.edit { prefs -> prefs[selectedCameraKey] = selectedCamera }

                // raise toast
                val toast = Toast.makeText(applicationContext, "Data Saved!", Toast.LENGTH_SHORT)
                toast.setGravity(Gravity.CENTER_VERTICAL, 0, 200)
                toast.show()

            }

        }
    }
}