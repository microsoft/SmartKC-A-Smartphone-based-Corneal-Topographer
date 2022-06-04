package com.example.kt;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;

import com.balsikandar.crashreporter.CrashReporter;

import java.io.File;


//implement the OnClickListener interface
public class MainActivity extends Activity
        implements OnClickListener {

    /** Called when the activity is first created. */
    private static final int REQUEST_CAMERA_PERMISSION = 200;
    public static String PACKAGE_NAME;
    public static String VERSION;
    static{ System.loadLibrary("opencv_java");}

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        //setting activity view
        setContentView(R.layout.activity_main);
        PACKAGE_NAME = getApplicationInfo().loadLabel(getPackageManager()).toString();
        VERSION="14";

        // setting app preferences
        setPreferences();

        // Add permission for camera and let user grant the permission
        // Checking for Camera, Write / Read Permissions
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE}, REQUEST_CAMERA_PERMISSION);
        }

        //get the Button reference
        View buttonClick = findViewById(R.id.new_button);
        buttonClick.setOnClickListener(this); //set event listener

        //get view_records button reference
        View viewRecords = findViewById(R.id.view_records);
        viewRecords.setOnClickListener(this);

        //get update preferences button reference
        View updatePrefs = findViewById(R.id.update_preferences);
        updatePrefs.setOnClickListener(this);


        File dir = new File(Environment.getExternalStorageDirectory(), MainActivity.PACKAGE_NAME);
        if (!dir.exists()) {
            dir.mkdirs();
        }
        // create directory for testing crash logs
        dir = new File(Environment.getExternalStorageDirectory(), MainActivity.PACKAGE_NAME+"/"+"test_crash_logs");
        if (!dir.exists()) {
            dir.mkdirs();
        }
        // this is to log crash reports (used external lib)
        CrashReporter.initialize(this, dir.toString());
    }

    //override the OnClickListener interface method
    @Override
    public void onClick(View arg0) {
        if(arg0.getId() == R.id.new_button){
            //define a new Intent for the second Activity
            Intent intent = new Intent(this, DataActivity.class);
            //finish current activity
            finish();
            //start the second Activity
            this.startActivity(intent);
        }
        else if(arg0.getId() == R.id.view_records){

            //define a new Intent for the second Activity
            Intent intent = new Intent(this, ViewRecordActivity.class);
            //finish current activity
            finish();
            //start the second Activity
            this.startActivity(intent);

        }
        else if(arg0.getId() == R.id.update_preferences){

            //define a new Intent for the second Activity
            Intent intent = new Intent(this, UpdatePreferences.class);
            //finish current activity
            finish();
            //start the second Activity
            this.startActivity(intent);

        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
                // close the app
                Toast.makeText(this, "Sorry!!!, you can't use this app without granting permission", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }

    public void setPreferences(){
        // setting app preferences
        SharedPreferences sharedPrefs = getSharedPreferences("KT_APP_PREFERENCES", MODE_PRIVATE);
        // initialize for the first time
        SharedPreferences.Editor ed;
        if(!sharedPrefs.contains("INITIALIZED")){
            Log.e("MAIN", "INITIALIZING PREFERENCES");
            ed = sharedPrefs.edit();
            //Indicate that the default shared prefs have been set
            ed.putBoolean("INITIALIZED", true);
            //Set some default shared pref
            ed.putFloat("CENTER_CUTOFF", 0.5F);
            ed.commit();
        }
    }
}