package com.example.kt;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.HashMap;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import com.opencsv.CSVWriter;


public class GetGTDataActivity extends Activity
        implements OnClickListener {

    public String dir_name;
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        //setting activity view
        setContentView(R.layout.activity_gt);

        //get the Button reference
        View buttonClick = findViewById(R.id.FinishTestButton);
        //set event listener
        buttonClick.setOnClickListener(this);

        //skip button reference
        View skipButton = findViewById(R.id.SkipTestButton);
        skipButton.setOnClickListener(this);

        // get extras
        Bundle bundle = getIntent().getExtras();
        dir_name = bundle.getString("dir_name");
    }

    //override the OnClickListener interface method
    @Override
    public void onClick(View arg0) {
        if(arg0.getId() == R.id.FinishTestButton){

            // get right eye data
            final EditText editRightSph = (EditText) findViewById(R.id.EditRightSph);
            String rightSph = editRightSph.getText().toString();

            final EditText editRightCyl = (EditText) findViewById(R.id.EditRightCyl);
            String rightCyl = editRightCyl.getText().toString();

            final EditText editRightAxis = (EditText) findViewById(R.id.EditRightAxis);
            String rightAxis = editRightAxis.getText().toString();

            final EditText editRightKC = (EditText) findViewById(R.id.EditRightKC);
            String rightKC = editRightKC.getText().toString();

            // get left eye data
            final EditText editLeftSph = (EditText) findViewById(R.id.EditLeftSph);
            String leftSph = editLeftSph.getText().toString();

            final EditText editLeftCyl = (EditText) findViewById(R.id.EditLeftCyl);
            String leftCyl = editLeftCyl.getText().toString();

            final EditText editLeftAxis = (EditText) findViewById(R.id.EditLeftAxis);
            String leftAxis = editLeftAxis.getText().toString();

            final EditText editLeftKC = (EditText) findViewById(R.id.EditLeftKC);
            String leftKC = editLeftKC.getText().toString();


            boolean check_gt = leftSph.length()>0 && leftCyl.length()>0 && leftAxis.length()>0 && leftKC.length() >0
                    && rightSph.length()>0 && rightCyl.length()>0 && rightAxis.length()>0 && rightKC.length() > 0;

            String gt_data[] = {rightSph, rightCyl, rightAxis, rightKC, leftSph, leftCyl, leftAxis, leftKC};
            Log.e("TEST", "GT_DATA "+gt_data);

            if(!check_gt){
                Toast toast = Toast.makeText(this, "Enter the Patient Refractive Data", Toast.LENGTH_SHORT);
                toast.setGravity(Gravity.CENTER_VERTICAL, 0, 200);

                // set color for toast
                View view = toast.getView();
                TextView text = view.findViewById(android.R.id.message);
                text.setTextColor(Color.RED);
                toast.show();
            }

            else {
                // raise toast
                Toast toast = Toast.makeText(this, "Test Complete!", Toast.LENGTH_SHORT);
                toast.setGravity(Gravity.CENTER_VERTICAL, 0, 200);
                toast.show();
                //define a new Intent for the main Activity
                Intent intent = new Intent(this, MainActivity.class);
                //write data
                write_gtdata(gt_data, dir_name);
                //finish current activity
                finish();
                //start the second Activity
                this.startActivity(intent);
            }
        }
        else if(arg0.getId() == R.id.SkipTestButton){
            // raise toast
            Toast toast = Toast.makeText(this, "Test Complete!", Toast.LENGTH_SHORT);
            toast.setGravity(Gravity.CENTER_VERTICAL, 0, 200);
            toast.show();

            //define a new Intent for the main Activity
            Intent intent = new Intent(this, MainActivity.class);
            //finish current activity
            finish();
            //start the second Activity
            this.startActivity(intent);
        }
    }

    public void write_gtdata(String gt_data[], String dir_name){
        File dir = new File(Environment.getExternalStorageDirectory(), MainActivity.PACKAGE_NAME);
        if (!dir.exists()) {
            dir.mkdirs();
        }
        dir = new File(Environment.getExternalStorageDirectory(), MainActivity.PACKAGE_NAME+"/"+dir_name);
        if (!dir.exists()) {
            dir.mkdirs();
        }

        String filePath = Environment.getExternalStorageDirectory() + File.separator +
                MainActivity.PACKAGE_NAME + File.separator + dir_name + File.separator + dir_name+"_gt_data.csv";

        File f = new File(filePath);
        CSVWriter writer = null;

        // File exist
        if(f.exists() && !f.isDirectory())
        {
            try {
                FileWriter mFileWriter = new FileWriter(filePath, true);
                writer = new CSVWriter(mFileWriter);
            }catch (FileNotFoundException e) {
                System.err.print("File not found");
            }catch (IOException e) {
                System.err.print("Something went wrong");
            }
        }
        else
        {
            try {
                writer = new CSVWriter(new FileWriter(filePath));
            }catch (FileNotFoundException e) {
                System.err.print("File not found");
            }catch (IOException e) {
                System.err.print("Something went wrong");
            }
        }

        String[] data = {"Right_Sph", "Right_Cyl", "Right_Axis", "Right_KC", "Left_Sph", "Left_Cyl", "Left_Axis", "Left_KC"};
        writer.writeNext(data);
        try {
            writer.writeNext(gt_data);
            writer.close();
        }catch (Exception e) {
            e.printStackTrace();
        }
    }
}
