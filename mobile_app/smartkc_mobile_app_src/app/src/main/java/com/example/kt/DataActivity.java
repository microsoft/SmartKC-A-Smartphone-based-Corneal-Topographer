// Reference Link: http://www.fampennings.nl/maarten/android/09keyboard/index.htm
package com.example.kt;

import java.util.HashMap;
import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;


public class DataActivity extends Activity
implements OnClickListener{

    //CustomKeyboard mCustomKeyboard;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        //setting activity view
        setContentView(R.layout.activity_data);

        // initialize custom keyboard
        //mCustomKeyboard= new CustomKeyboard(this, R.id.keyboardview, R.xml.hexkbd );
        //mCustomKeyboard.registerEditText(R.id.EditPatientId);
        //mCustomKeyboard.registerEditText(R.id.EditPatientAge);

        //get the Button reference
        View buttonClick = findViewById(R.id.StartTestButton);
        //set event listener
        buttonClick.setOnClickListener(this);
    }

    //override the OnClickListener interface method
    @Override
    public void onClick(View arg0) {
        if(arg0.getId() == R.id.StartTestButton){

            // read data from the form
            final EditText patientId = (EditText) findViewById(R.id.EditPatientId);
            String patId = patientId.getText().toString();

            final EditText patientAge = (EditText) findViewById(R.id.EditPatientAge);
            String patAge = patientAge.getText().toString();

            final Spinner feedbackSpinner = (Spinner) findViewById(R.id.SpinnerGenderType);
            String patGender = feedbackSpinner.getSelectedItem().toString();



            // checks for fields
            if(patId.length() == 0) {
                Toast toast = Toast.makeText(this, "Enter the Patient Id", Toast.LENGTH_SHORT);
                toast.setGravity(Gravity.CENTER_VERTICAL, 0, 200);

                // set color for toast
                View view = toast.getView();
                TextView text = view.findViewById(android.R.id.message);
                text.setTextColor(Color.RED);
                toast.show();
            }
            else if(patAge.length() == 0){
                Toast toast = Toast.makeText(this, "Enter the Patient Age", Toast.LENGTH_SHORT);
                toast.setGravity(Gravity.CENTER_VERTICAL, 0, 200);

                // set color for toast
                View view = toast.getView();
                TextView text = view.findViewById(android.R.id.message);
                text.setTextColor(Color.RED);
                toast.show();
            }

            else {

                //define a new Intent for the second Activity
                Intent intent = new Intent(this, PromptActivity.class);
                String dir_name = patId + "_" + patAge + "_" + patGender;
                String left_right = "right";
                String number_of_images = "3";
                HashMap<String, String> hash_map = new HashMap<>();

                //track current variables
                intent.putExtra("dir_name", dir_name);
                intent.putExtra("left_right", left_right);
                intent.putExtra("number_of_images", number_of_images);
                intent.putExtra("hash_map", hash_map);

                //finish current activity
                finish();
                //start the second Activity
                this.startActivity(intent);
            }
        }
    }



    /*
    @Override public void onBackPressed() {
        // NOTE Trap the back key: when the CustomKeyboard is still visible hide it, only when it is invisible, finish activity
        if( mCustomKeyboard.isCustomKeyboardVisible() ) mCustomKeyboard.hideCustomKeyboard(); else this.finish();
    }
     */

}
