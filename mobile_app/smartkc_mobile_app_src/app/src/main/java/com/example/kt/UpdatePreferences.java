package com.example.kt;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.widget.EditText;
import android.widget.Toast;

public class UpdatePreferences extends Activity
            implements View.OnClickListener {

        @Override
        public void onCreate(Bundle savedInstanceState) {
            super.onCreate(savedInstanceState);
            //setting activity view
            setContentView(R.layout.activity_update);
            //get the Button reference
            View buttonClick = findViewById(R.id.SavePreference);
            //set event listener
            buttonClick.setOnClickListener(this);

            // update EditText hint with the current cutoff value
            SharedPreferences sharedPrefs = this.getSharedPreferences("KT_APP_PREFERENCES", Context.MODE_PRIVATE);
            float current_cutoff = sharedPrefs.getFloat("CENTER_CUTOFF", 0.5F);
            final EditText editCenterCutoff = (EditText) findViewById(R.id.EditPreference);
            editCenterCutoff.setHint(String.valueOf(current_cutoff));
        }

        //override the OnClickListener interface method
        @Override
        public void onClick(View arg0) {
            if(arg0.getId() == R.id.SavePreference){

                // raise toast
                Toast toast = Toast.makeText(this, "Data Saved!", Toast.LENGTH_SHORT);
                toast.setGravity(Gravity.CENTER_VERTICAL, 0, 200);
                toast.show();

                // get SharedPreferences object
                SharedPreferences sharedPrefs = this.getSharedPreferences("KT_APP_PREFERENCES", Context.MODE_PRIVATE);

                // read data from the form
                final EditText editCenterCutoff = (EditText) findViewById(R.id.EditPreference);
                float centerCutoff = sharedPrefs.getFloat("CENTER_CUTOFF", 0.5F); // initialize as the current cutoff
                try{
                    centerCutoff = Float.parseFloat(editCenterCutoff.getText().toString());
                }catch(Exception e){
                    e.printStackTrace();
                }

                // center cutoff check
                if(centerCutoff > 1.0)
                    centerCutoff = 1.0F;
                else if(centerCutoff < 0.0)
                    centerCutoff = 0.5F;

                // update SharedPreferences here
                SharedPreferences.Editor editor = sharedPrefs.edit();
                editor.putFloat("CENTER_CUTOFF", centerCutoff);
                editor.apply();

                //define a new Intent for the second Activity
                Intent intent = new Intent(this, MainActivity.class);
                //finish current activity
                finish();
                //start the second Activity
                this.startActivity(intent);
            }
        }

    }
