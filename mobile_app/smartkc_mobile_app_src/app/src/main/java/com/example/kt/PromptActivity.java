package com.example.kt;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.ImageView;

import java.util.HashMap;

public class PromptActivity extends Activity
        implements OnClickListener{

    //class variables
    public String dir_name;
    public String left_right;
    public String number_of_images;
    public HashMap<String, String> hash_map;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // retrieve getExtras
        Bundle bundle = getIntent().getExtras();
        dir_name = bundle.getString("dir_name");
        left_right = bundle.getString("left_right");
        number_of_images = bundle.getString("number_of_images");
        // there may be a better way to do this
        hash_map =  (HashMap<String, String>)getIntent().getSerializableExtra("hash_map");

        //setting activity view
        setContentView(R.layout.activity_prompt);

        // set prompt text and Image view
        ImageView simpleImageView=(ImageView) findViewById(R.id.simpleImageView);
        if (left_right.equals("right")){
            simpleImageView.setImageResource(R.drawable.right_prompt);
        }
        else if(left_right.equals("left")){
            simpleImageView.setImageResource(R.drawable.left_prompt);
        }

        //get the Button reference
        View buttonClick = findViewById(R.id.goButton);
        //set event listener
        buttonClick.setOnClickListener(this);
    }

    //override the OnClickListener interface method
    @Override
    public void onClick(View arg0) {
        if(arg0.getId() == R.id.goButton){

            Intent intent = new Intent(this, CameraActivityNew.class);
            // add extras to intent
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
