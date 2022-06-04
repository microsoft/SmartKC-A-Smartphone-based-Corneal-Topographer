package com.example.kt;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;

import com.github.chrisbanes.photoview.PhotoView;
import com.opencsv.CSVWriter;

import org.apache.commons.io.comparator.LastModifiedFileComparator;
import org.jetbrains.annotations.NotNull;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.HashMap;

import kotlin.jvm.internal.Intrinsics;
import kotlin.ranges.IntProgression;
import kotlin.ranges.IntRange;
import kotlin.ranges.RangesKt;

import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

public class CheckImages extends Activity implements OnClickListener {

    @NotNull
    private final ImageCheck image_checker = new ImageCheck();
    @NotNull
    public final ImageCheck getImage_checker() {
        return this.image_checker;
    }

    /** Called when the activity is first created. */
    private int image_index;
    private int maxCounts;
    private int origMaxCounts;
    public File[] imageFiles;
    // offset distance measure
    double offset_distance = -1.0;

    //public class variables
    public String dir_name;
    public String left_right;
    public HashMap<String, String> hash_map;
    public float centerCutoff = 1.0F;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_checkimages);

        // retrieve getExtras
        Bundle bundle = getIntent().getExtras();
        dir_name = bundle.getString("dir_name");
        left_right = bundle.getString("left_right");
        hash_map =  (HashMap<String, String>)getIntent().getSerializableExtra("hash_map");
        origMaxCounts = Integer.parseInt(bundle.getString("number_of_images"));
        image_index = 0; // initialize image index as 0

        // get app preferences
        SharedPreferences sharedPrefs = this.getSharedPreferences("KT_APP_PREFERENCES", Context.MODE_PRIVATE);
        centerCutoff = sharedPrefs.getFloat("CENTER_CUTOFF", 0.5F);

        // list images in the directory
        File dir =  new File(Environment.getExternalStorageDirectory(), MainActivity.PACKAGE_NAME+"/"+dir_name);
        imageFiles = dir.listFiles(new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return name.toLowerCase().startsWith(left_right);
            }
        });
        Arrays.sort(imageFiles, LastModifiedFileComparator.LASTMODIFIED_REVERSE);
        // incase there are less number of images than maxCounts
        maxCounts = Math.min(imageFiles.length, origMaxCounts);
        //Log.e("Counts", "maxCounts "+maxCounts);

        // previous
        Button btnNo = (Button)findViewById(R.id.no_btn);
        btnNo.setOnClickListener(this);

        // next
        Button btnYes = (Button)findViewById(R.id.yes_btn);
        btnYes.setOnClickListener(this);

        // remove view
        removeView();
        // show image
        //showImage();
        AsyncTask.execute(new Runnable() {
            @Override
            public void run() {
                showImage();
            }
        });
    }


    private void showImage() {

        //PhotoView imgView = (PhotoView) findViewById(R.id.myimage);
        //imgView.setImageURI(Uri.fromFile(imageFiles[image_index]));

        // check if file is empty
        boolean fileEmpty = imageFiles[image_index].exists() && imageFiles[image_index].length() == 0;
        if (fileEmpty) {
            // emulate click
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    pressNo();
                }
            });
            return;
        }
        else {
            // Find and display center, check if No then disable Yes
            boolean check = false;
            try {
                check = checkCenter(imageFiles[image_index].toString(), centerCutoff);
            }catch (Exception e){
                runOnUiThread(new Runnable() {
                                  @Override
                                  public void run() {
                                      pressNo();
                                  }
                              });
                e.printStackTrace();
                Log.e("CHECK_IMAGES", "Center not found or empty image!");
                return;
            }

            if (!check) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Button btnYes = (Button) findViewById(R.id.yes_btn);
                        btnYes.setClickable(false);
                        btnYes.setVisibility(View.GONE);

                        TextView questionView = findViewById(R.id.questionView);
                        questionView.setText("The image is not good!");
                    }
                });
            }

            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    // set prompt text and Image view
                    TextView textView = (TextView) findViewById(R.id.textView);
                    if (left_right.equals("right"))
                        textView.setText("Right Eye: Image " + (image_index + 1) + "/" + maxCounts);
                    else
                        textView.setText("Left Eye: Image " + (image_index + 1) + "/" + maxCounts);
                }
            });

            return;
        }
    }

    public void onClick(View v) {

        switch (v.getId()) {

            case (R.id.no_btn):
                hash_map.put(imageFiles[image_index].getName(), "No");
                hash_map.put(imageFiles[image_index].getName()+"_offset", ""+offset_distance);
                image_index += 1;
                if (image_index >= maxCounts) {
                    // if all images clicked NO then REDO
                    Intent intent = new Intent(this, CameraActivityNew.class);
                    // add extras to intent
                    intent.putExtra("dir_name", dir_name);
                    intent.putExtra("left_right", left_right);
                    intent.putExtra("number_of_images", ""+origMaxCounts);
                    intent.putExtra("hash_map", hash_map);
                    //finish current activity
                    finish();
                    //start the second Activity
                    this.startActivity(intent);
                }
                else {
                    removeView();
                    AsyncTask.execute(new Runnable() {
                        @Override
                        public void run() {
                            showImage();
                        }
                    });

                }
                break;

            case (R.id.yes_btn):

                hash_map.put(imageFiles[image_index].getName(), "Yes");
                hash_map.put(imageFiles[image_index].getName()+"_offset", ""+offset_distance);
                Intent intent = new Intent(this, GetGTDataActivity.class);

                if (left_right.equals("right")) {
                    intent = new Intent(this, PromptActivity.class);
                    // add extras to intent
                    intent.putExtra("dir_name", dir_name);
                    intent.putExtra("left_right", "left"); // move to left
                    intent.putExtra("number_of_images", ""+origMaxCounts);
                    intent.putExtra("hash_map", hash_map);
                } else if (left_right.equals("left")){
                    // write the meta data file since both left and right complete
                    write_metadata();
                    // add extras to intent
                    intent.putExtra("dir_name", dir_name);
                    }

                //finish current activity
                finish();
                //start the second Activity
                this.startActivity(intent);

                break;
        }
    }

    private void pressNo(){
        View noButtomView = findViewById(R.id.no_btn);
        noButtomView.performClick();
        return;
    }

    private void removeView(){

        // remove photo
        PhotoView imgView = (PhotoView) findViewById(R.id.myimage);
        imgView.setVisibility(View.GONE);
        // remove Yes button
        Button btnYes = (Button) findViewById(R.id.yes_btn);
        btnYes.setClickable(false);
        btnYes.setVisibility(View.GONE);
        // remove No button
        Button btnNo = (Button) findViewById(R.id.no_btn);
        btnNo.setClickable(false);
        btnNo.setVisibility(View.GONE);
        // set loading text
        TextView questionView = findViewById(R.id.questionView);
        questionView.setText("Analyzing Image " + (image_index+1) + " ...");
        // remove pinch view
        TextView pinchView = findViewById(R.id.pinchView);
        pinchView.setVisibility(View.GONE);
        // remove text view
        TextView textView = findViewById(R.id.textView);
        textView.setVisibility(View.GONE);
    }

    private void addView(){

        // remove photo
        PhotoView imgView = (PhotoView) findViewById(R.id.myimage);
        imgView.setVisibility(View.VISIBLE);
        // remove Yes button
        Button btnYes = (Button) findViewById(R.id.yes_btn);
        btnYes.setClickable(true);
        btnYes.setVisibility(View.VISIBLE);
        // remove No button
        Button btnNo = (Button) findViewById(R.id.no_btn);
        btnNo.setClickable(true);
        btnNo.setVisibility(View.VISIBLE);
        // set loading text
        TextView questionView = findViewById(R.id.questionView);
        questionView.setText("Is the image in focus and sharp?");
        // remove pinch view
        TextView pinchView = findViewById(R.id.pinchView);
        pinchView.setVisibility(View.VISIBLE);
        // remove text view
        TextView textView = findViewById(R.id.textView);
        textView.setVisibility(View.VISIBLE);
    }

    public void write_metadata(){
        File dir = new File(Environment.getExternalStorageDirectory(), MainActivity.PACKAGE_NAME+"/"+dir_name);
        File[] files = dir.listFiles();

        Arrays.sort(files, LastModifiedFileComparator.LASTMODIFIED_REVERSE);
        Log.e("Finish", "Inside Write Metadata Size: "+ files.length);

        String filePath = Environment.getExternalStorageDirectory() + File.separator +
                MainActivity.PACKAGE_NAME + File.separator + dir_name + File.separator + dir_name+".csv";
        Log.e("Finish", "Inside Write Metadata filePath "+ filePath);
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

        String[] data = {"Patient Id", "Age", "Gender", "App Version", "Image Name", "Date", "Time", "Is OK", "Offset", "Cutoff"};
        writer.writeNext(data);
        try {

            //List<String> letters = Arrays.asList(string.split(""));
            for (int i = 0; i < files.length; i++) {
                Log.d("Files", "FileName:" + files[i].getName());
                String[] namesList = dir_name.split("_");
                int id = (i + 1);
                String patientId = namesList[0];
                String patientAge = namesList[1];
                String patientGender = namesList[2];
                String isOk = hash_map.containsKey(files[i].getName())? hash_map.get(files[i].getName()) : "NA";
                String offset = hash_map.containsKey(files[i].getName()+"_offset")? hash_map.get(files[i].getName()+"_offset") : "NA";
                SimpleDateFormat sdf = new SimpleDateFormat("MM/dd/yyyy, hh:mm:ss aa");
                String[] date_time = sdf.format(files[i].lastModified()).split(",");
                String[] curr_data = {patientId, patientAge, patientGender, MainActivity.VERSION, files[i].getName(), date_time[0], date_time[1], isOk, offset, String.valueOf(centerCutoff)};
                writer.writeNext(curr_data);
            }
            writer.close();
        }catch (Exception e) {
            e.printStackTrace();
        }
    }

    private boolean checkCenter(String currImage, Float centerThresh){

        //String imageString = imageFile.toString();
        Bitmap bitmap = BitmapFactory.decodeFile(currImage);

        File imageFile = new File(currImage);
        // get Exif data for rotation
        try {
            ExifInterface exif = new ExifInterface(imageFile.getAbsolutePath());
            int rotation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            int rotationInDegrees = exifToDegrees(rotation);
            Matrix matrix = new Matrix();
            if (rotation != 0f) {
                matrix.preRotate(rotationInDegrees);
                bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
            }
        }
        catch(Exception e){
            e.printStackTrace();
        }

        Mat imageFull = new Mat();
        Utils.bitmapToMat(bitmap, imageFull);
        Mat image = new Mat();
        Imgproc.resize(imageFull, image, new Size(480*4, 640*4) , 0, 0, Imgproc.INTER_LINEAR);

        int zoom_factor = 2;
        // add multiplying factor & choosing parameters for detecting circles based on 3000x4000 image
        float basewidth = 3000.0F, baseheight = 4000.0F;
        float baseminDist = 1200.0F*zoom_factor;
        float baseminR = 595.0F*zoom_factor, basemaxR = 615.0F*zoom_factor;
        double normfactor = sqrt((basewidth * basewidth + baseheight * baseheight) /
                (image.cols() * image.cols() + image.rows() * image.rows()));
        Float[] crosshair_center = detectCrossHair(image.clone(), 2.5,
                baseminDist / normfactor, (int)(baseminR/normfactor)-5, (int)(basemaxR/normfactor)+5);

        int start = (int)(30*zoom_factor/normfactor), end = (int)(100*zoom_factor/normfactor);
        int jump = (int)(10*zoom_factor/normfactor);
        Float[] mire_center = detectMireCenter(image.clone(), 2.5, baseminDist / normfactor, start, end, jump);

        // draw cross_hair
        try {
            Core.line(image, new Point(crosshair_center[0]-25, crosshair_center[1]), new Point(crosshair_center[0]+25, crosshair_center[1]), new Scalar(0.0, 0.0, 255.0), 4);
            Core.line(image, new Point(crosshair_center[0], crosshair_center[1]-25), new Point(crosshair_center[0], crosshair_center[1]+25), new Scalar(0.0, 0.0, 255.0), 4);
        }catch (Exception e) {
            e.printStackTrace();
        }

        try{
            Core.circle(image, new Point(mire_center[0], mire_center[1]), 15, new Scalar(0.0, 255.0, 0.0), -1, 8, 0);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }

        // distance b/w mireCenter and crossHair
        double dist = sqrt(pow(crosshair_center[0]-mire_center[0], 2)+pow(crosshair_center[1]-mire_center[1], 2));
        //boolean check1 = dist < (centerThresh*zoom_factor/normfactor); // check 1: if the offset is within threshold
        offset_distance = dist * 30 / (2*crosshair_center[2]+0.000001); // update offset distance in mm
        boolean check1 = offset_distance <= centerThresh; // check 1: if the offset is within threshold in mm
        //Log.e("Counts", "CENTER_CUTOFF_PREFERENCE "+centerThresh+" Offset "+offset_distance+" dist in pixels "+dist);
        //image = this.image_checker.sharpen(image);
        //image = this.image_checker.autoCanny(image, 0.33);

        // check 2: image is neither over-exposed or under-exposed
        Mat gray = image.clone();
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_RGB2GRAY); // converting image to grayscale
        gray = image_checker.cropResultWindow(gray, (int)(500*zoom_factor/normfactor),
                new Point(mire_center[0], mire_center[1]));
        ExposureResult exposure = image_checker.checkExposure(gray.clone());
        boolean check2 = exposure == ExposureResult.NORMAL;
        // check 3: image is sharp enough
        boolean check3 = image_checker.checkSharpness(gray, 0.96);

        Bitmap bitmapSmall = Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(image, bitmapSmall);
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                addView();
                PhotoView imgView = (PhotoView) findViewById(R.id.myimage);
                imgView.setImageBitmap(bitmapSmall);

            }
        });

        return check1 && check2 && check3;
    }

    // this code was auto de-compiled to JAVA from Kotlin (have to re-factor this)
    private final Float[] detectCrossHair(Mat image, double dp, double minDist, int minR, int maxR) {
        Float[] output_points = new Float[]{-1.0F, -1.0F, 0.0F};
        Mat image_sharpened = this.image_checker.sharpen(image);
        //Imgproc.medianBlur(image_sharpened, image_sharpened, 5);
        Float[] detected_circles = this.image_checker.detectCircles(image_sharpened, dp, minDist, minR, maxR);
        float xx = detected_circles[0], yy = detected_circles[1], rad = detected_circles[2];
        float image_xx = (float)image.cols() / (float)2, image_yy = (float)image.rows() / (float)2;
        if (xx > image_xx - (float)75 && xx < image_xx + (float)75 && yy > image_yy - (float)100 && yy < image_yy + (float)100) {
            output_points[0] = xx;
            output_points[1] = yy;
            output_points[2] = rad;
            return output_points;
        }
        return output_points;
    }

    private final Float[] detectMireCenter(Mat image, double dp, double minDist, int minR, int maxR, int jump) {
        Float[] output_points =  new Float[]{-1.0F, -1.0F};
        float xx=0.0F, yy=0.0F;
        int rCount = 0;
        for(int currRadius = minR; currRadius <= maxR; currRadius += jump){
            Float[] detected_circles = this.image_checker.detectCircles(image.clone(), dp, minDist, currRadius, currRadius + jump);
            if (detected_circles[2] > 0) {
                xx += detected_circles[0];
                yy += detected_circles[1];
                rCount += 1;
            }
        }

        xx /= (float)rCount; yy /= (float)rCount;

        output_points[0] = xx;
        output_points[1] = yy;
        return output_points;
    }

    private int exifToDegrees(int exifOrientation) {
        if (exifOrientation == ExifInterface.ORIENTATION_ROTATE_90) { return 90; }
        else if (exifOrientation == ExifInterface.ORIENTATION_ROTATE_180) {  return 180; }
        else if (exifOrientation == ExifInterface.ORIENTATION_ROTATE_270) {  return 270; }
        return 0;
    }

    /*
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