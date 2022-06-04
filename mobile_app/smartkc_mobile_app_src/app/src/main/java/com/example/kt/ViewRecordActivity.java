// Code reference: http://truelogic.org/wordpress/2016/05/01/showing-dynamic-data-in-a-tablelayout-in-android/
package com.example.kt;

import org.apache.commons.io.comparator.LastModifiedFileComparator;
import android.app.ProgressDialog;
import android.content.Intent;
import android.graphics.Color;
import android.os.AsyncTask;
import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;

import java.io.File;
import java.nio.file.attribute.FileTime;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.nio.file.Files;
import java.nio.file.LinkOption;

import java.util.Arrays;
import java.util.Date;

class PatientData {

    public int id;
    //public Date recordDate;
    public String recordDate;
    public String patientId;
    public String patientGender;
    public String patientAge;

}
class Records {

    public PatientData[] getRecords(String base_dir) {

        File dir = new File(Environment.getExternalStorageDirectory(), base_dir);
        File[] files = dir.listFiles();

        // if files dir empty
        if (files == null){
            PatientData[] data = new PatientData[1];
            PatientData row = new PatientData();
            row.id = (1);
            row.patientId = "NULL";
            row.patientAge = "NULL";
            row.patientGender = "NULL";
            //row.recordDate = new Date();
            row.recordDate = "NULL";
            data[0] = row;
            return data;
        }

        Arrays.sort(files, LastModifiedFileComparator.LASTMODIFIED_REVERSE);
        Log.d("Files", "Size: "+ files.length);

        PatientData[] data = new PatientData[files.length];
        for (int i = 0; i < files.length; i++)
        {
            Log.d("Files", "FileName:" + files[i].getName());
            String[] namesList = files[i].getName().split("_");
            PatientData row = new PatientData();
            row.id = (i+1);
            row.patientId = namesList[0];
            row.patientAge = namesList[1];
            row.patientGender = namesList[2];
            //row.recordDate = new SimpleDateFormat(files[i].lastModified());
            SimpleDateFormat sdf = new SimpleDateFormat("MM/dd/yyyy \n hh:mm:ss aa");
            row.recordDate =  sdf.format(files[i].lastModified());

            data[i] = row;
        }
        return data;

    }
}

public class ViewRecordActivity extends AppCompatActivity {

    private TableLayout mTableLayout;
    ProgressDialog mProgressBar;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_viewrecord);

        mProgressBar = new ProgressDialog(this);

        // setup the table
        mTableLayout = (TableLayout) findViewById(R.id.tableRecords);
        mTableLayout.setStretchAllColumns(true);

        //get view_records button reference
        View viewRecords = findViewById(R.id.back_button);
        viewRecords.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //define a new Intent for the second Activity
                Intent intent = new Intent(ViewRecordActivity.this, MainActivity.class);
                //finish current activity
                finish();
                //start the second Activity
                ViewRecordActivity.this.startActivity(intent);
            }
        });

        startLoadData();
    }


    public void startLoadData() {

        mProgressBar.setCancelable(false);
        mProgressBar.setMessage("Fetching Records..");
        mProgressBar.setProgressStyle(ProgressDialog.STYLE_SPINNER);
        mProgressBar.show();
        new LoadDataTask().execute(0);

    }

    public void loadData() {

        int leftRowMargin=0;
        int topRowMargin=0;
        int rightRowMargin=0;
        int bottomRowMargin = 0;
        int textSize = 0, smallTextSize = 0, mediumTextSize = 0;

        textSize = (int) getResources().getDimension(R.dimen.font_size_verysmall);
        smallTextSize = (int) getResources().getDimension(R.dimen.font_size_small);
        mediumTextSize = (int) getResources().getDimension(R.dimen.font_size_medium);

        Records invoices = new Records();
        PatientData[] data = invoices.getRecords(MainActivity.PACKAGE_NAME);
        SimpleDateFormat dateFormat = new SimpleDateFormat("dd MMM, yyyy");

        int rows = data.length;
        getSupportActionBar().setTitle("Total Records (" + String.valueOf(rows) + ")");
        TextView textSpacer = null;

        mTableLayout.removeAllViews();

        // -1 means heading row
        for(int i = -1; i < rows; i ++) {
            PatientData row = null;
            if (i > -1)
                row = data[i];
            else {
                textSpacer = new TextView(this);
                textSpacer.setText("");

            }
            // data columns
            final TextView tv = new TextView(this);
            tv.setLayoutParams(new TableRow.LayoutParams(TableRow.LayoutParams.WRAP_CONTENT,
                    TableRow.LayoutParams.WRAP_CONTENT));

            tv.setGravity(Gravity.LEFT);

            tv.setPadding(5, 15, 0, 15);
            if (i == -1) {
                tv.setText("Id#");
                tv.setBackgroundColor(Color.parseColor("#f0f0f0"));
                tv.setTextSize(TypedValue.COMPLEX_UNIT_PX, mediumTextSize);
            } else {
                tv.setBackgroundColor(Color.parseColor("#f8f8f8"));
                tv.setText(String.valueOf(row.id));
                tv.setTextSize(TypedValue.COMPLEX_UNIT_PX, smallTextSize);
            }

            final TextView tv2 = new TextView(this);
            if (i == -1) {
                tv2.setLayoutParams(new TableRow.LayoutParams(TableRow.LayoutParams.MATCH_PARENT,
                        TableRow.LayoutParams.WRAP_CONTENT));
                tv2.setTextSize(TypedValue.COMPLEX_UNIT_PX, mediumTextSize);
            } else {
                tv2.setLayoutParams(new TableRow.LayoutParams(TableRow.LayoutParams.WRAP_CONTENT,
                        TableRow.LayoutParams.MATCH_PARENT));
                tv2.setTextSize(TypedValue.COMPLEX_UNIT_PX, smallTextSize);
            }

            tv2.setGravity(Gravity.LEFT);

            tv2.setPadding(5, 15, 0, 15);
            if (i == -1) {
                tv2.setText("Date/Time");
                tv2.setBackgroundColor(Color.parseColor("#f7f7f7"));
            }else {
                tv2.setBackgroundColor(Color.parseColor("#ffffff"));
                tv2.setTextColor(Color.parseColor("#000000"));
                //tv2.setText(dateFormat.format(row.recordDate));
                tv2.setText(row.recordDate);
            }


            final LinearLayout layPatient = new LinearLayout(this);
            layPatient.setOrientation(LinearLayout.VERTICAL);
            layPatient.setPadding(0, 10, 0, 10);
            layPatient.setBackgroundColor(Color.parseColor("#f8f8f8"));

            final TextView tv3 = new TextView(this);
            if (i == -1) {
                tv3.setLayoutParams(new TableRow.LayoutParams(TableRow.LayoutParams.MATCH_PARENT,
                        TableRow.LayoutParams.MATCH_PARENT));
                tv3.setPadding(5, 5, 0, 5);
                tv3.setTextSize(TypedValue.COMPLEX_UNIT_PX, mediumTextSize);
            } else {
                tv3.setLayoutParams(new TableRow.LayoutParams(TableRow.LayoutParams.MATCH_PARENT,
                        TableRow.LayoutParams.MATCH_PARENT));
                tv3.setPadding(5, 0, 0, 5);
                tv3.setTextSize(TypedValue.COMPLEX_UNIT_PX, smallTextSize);
            }

            tv3.setGravity(Gravity.TOP);


            if (i == -1) {
                tv3.setText("Patient Id");
                tv3.setBackgroundColor(Color.parseColor("#f0f0f0"));
            } else {
                tv3.setBackgroundColor(Color.parseColor("#f8f8f8"));
                tv3.setTextColor(Color.parseColor("#000000"));
                tv3.setText(row.patientId);
            }
            layPatient.addView(tv3);


            if (i > -1) {
                final TextView tv3b = new TextView(this);
                tv3b.setLayoutParams(new TableRow.LayoutParams(TableRow.LayoutParams.WRAP_CONTENT,
                        TableRow.LayoutParams.WRAP_CONTENT));

                tv3b.setGravity(Gravity.RIGHT);
                tv3b.setTextSize(TypedValue.COMPLEX_UNIT_PX, smallTextSize);
                tv3b.setPadding(5, 1, 0, 5);
                tv3b.setTextColor(Color.parseColor("#aaaaaa"));
                tv3b.setBackgroundColor(Color.parseColor("#f8f8f8"));
                tv3b.setText("");
                layPatient.addView(tv3b);
            }


            final LinearLayout layAgeSex = new LinearLayout(this);
            layAgeSex.setOrientation(LinearLayout.VERTICAL);
            layAgeSex.setGravity(Gravity.RIGHT);
            layAgeSex.setPadding(0, 10, 0, 10);
            layAgeSex.setLayoutParams(new TableRow.LayoutParams(TableRow.LayoutParams.MATCH_PARENT,
                    TableRow.LayoutParams.MATCH_PARENT));

            final TextView tv4 = new TextView(this);
            if (i == -1) {
                tv4.setLayoutParams(new TableRow.LayoutParams(TableRow.LayoutParams.MATCH_PARENT,
                        TableRow.LayoutParams.MATCH_PARENT));
                tv4.setPadding(5, 5, 1, 5);
                layAgeSex.setBackgroundColor(Color.parseColor("#f7f7f7"));
            } else {
                tv4.setLayoutParams(new TableRow.LayoutParams(TableRow.LayoutParams.MATCH_PARENT,
                        TableRow.LayoutParams.WRAP_CONTENT));
                tv4.setPadding(5, 0, 1, 5);
                layAgeSex.setBackgroundColor(Color.parseColor("#ffffff"));
            }
            tv4.setGravity(Gravity.RIGHT);

            if (i == -1) {
                tv4.setText("Age/Sex");
                tv4.setBackgroundColor(Color.parseColor("#f7f7f7"));
                tv4.setTextSize(TypedValue.COMPLEX_UNIT_PX, mediumTextSize);
            } else {
                tv4.setBackgroundColor(Color.parseColor("#ffffff"));
                tv4.setTextColor(Color.parseColor("#000000"));
                tv4.setText(row.patientAge);
                tv4.setTextSize(TypedValue.COMPLEX_UNIT_PX, smallTextSize);
            }
            layAgeSex.addView(tv4);


            if (i > -1) {
                final TextView tv4b = new TextView(this);
                tv4b.setLayoutParams(new TableRow.LayoutParams(TableRow.LayoutParams.MATCH_PARENT,
                        TableRow.LayoutParams.WRAP_CONTENT));

                tv4b.setGravity(Gravity.RIGHT);
                tv4b.setTextSize(TypedValue.COMPLEX_UNIT_PX, textSize);
                tv4b.setPadding(2, 2, 1, 5);
                tv4b.setTextColor(Color.parseColor("#00afff"));
                tv4b.setBackgroundColor(Color.parseColor("#ffffff"));
                tv4b.setText(row.patientGender);
                layAgeSex.addView(tv4b);
            }


            // add table row
            final TableRow tr = new TableRow(this);
            tr.setId(i + 1);
            TableLayout.LayoutParams trParams = new TableLayout.LayoutParams(TableLayout.LayoutParams.MATCH_PARENT,
                    TableLayout.LayoutParams.WRAP_CONTENT);
            trParams.setMargins(leftRowMargin, topRowMargin, rightRowMargin, bottomRowMargin);
            tr.setPadding(0,0,0,0);
            tr.setLayoutParams(trParams);

            tr.addView(tv);
            tr.addView(tv2);
            tr.addView(layPatient);
            tr.addView(layAgeSex);

            if (i > -1) {
                tr.setOnClickListener(new View.OnClickListener() {
                    public void onClick(View v) {
                        TableRow tr = (TableRow) v;
                        //do whatever action is needed

                    }
                });
            }
            mTableLayout.addView(tr, trParams);

            if (i > -1) {

                // add separator row
                final TableRow trSep = new TableRow(this);
                TableLayout.LayoutParams trParamsSep = new TableLayout.LayoutParams(TableLayout.LayoutParams.MATCH_PARENT,
                        TableLayout.LayoutParams.WRAP_CONTENT);
                trParamsSep.setMargins(leftRowMargin, topRowMargin, rightRowMargin, bottomRowMargin);

                trSep.setLayoutParams(trParamsSep);
                TextView tvSep = new TextView(this);
                TableRow.LayoutParams tvSepLay = new TableRow.LayoutParams(TableRow.LayoutParams.MATCH_PARENT,
                        TableRow.LayoutParams.WRAP_CONTENT);
                tvSepLay.span = 4;
                tvSep.setLayoutParams(tvSepLay);
                tvSep.setBackgroundColor(Color.parseColor("#d9d9d9"));
                tvSep.setHeight(1);
                trSep.addView(tvSep);
                mTableLayout.addView(trSep, trParamsSep);
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////////

    //
    // The params are dummy and not used
    //
    class LoadDataTask extends AsyncTask<Integer, Integer, String> {
        @Override
        protected String doInBackground(Integer... params) {

            try {
                Thread.sleep(2000);

            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            return "Task Completed.";
        }
        @Override
        protected void onPostExecute(String result) {
            mProgressBar.hide();
            loadData();
        }
        @Override
        protected void onPreExecute() {
        }
        @Override
        protected void onProgressUpdate(Integer... values) {

        }
    }

}