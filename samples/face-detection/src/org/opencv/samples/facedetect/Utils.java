package org.opencv.samples.facedetect;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;

import static android.os.Environment.getExternalStorageDirectory;

public class Utils {

    public static void setLog(final String str) {
        Runnable r = new Runnable() {
            @Override
            public void run() {
                MainActivity.index ++;
                try {
                    final FileWriter fileOut = new FileWriter( new File(getExternalStorageDirectory() +"/Download/Log.txt"), true );
                    fileOut.append(MainActivity.index + ":" + str);
                    fileOut.close();
                }
                catch ( final IOException e ) {
                    e.printStackTrace();
                }
            }
        };
        Thread t = new Thread(r);
        t.start();
    }
}
