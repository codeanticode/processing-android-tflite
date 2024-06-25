/**
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     https://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//import tensorflowlite.*;
import processing.core.PApplet;

import org.tensorflow.lite.*;
import android.app.Activity;
import java.util.List;
import java.io.*;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;
import java.util.Arrays;
import java.util.ArrayList;

// Todo extract byte buffer allocation
// Figure out naming of variables

public class TfliteManager {
  PApplet parent;
  Activity activity;
  List<String> labelList;
  Interpreter interpreter;
  Classifier classifier;

  /** Can be set by users. */
  int INPUT_SIZE;
  boolean QUANT = false;

  // show results
  int RESULTS_TO_SHOW = 3;

  public TfliteManager(PApplet parent) {
    this.parent = parent;
    this.activity = parent.getActivity();
  }

  public void setInputSize(int size) {
    INPUT_SIZE = size;
  }

  public void isQuantModel(boolean quant) {
    QUANT = quant;
  }

  public Bitmap loadImage(String path) {
    // How to print all images in assets folder:
    //String[] test = surface.getAssets().list("");
    //System.out.println(Arrays.toString(test));

    Bitmap loadedImage = null;

    try {
      //InputStream is = surface.getAssets().open(path);
      InputStream is = this.activity.getAssets().open(path);
      loadedImage = loadFrom(is);
      is.close();
    }
    catch (IOException e) {
    }

    return loadedImage;
  }

  Bitmap loadFrom(InputStream is) { 
    try {
      return BitmapFactory.decodeStream(is);
    } 
    catch (Exception e) {
      System.out.println(e);
      return null;
    }
  }

  public Bitmap resizeImage(Bitmap bitmap) {
    return Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);
  }

  public Bitmap cropCenterImage(Bitmap bitmap) {
    Bitmap result;
    int w = bitmap.getWidth();
    int h = bitmap.getHeight();

    if (w == h) {
      return bitmap;
    }

    if (w > h) {
      result = Bitmap.createBitmap(bitmap, w/2 - h/2, 0, h, h);
    } else {
      result = Bitmap.createBitmap(bitmap, 0, h/2 - w/2, w, w);
    }

    return result;
  }

  // TODO: should probably catch when folks are using quant vs not spits an IllegalArgumentException
  public void loadModel(String modelPath, String labelPath) {
    try {
      if (QUANT) {
        this.classifier = new ClassifierQuantizedMobileNet(activity, modelPath, labelPath, INPUT_SIZE, INPUT_SIZE);
      } else {
        this.classifier = new ClassifierFloatMobileNet(this.activity, modelPath, labelPath, INPUT_SIZE, INPUT_SIZE);
      }
    } 
    catch (IOException e) { 
      System.out.println("Failed to load classifier");
      System.out.println(e.getMessage());
      //return;
    }
  }

  public void setMaxResults(int maxResults) {
    this.classifier.setMaxResults(maxResults);
  }

  public String recognizeImage(Bitmap img) {
    List<Classifier.Recognition> results = this.classifier.recognizeImage(img);
    return results.toString();
  }
}
