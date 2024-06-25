package tflite.example;

import android.graphics.Bitmap;

import org.tensorflow.lite.task.vision.classifier.Classifications;

import java.util.List;

import processing.core.*;

// Adapted from
// https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android

// More examples here:
// https://www.tensorflow.org/lite/examples

public class Sketch extends PApplet implements ImageClassifierHelper.ClassifierListener {
  protected ImageClassifierHelper imageClassifierHelper;
  protected PImage image;

  public void settings() {
    fullScreen();
  }

  public void setup() {
    orientation(PORTRAIT);
    image = loadImage("cat.jpeg");
    image.loadPixels();
    PFont font = createFont("SansSerif", 40 * displayDensity);
    textFont(font);

    imageClassifierHelper = new ImageClassifierHelper(getContext(), this);


//    // Approach 1: With Interpreter(MappedBufferByte)
//    Activity activity = getActivity();
//    String modelFile = dataPath("mobilenet_v1_0.5_224.tflite");
//    Interpreter tflite;
//
//     try {
//       tflite = new Interpreter(hello.loadModelFile(activity, modelFile));
//     } catch (IOException e) {
//       e.printStackTrace();
//    }
  }

  public void draw() {
    //background(0);
    background(255, 204, 0);
    fill(146);

    imageClassifierHelper.classify((Bitmap)image.getNative(), 0);
    //text(hello.sayHello(), mouseX, mouseY);
  }

  @Override
  public void onError(String error) {
    print("Error: ", error);
  }

  @Override
  public void onResults(List<Classifications> results, long inferenceTime) {
    for (Classifications cl: results) {
      println(cl);
    }
  }
}

