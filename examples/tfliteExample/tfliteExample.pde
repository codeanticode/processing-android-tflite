import processing.android.tensorflowlite.*;
import org.tensorflow.lite.*;
import android.app.Activity;

Tflite hello;

void setup() {
  fullScreen();
  //size(400, 400);

  hello = new Tflite(this);

  PFont font = createFont("SansSerif", 40 * displayDensity);
  textFont(font);

  // Approach 1: With Interpreter(MappedBufferByte)
  Activity activity = getActivity();
  String modelFile = dataPath("mobilenet_v1_1.0_224_quant.tflite");
  String labelFile = dataPath("labels_mobilenet_quant_v1_224.txt");
  Interpreter tflite;

   hello.loadModel(modelFile, labelFile);
}

void draw() {
  //background(0);
  background(255, 204, 0);
  fill(146);
  //text(hello.sayHello(), mouseX, mouseY);
}