import tensorflowlite.*;
import org.tensorflow.lite.*;
import android.app.Activity;

Tflite hello;

void setup() {
  fullScreen();
  //size(400, 400);
  
  hello = new Tflite();
  
  PFont font = createFont("SansSerif", 40 * displayDensity);
  textFont(font);
  
  // Approach 1: With Interpreter(MappedBufferByte)
  Activity activity = getActivity();  
  String modelFile = dataPath("squeezenet.tflite"); 
  Interpreter tflite;
  
   try {
     tflite=new Interpreter(hello.loadModelFile(activity, modelFile));
   } catch (IOException e) {
     e.printStackTrace();
  }
}

void draw() {
  //background(0);
  background(255, 204, 0);
  fill(146);
  //text(hello.sayHello(), mouseX, mouseY);
}