//import ketai.camera.*;
import android.util.DisplayMetrics;
 
float ts;

KetaiCamera cam;

void settings(){
  fullScreen();
}

void setup() {
  //orientation(PORTRAIT);
  //imageMode(CENTER);
  
  /*
  DisplayMetrics metrics = new DisplayMetrics();    
  getActivity().getWindowManager().getDefaultDisplay().getMetrics(metrics);    
  int screenDensity = metrics.densityDpi; 
  println(screenDensity);
 
  ts = screenDensity/160;
  println(ts);
  */
  
  //int newW = (int) (displayWidth * ts);
  //int newH = (int) (displayHeight * ts);
  
  cam = new KetaiCamera(this, displayWidth, displayHeight, 24);
}

void draw() {
  //image(cam, width, height);
  image(cam, 0, 0, displayWidth, displayHeight);
}

void onCameraPreviewEvent()
{
  cam.read();
}

// start/stop camera preview by tapping the screen
void mousePressed()
{
  if (cam.isStarted())
  {
    cam.stop();
  }
  else
    cam.start();
    //cam.setRotation(90);
}
//void keyPressed() {
//  if (key == CODED) {
//    if (keyCode == MENU) {
//      if (cam.isFlashEnabled())
//        cam.disableFlash();
//      else
//        cam.enableFlash();
//    }
//  }
//}
