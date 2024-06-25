package tflite.example;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;
import android.view.Surface;

import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.core.BaseOptions;
import org.tensorflow.lite.task.core.vision.ImageProcessingOptions;
import org.tensorflow.lite.task.vision.classifier.Classifications;
import org.tensorflow.lite.task.vision.classifier.ImageClassifier;

import java.io.IOException;
import java.util.List;

public class ImageClassifierHelper {
  private float threshold = 0.5f;
  private int numThreads = 2;
  private int maxResults = 3;
  private int currentDelegate = 0;
  private int currentModel = 0;
  private Context context;
  private ClassifierListener imageClassifierListener;
  private ImageClassifier imageClassifier;

  public ImageClassifierHelper(Context context, ClassifierListener imageClassifierListener) {
    this.context = context;
    this.imageClassifierListener = imageClassifierListener;
    setupImageClassifier();
  }
  public ImageClassifierHelper(float threshold, int numThreads, int maxResults, int currentDelegate, int currentModel, Context context, ClassifierListener imageClassifierListener) {
    this.threshold = threshold;
    this.numThreads = numThreads;
    this.maxResults = maxResults;
    this.currentDelegate = currentDelegate;
    this.currentModel = currentModel;
    this.context = context;
    this.imageClassifierListener = imageClassifierListener;
    setupImageClassifier();
  }

  public void clearImageClassifier() {
    imageClassifier = null;
  }

  private void setupImageClassifier() {
    ImageClassifier.ImageClassifierOptions.Builder optionsBuilder = ImageClassifier.ImageClassifierOptions.builder()
            .setScoreThreshold(threshold)
            .setMaxResults(maxResults);

    BaseOptions.Builder baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads);

    switch (currentDelegate) {
      case DELEGATE_CPU:
        // Default
        break;
      case DELEGATE_GPU:
        if (new CompatibilityList().isDelegateSupportedOnThisDevice()) {
          baseOptionsBuilder.useGpu();
        } else {
          if (imageClassifierListener != null) {
            imageClassifierListener.onError("GPU is not supported on this device");
          }
        }
        break;
      case DELEGATE_NNAPI:
        baseOptionsBuilder.useNnapi();
        break;
    }

    optionsBuilder.setBaseOptions(baseOptionsBuilder.build());

    String modelName;
    switch (currentModel) {
      case MODEL_MOBILENETV1:
        modelName = "mobilenetv1.tflite";
        break;
      case MODEL_EFFICIENTNETV0:
        modelName = "efficientnet-lite0.tflite";
        break;
      case MODEL_EFFICIENTNETV1:
        modelName = "efficientnet-lite1.tflite";
        break;
      case MODEL_EFFICIENTNETV2:
        modelName = "efficientnet-lite2.tflite";
        break;
      default:
        modelName = "mobilenetv1.tflite";
        break;
    }

    try {
      imageClassifier = ImageClassifier.createFromFileAndOptions(context, modelName, optionsBuilder.build());
    } catch (IllegalStateException | IOException e) {
      if (imageClassifierListener != null) {
        imageClassifierListener.onError("Image classifier failed to initialize. See error logs for details");
      }
      Log.e(TAG, "TFLite failed to load model with error: " + e.getMessage());
    }
  }

  public void classify(Bitmap image, int rotation) {
    if (imageClassifier == null) {
      setupImageClassifier();
    }

    long inferenceTime = SystemClock.uptimeMillis();

    ImageProcessor imageProcessor = new ImageProcessor.Builder().build();

    TensorImage tensorImage = imageProcessor.process(TensorImage.fromBitmap(image));

    ImageProcessingOptions imageProcessingOptions = ImageProcessingOptions.builder()
            .setOrientation(getOrientationFromRotation(rotation))
            .build();

    List<Classifications> results = imageClassifier.classify(tensorImage, imageProcessingOptions);
    inferenceTime = SystemClock.uptimeMillis() - inferenceTime;

    if (imageClassifierListener != null) {
      imageClassifierListener.onResults(results, inferenceTime);
    }
  }

  private ImageProcessingOptions.Orientation getOrientationFromRotation(int rotation) {
    switch (rotation) {
      case Surface.ROTATION_270:
        return ImageProcessingOptions.Orientation.BOTTOM_RIGHT;
      case Surface.ROTATION_180:
        return ImageProcessingOptions.Orientation.RIGHT_BOTTOM;
      case Surface.ROTATION_90:
        return ImageProcessingOptions.Orientation.TOP_LEFT;
      default:
        return ImageProcessingOptions.Orientation.RIGHT_TOP;
    }
  }

  public interface ClassifierListener {
    void onError(String error);
    void onResults(List<Classifications> results, long inferenceTime);
  }

  public static final int DELEGATE_CPU = 0;
  public static final int DELEGATE_GPU = 1;
  public static final int DELEGATE_NNAPI = 2;
  public static final int MODEL_MOBILENETV1 = 0;
  public static final int MODEL_EFFICIENTNETV0 = 1;
  public static final int MODEL_EFFICIENTNETV1 = 2;
  public static final int MODEL_EFFICIENTNETV2 = 3;

  private static final String TAG = "ImageClassifierHelper";
}
