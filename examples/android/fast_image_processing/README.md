# Fast image preprocessing/postprocessing for full-HD FP32

Problem
-------

Using TensorFlow Lite Task's `ImageProcessor` for full-HD FP32 images (e.g. 1920×1080×3)
can be very slow on Android because `ImageProcessor` builds temporaries, performs per-op copies
and creates intermediate objects each frame. For ~2M pixels this per-frame allocation and
per-pixel Java overhead can add ~150–180 ms to your pipeline.

Solution
--------

Replace `ImageProcessor` for large FP32 frames with a tight manual conversion that:

- Allocates a direct `ByteBuffer` once (native order) and keeps an `asFloatBuffer()` view.
- Copies pixels once from a `Bitmap` into an `IntArray` with `bitmap.getPixels(...)`.
- Converts the `IntArray` to floats in a single loop into the reusable `FloatBuffer`.
- Feed the interpreter directly from the `ByteBuffer`/`FloatBuffer`.
- Postprocess by reading floats out in a single loop into an `IntArray` and calling
  `bitmap.setPixels(...)` once.

This example includes `FastImageProcessor.kt` which implements these steps and shows
how to create and reuse buffers and bitmaps.

Usage example (Kotlin)
----------------------

1. Create the processor once for your model input size (e.g. 1920×1080):

```kotlin
val w = 1920
val h = 1080
val fast = FastImageProcessor(w, h)
val inputBitmap = ... // source frame, already scaled to w x h
val outputBitmap = fast.createReusableBitmap()
```

2. Per-frame pipeline:

```kotlin
// Preprocess
fast.preprocess(inputBitmap)
val inputByteBuffer = fast.getInputByteBuffer() // rewound

// Run interpreter (example: interpreter.run(inputByteBuffer, outputFloatBuffer) )

// Postprocess (outFb is a FloatBuffer containing RGB floats in [0,1])
fast.postprocessToBitmap(outFb, outputBitmap)
// render outputBitmap
```

Notes and tips
--------------

- If acceptable for accuracy, quantize your model to UINT8: then you can use
  `bitmap.copyPixelsToBuffer()` and avoid float conversions entirely, which saves
  a large amount of CPU time.
- If you still need more speed, a JNI/C++ implementation of the conversion loop
  can reduce overhead further, but try the Java direct-buffer approach first.
- Profile with Android Studio to measure `preprocess -> model -> postprocess` times
  and confirm reductions (goal: preprocess+postprocess << 150 ms, ideally <40 ms).

Where to find the file
----------------------

`examples/android/fast_image_processing/FastImageProcessor.kt`
