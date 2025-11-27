package examples.android.fast_image_processing

import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * FastImageProcessor — small helper to preprocess and postprocess full-HD FP32 images
 * without using TensorFlow Lite Task's ImageProcessor (which allocates per call and is
 * expensive for ~2M pixels). This keeps reusable direct ByteBuffer/FloatBuffer,
 * IntArray buffers and Bitmaps to avoid per-frame allocation and GC.
 *
 * Usage:
 *  - Create one instance for fixed width/height (e.g. 1920x1080) and reuse it every frame.
 *  - Call `preprocess(bitmap)` to fill the internal FloatBuffer with normalized float RGB
 *    data in the order [R, G, B, R, G, B, ...]. Then pass the underlying `byteBuffer`
 *    to the TFLite Interpreter if it accepts ByteBuffer, or use `floatBuffer` for other APIs.
 *  - After inference, call `postprocessToBitmap(outputFloatBuffer, outBitmap)` to write the
 *    pixels into a reusable Bitmap. Avoid creating new Bitmaps per frame.
 */
class FastImageProcessor(
    private val width: Int,
    private val height: Int,
    private val channels: Int = 3 // RGB
) {
    private val pixelCount = width * height

    // Direct ByteBuffer for FP32: pixelCount * channels * 4 bytes
    val byteBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(pixelCount * channels * 4)
        .order(ByteOrder.nativeOrder())
    val floatBuffer: FloatBuffer = byteBuffer.asFloatBuffer()

    // Reusable temporary arrays to avoid allocations each frame
    private val intPixels = IntArray(pixelCount)
    private val outIntPixels = IntArray(pixelCount)

    // Create an output bitmap once and reuse it (ARGB_8888 expects 4 bytes/pixel)
    fun createReusableBitmap(): Bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

    /** Preprocess a source Bitmap into the internal FloatBuffer (RGB order, normalized to [0,1]).
     * The FloatBuffer is rewound at the end and ready for reading by typical frameworks.
     */
    fun preprocess(bitmap: Bitmap) {
        require(bitmap.width == width && bitmap.height == height) { "Bitmap size mismatch" }

        // Copy pixels once into an IntArray (fast native implementation)
        bitmap.getPixels(intPixels, 0, width, 0, 0, width, height)

        floatBuffer.rewind()
        val inv255 = 1.0f / 255.0f

        var i = 0
        val n = intPixels.size
        while (i < n) {
            val p = intPixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            floatBuffer.put(r * inv255)
            floatBuffer.put(g * inv255)
            floatBuffer.put(b * inv255)
            i++
        }

        // Rewind the FloatBuffer so downstream readers can read from start
        floatBuffer.rewind()
    }

    /** Alternative: get the underlying ByteBuffer (rewound) for Interpreter that accepts
     * ByteBuffer input. Call `byteBuffer.rewind()` if you modified floatBuffer directly.
     */
    fun getInputByteBuffer(): ByteBuffer {
        byteBuffer.rewind()
        return byteBuffer
    }

    /** Postprocess an output FloatBuffer (RGB float in [0,1]) into a provided reusable Bitmap.
     * This writes to the Bitmap by assembling an IntArray and calling setPixels once.
     */
    fun postprocessToBitmap(outFb: FloatBuffer, outBitmap: Bitmap): Bitmap {
        require(outBitmap.width == width && outBitmap.height == height) { "Bitmap size mismatch" }

        outIntPixels.fill(0)
        outFb.rewind()

        val mul = 255f
        var i = 0
        val len = pixelCount
        while (i < len) {
            val r = (outFb.get() * mul).toInt().coerceIn(0, 255)
            val g = (outFb.get() * mul).toInt().coerceIn(0, 255)
            val b = (outFb.get() * mul).toInt().coerceIn(0, 255)
            outIntPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            i++
        }

        outBitmap.setPixels(outIntPixels, 0, width, 0, 0, width, height)
        outFb.rewind()
        return outBitmap
    }

    /** Optional: helper to postprocess into a newly-created Bitmap if caller wants that.
     * Avoid using in a hot path; prefer reusing a bitmap created by `createReusableBitmap()`.
     */
    fun postprocessToNewBitmap(outFb: FloatBuffer): Bitmap {
        val b = createReusableBitmap()
        return postprocessToBitmap(outFb, b)
    }
}
