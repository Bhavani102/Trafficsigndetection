package com.surendramaran.yolov8tflite

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.BufferedReader
import java.io.InputStreamReader

class Detector(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String,
    private val detectorListener: DetectorListener
) {
    private var interpreter: Interpreter? = null
    private var labels = mutableListOf<String>()
    private lateinit var gpuDelegate: GpuDelegate

    private var tensorWidth = 320
    private var tensorHeight = 320

    private val imageProcessor = ImageProcessor.Builder().build()

    fun setup() {
        val model = FileUtil.loadMappedFile(context, modelPath)
        val options = Interpreter.Options()

        // Enable GPU Acceleration
        gpuDelegate = GpuDelegate()
        options.addDelegate(gpuDelegate)
        options.numThreads = 2  // Optimized for performance

        interpreter = Interpreter(model, options)

        context.assets.open(labelPath).bufferedReader().useLines { lines ->
            labels.addAll(lines)
        }
    }

    fun clear() {
        interpreter?.close()
        gpuDelegate.close()
    }

    fun detect(frame: Bitmap) {
        val startTime = SystemClock.uptimeMillis()

        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)
        val tensorImage = TensorImage.fromBitmap(resizedBitmap)
        val imageBuffer = imageProcessor.process(tensorImage).buffer

        val output = TensorBuffer.createFixedSize(intArrayOf(1, 84, 8400), TensorBuffer.FLOAT32)
        interpreter?.run(imageBuffer, output.buffer)

        val bestBoxes = bestBox(output.floatArray)
        val inferenceTime = SystemClock.uptimeMillis() - startTime

        if (bestBoxes.isNullOrEmpty()) {
            detectorListener.onEmptyDetect()
        } else {
            detectorListener.onDetect(bestBoxes, inferenceTime)
        }
    }

    private fun bestBox(array: FloatArray): List<BoundingBox>? {
        val boundingBoxes = mutableListOf<BoundingBox>()

        for (i in 0 until array.size step 6) {
            val confidence = array[i + 4]
            if (confidence > 0.4F) {  // Faster confidence thresholding
                boundingBoxes.add(
                    BoundingBox(
                        array[i], array[i + 1], array[i + 2], array[i + 3],
                        confidence, 0, labels[0]
                    )
                )
            }
        }

        return applyFastNMS(boundingBoxes)
    }

    private fun applyFastNMS(boxes: List<BoundingBox>): List<BoundingBox> {
        val sortedBoxes = boxes.sortedByDescending { it.cnf }
        return sortedBoxes.take(10)  // Select Top-10 high confidence boxes
    }

    interface DetectorListener {
        fun onEmptyDetect()
        fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long)
    }
}
