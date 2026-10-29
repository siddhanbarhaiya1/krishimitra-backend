
package krishimitra;

import ai.onnxruntime.*;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import javax.imageio.ImageIO;

@Service
public class CropDiseaseService {

    private static final int IMAGE_SIZE = 224;
    private static final int RESIZE_SIZE = 256;
    private static final float TEMPERATURE = 1.210394f;

    private static final float[] MEAN = {
        0.485f, 0.456f, 0.406f
    };

    private static final float[] STD = {
        0.229f, 0.224f, 0.225f
    };

    private static final String[] CLASS_NAMES = {
        "Pepper__bell___Bacterial_spot",
        "Pepper__bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight",
        "Tomato_Late_blight",
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite",
        "Tomato__Target_Spot",
        "Tomato__Tomato_YellowLeaf__Curl_Virus",
        "Tomato__Tomato_mosaic_virus",
        "Tomato_healthy"
    };

    private final OrtEnvironment environment;
    private final OrtSession session;

    public CropDiseaseService() throws Exception {
        environment = OrtEnvironment.getEnvironment();

        ClassPathResource resource =
            new ClassPathResource("models/model.onnx");

        Path modelPath = Files.createTempFile(
            "krishimitra-model-", ".onnx"
        );

        modelPath.toFile().deleteOnExit();

        try (InputStream input = resource.getInputStream()) {
            Files.copy(
                input,
                modelPath,
                java.nio.file.StandardCopyOption.REPLACE_EXISTING
            );
        }

        session = environment.createSession(
            modelPath.toString(),
            new OrtSession.SessionOptions()
        );
    }

    public Map<String, Object> predict(BufferedImage image)
            throws Exception {

        // Step 1: Resize the shorter side to 256,
        // preserving the original image's aspect ratio.
        int originalWidth = image.getWidth();
        int originalHeight = image.getHeight();

        if (originalWidth <= 0 || originalHeight <= 0) {
            throw new IllegalArgumentException(
                "Invalid image dimensions."
            );
        }

        double scale = (double) RESIZE_SIZE
            / Math.min(originalWidth, originalHeight);

        int resizedWidth = Math.max(
            RESIZE_SIZE,
            (int) Math.round(originalWidth * scale)
        );

        int resizedHeight = Math.max(
            RESIZE_SIZE,
            (int) Math.round(originalHeight * scale)
        );

        BufferedImage resized = new BufferedImage(
            resizedWidth,
            resizedHeight,
            BufferedImage.TYPE_INT_RGB
        );

        Graphics2D graphics = resized.createGraphics();

        try {
            graphics.setRenderingHint(
                RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR
            );

            graphics.setRenderingHint(
                RenderingHints.KEY_RENDERING,
                RenderingHints.VALUE_RENDER_QUALITY
            );

            graphics.drawImage(
                image,
                0,
                0,
                resizedWidth,
                resizedHeight,
                null
            );
        } finally {
            graphics.dispose();
        }

        // Step 2: Center crop to 224 x 224.
        int cropX = (resizedWidth - IMAGE_SIZE) / 2;
        int cropY = (resizedHeight - IMAGE_SIZE) / 2;

        BufferedImage cropped = resized.getSubimage(
            cropX,
            cropY,
            IMAGE_SIZE,
            IMAGE_SIZE
        );

        // Step 3: Convert RGB pixels to normalized NCHW float data.
        float[] inputData = new float[
            3 * IMAGE_SIZE * IMAGE_SIZE
        ];

        int planeSize = IMAGE_SIZE * IMAGE_SIZE;

        for (int y = 0; y < IMAGE_SIZE; y++) {
            for (int x = 0; x < IMAGE_SIZE; x++) {

                int rgb = cropped.getRGB(x, y);

                float red =
                    ((rgb >> 16) & 0xFF) / 255.0f;

                float green =
                    ((rgb >> 8) & 0xFF) / 255.0f;

                float blue =
                    (rgb & 0xFF) / 255.0f;

                int pixel = y * IMAGE_SIZE + x;

                inputData[pixel] =
                    (red - MEAN[0]) / STD[0];

                inputData[planeSize + pixel] =
                    (green - MEAN[1]) / STD[1];

                inputData[2 * planeSize + pixel] =
                    (blue - MEAN[2]) / STD[2];
            }
        }

        // Step 4: Run the ONNX model.
        long[] shape = {
            1, 3, IMAGE_SIZE, IMAGE_SIZE
        };

        try (
            OnnxTensor tensor = OnnxTensor.createTensor(
                environment,
                FloatBuffer.wrap(inputData),
                shape
            );

            OrtSession.Result result = session.run(
                Map.of("input", tensor)
            )
        ) {
            float[][] logits = (float[][]) result
                .get("logits")
                .orElseThrow()
                .getValue();

            // Step 5: Apply temperature-scaled softmax.
            double[] probabilities =
                new double[logits[0].length];

            double maxLogit = Double.NEGATIVE_INFINITY;

            for (float logit : logits[0]) {
                maxLogit = Math.max(
                    maxLogit,
                    logit / TEMPERATURE
                );
            }

            double sum = 0.0;

            for (int i = 0; i < logits[0].length; i++) {
                probabilities[i] = Math.exp(
                    logits[0][i] / TEMPERATURE - maxLogit
                );

                sum += probabilities[i];
            }

            if (sum == 0.0 || !Double.isFinite(sum)) {
                throw new IllegalStateException(
                    "The model returned invalid scores."
                );
            }

            int bestIndex = 0;

            for (int i = 0; i < probabilities.length; i++) {
                probabilities[i] /= sum;

                if (probabilities[i] >
                        probabilities[bestIndex]) {
                    bestIndex = i;
                }
            }
System.out.println("DEBUG - Top 5 predictions:");

int[] topIndices = new int[5];
java.util.Arrays.fill(topIndices, -1);

for (int rank = 0; rank < 5; rank++) {
    int nextIndex = -1;

    for (int i = 0; i < probabilities.length; i++) {
        boolean alreadySelected = false;

        for (int selected : topIndices) {
            if (selected == i) {
                alreadySelected = true;
                break;
            }
        }

        if (!alreadySelected &&
            (nextIndex == -1 ||
             probabilities[i] > probabilities[nextIndex])) {
            nextIndex = i;
        }
    }

    if (nextIndex == -1) {
        break;
    }

    topIndices[rank] = nextIndex;

    System.out.printf(
        "%d. %s = %.2f%%%n",
        rank + 1,
        CLASS_NAMES[nextIndex],
        probabilities[nextIndex] * 100.0
    );
}
            boolean isTomatoClass = bestIndex >= 5
                && bestIndex < CLASS_NAMES.length;

            double confidence =
                probabilities[bestIndex] * 100.0;

            System.out.println(
                "DEBUG - Top predicted class: "
                    + CLASS_NAMES[bestIndex]
            );

            System.out.println(
                "DEBUG - Confidence: "
                    + confidence + "%"
            );

            return Map.of(
                "crop", isTomatoClass ? "tomato" : "unknown",

                "disease", isTomatoClass
                    ? CLASS_NAMES[bestIndex]
                    : "Unable to confirm a tomato leaf",

                "confidence", confidence,

                "message", isTomatoClass
                    ? "Prediction generated. Please verify with a farming expert."
                    : "This image may not be a tomato leaf. Please upload a clear tomato leaf image."
            );
        }
    }
}
