import ai.onnxruntime.*;
import java.util.*;
import java.nio.FloatBuffer;
import ai.djl.Model;
import ai.djl.translate.TranslateException;
import ai.djl.util.Utils;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import joblib.*;

public class CyberbullyingDetectionTest {
    public static void main(String[] args) {
        String modelPath = "C:/Projects/CyberBulling Detection/ChatAPP/ChatAppAdmin/app/src/main/res/assets/model.onnx";
        String vectorizerPath = "C:/Projects/CyberBulling Detection/ChatAPP/ChatAppAdmin/app/src/main/res/assets/vectorizer.pkl";
        String inputText = "You are stupid and Fuck youuuu"; // Example input

        try {
            // Load TF-IDF vectorizer
            TfidfVectorizer vectorizer = (TfidfVectorizer) joblib.load(new File(vectorizerPath));

            // Initialize ONNX Runtime
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            options.addCPU(true);
            OrtSession session = env.createSession(modelPath, options);

            // Get model input name and expected shape
            String inputName = getModelInputName(session);
            long[] expectedShape = getModelInputShape(session, inputName);
            System.out.println("Expected input shape: " + Arrays.toString(expectedShape));

            // Convert input text into correct TF-IDF vector
            float[][] inputVector = preprocessInputText(inputText, vectorizer, expectedShape[1]);

            // Create ONNX tensor
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputVector[0]), new long[]{1, expectedShape[1]});

            // Run model inference
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put(inputName, inputTensor);
            OrtSession.Result results = session.run(inputs);

            // Process output
            float[][] output = (float[][]) results.get(0).getValue();
            int prediction = (output[0][0] > 0.5) ? 1 : 0;

            System.out.println("Classification: " + (prediction == 1 ? "Cyberbullying" : "Not Cyberbullying"));

            // Close resources
            inputTensor.close();
            session.close();
            env.close();

        } catch (OrtException | IOException | TranslateException e) {
            System.err.println("Error: " + e.getMessage());
        }
    }

    // Get model input name
    private static String getModelInputName(OrtSession session) throws OrtException {
        return session.getInputInfo().keySet().iterator().next();
    }

    // Get model input shape
    private static long[] getModelInputShape(OrtSession session, String inputName) throws OrtException {
        NodeInfo nodeInfo = session.getInputInfo().get(inputName);
        if (nodeInfo.getInfo() instanceof TensorInfo) {
            return ((TensorInfo) nodeInfo.getInfo()).getShape();
        } else {
            throw new OrtException("Input is not a tensor!");
        }
    }

    // **🔥 Fix: Correct preprocessing for TF-IDF input**
    private static float[][] preprocessInputText(String text, TfidfVectorizer vectorizer, long featureSize) throws TranslateException {
        // Convert text to TF-IDF vector
        float[] tfidfVector = vectorizer.transform(Collections.singletonList(text)).toArray();

        // Ensure correct shape (31361 features)
        float[][] vector = new float[1][(int) featureSize];
        System.arraycopy(tfidfVector, 0, vector[0], 0, Math.min(tfidfVector.length, (int) featureSize));

        return vector;
    }
}
