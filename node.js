/*
1. Tải onnx và onnx.data
2. Tải label.json
Code nodejs cần follow 
*/


const ort = require("onnxruntime-node");
const sharp = require("sharp");
const fs = require("fs");

const LABELS = JSON.parse(fs.readFileSync("labels.json", "utf-8"));
const IMG_SIZE = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

async function preprocessImage(imagePath) {
    // Resize → CenterCrop → normalize (mirrors Python transform)
    const { data, info } = await sharp(imagePath)
        .resize(256, 256)
        .extract({ left: 16, top: 16, width: IMG_SIZE, height: IMG_SIZE }) // center crop 224x224
        .raw()
        .toBuffer({ resolveWithObject: true });

    // HWC → CHW  +  normalize
    const float32 = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
    for (let h = 0; h < IMG_SIZE; h++) {
        for (let w = 0; w < IMG_SIZE; w++) {
            const srcIdx = (h * IMG_SIZE + w) * info.channels;
            for (let c = 0; c < 3; c++) {
                const pixel = data[srcIdx + c] / 255.0;
                float32[c * IMG_SIZE * IMG_SIZE + h * IMG_SIZE + w] =
                    (pixel - MEAN[c]) / STD[c];
            }
        }
    }

    return new ort.Tensor("float32", float32, [1, 3, IMG_SIZE, IMG_SIZE]);
}

async function predict(imagePath, topK = 3) {
    const session = await ort.InferenceSession.create("mobilenetv3_finetune.onnx");
    const inputTensor = await preprocessImage(imagePath);

    const results = await session.run({ input: inputTensor });
    const logits = results.output.data; // Float32Array

    // Softmax
    const max = Math.max(...logits);
    const exps = Array.from(logits).map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map((v) => v / sum);

    // Top-K
    const indexed = probs.map((p, i) => ({ index: i, prob: p }));
    indexed.sort((a, b) => b.prob - a.prob);

    return indexed.slice(0, topK).map(({ index, prob }) => ({
        label_id: index,
        label: LABELS[index],
        confidence: parseFloat(prob.toFixed(4)),
    }));
}

// ===== Run =====
predict("test.jpg", 3).then(console.log).catch(console.error);