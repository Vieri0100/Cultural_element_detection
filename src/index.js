import React from "react";
import ReactDOM from "react-dom/client";
import {StrictMode} from "react";
import { createRoot } from "react-dom/client";
import * as tf from "@tensorflow/tfjs";
import { loadGraphModel } from "@tensorflow/tfjs-converter";
import "./styles.css";

tf.setBackend("webgl");

const threshold = 0.75;
const MODEL_INPUT_SIZE = [640, 640];

async function load_model() {
  const model = await loadGraphModel(
    "https://raw.githubusercontent.com/Vieri0100/Cultural_element_detection/main/models/element_detector/model.json"
  );
  return model;
}

let classesDir = {
  1: { name: "Feilai_Feng_Statue", id: 1 },
};

class App extends React.Component {
  imageRef = React.createRef();
  canvasRef = React.createRef();
  fileInputRef = React.createRef();
  model = null;

  handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      if (!this.model) {
        this.model = await load_model();
      }
      const imageElement = this.imageRef.current;
      imageElement.src = URL.createObjectURL(file);
    }
  };

  handleImageLoad = () => {
    const image = this.imageRef.current;
    const canvas = this.canvasRef.current;
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    this.detectImage(image, this.model);
  };

  detectImage = (image, model) => {
    tf.engine().startScope();
    const inputTensor = tf.tidy(() => {
      const tfimg = tf.browser.fromPixels(image);
      const resized = tf.image.resizeBilinear(tfimg, MODEL_INPUT_SIZE);
      const intTensor = resized.toInt();
      const expanded = intTensor.expandDims();
      return expanded;
    });

    model.executeAsync({ 'input_tensor': inputTensor }).then((predictions) => {
      this.renderPredictions(predictions, image);
      tf.engine().endScope();
    });
  };

  buildDetectedObjects = async (scores, boxes, classes) => {
    const image = this.imageRef.current;

    const flatBoxes = boxes[0].map(box => box.flat ? box.flat() : box);
    const boxesTensor = tf.tensor2d(flatBoxes);
    
    const flatScores = scores[0].flat(); 
    const scoresTensor = tf.tensor1d(flatScores);

    const selectedIndices = await tf.image.nonMaxSuppressionAsync(
      boxesTensor, scoresTensor, 20, 0.5, threshold
    );

    const selectedIndicesArray = await selectedIndices.array();

    const detectionObjects = selectedIndicesArray.map((i) => {
      const numericScore = Number(scores[0][i]);
      const box = boxes[0][i];
      
      const minY = box[0] * image.naturalHeight;
      const minX = box[1] * image.naturalWidth;
      const maxY = box[2] * image.naturalHeight;
      const maxX = box[3] * image.naturalWidth;
      
      const bbox = [minX, minY, maxX - minX, maxY - minY];
      
      const classId = parseInt(classes[i]); 
      
      const label = classesDir[classId]?.name || "Feilai_Feng_Statue";
      
      return {
        class: classId,
        label: label,
        score: numericScore.toFixed(4),
        bbox: bbox,
      };
    });

    boxesTensor.dispose();
    scoresTensor.dispose();
    selectedIndices.dispose();
    
    return detectionObjects;
  };

  renderPredictions = async (predictions, image) => {
    const ctx = this.canvasRef.current.getContext("2d");
    
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";
    
    const boxes = predictions[4].arraySync();
    const scores = predictions[5].arraySync();
    const classes = predictions[6].dataSync();
    
    const detections = await this.buildDetectedObjects(scores, boxes, classes);
    
    detections.forEach((item) => {
      const [x, y, width, height] = item.bbox;
      ctx.strokeStyle = "#00FFFF";
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, width, height);
      ctx.fillStyle = "#00FFFF";
      const text = `${item.label} ${(100 * item.score).toFixed(2)}%`;
      const textWidth = ctx.measureText(text).width;
      const textHeight = parseInt(font, 10);
      ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
    });
    
    detections.forEach((item) => {
      const [x, y] = item.bbox;
      ctx.fillStyle = "#000000";
      ctx.fillText(`${item.label} ${(100 * item.score).toFixed(2)}%`, x, y);
    });
  };
  
  render() {
    return (
      <div style={{ position: "relative", display: "inline-block" }}>
        <input
          type="file"
          accept="image/*"
          ref={this.fileInputRef}
          onChange={this.handleFileUpload}
        />
        <img
          ref={this.imageRef}
          id="frame"
          alt="Uploaded"
          onLoad={this.handleImageLoad}
          style={{ display: "block" }}
        />
        <canvas
          ref={this.canvasRef}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            pointerEvents: "none",
          }}
        />
      </div>
    );
  }
}

const rootElement = document.getElementById("root");
const root = createRoot(rootElement);

root.render(
  <StrictMode>
    <App />
  </StrictMode>
);
