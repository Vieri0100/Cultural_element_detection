import React from "react";
import ReactDOM from "react-dom";
import * as tf from "@tensorflow/tfjs";
import { loadGraphModel } from "@tensorflow/tfjs-converter";
import "./styles.css";
tf.setBackend("webgl");

const threshold = 0.75;

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
  videoRef = React.createRef();
  canvasRef = React.createRef();
  fileInputRef = React.createRef();

  handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      const videoElement = this.videoRef.current;
      videoElement.src = URL.createObjectURL(file);

      const model = await load_model();
      videoElement.onloadeddata = () => {
        this.detectFrame(videoElement, model);
      };
    }
  };

  detectFrame = (video, model) => {
    tf.engine().startScope();
    model.executeAsync(this.process_input(video)).then((predictions) => {
      this.renderPredictions(predictions, video);
      requestAnimationFrame(() => {
        this.detectFrame(video, model);
      });
      tf.engine().endScope();
    });
  };

  process_input(video_frame) {
    const tfimg = tf.browser.fromPixels(video_frame).toInt();
    const expandedimg = tfimg.transpose([0, 1, 2]).expandDims();
    return expandedimg;
  } 

  buildDetectedObjects(scores, threshold, boxes, classes, classesDir) {
    const detectionObjects = [];
    const video_frame = document.getElementById("frame");

    scores[0].forEach((score, i) => {
      if (score > threshold) {
        const bbox = [];
        const minY = boxes[0][i][0] * video_frame.offsetHeight;
        const minX = boxes[0][i][1] * video_frame.offsetWidth;
        const maxY = boxes[0][i][2] * video_frame.offsetHeight;
        const maxX = boxes[0][i][3] * video_frame.offsetWidth;
        bbox[0] = minX;
        bbox[1] = minY;
        bbox[2] = maxX - minX;
        bbox[3] = maxY - minY;

        detectionObjects.push({
          class: classes[i],
          label: classesDir[classes[i]].name,
          score: score.toFixed(4),
          bbox: bbox,
        });
      }
    });

    return detectionObjects;
  }

  renderPredictions = (predictions) => {
    const ctx = this.canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";

    const boxes = predictions[4].arraySync();
    const scores = predictions[5].arraySync();
    const classes = predictions[6].dataSync();

    const detections = this.buildDetectedObjects(
      scores,
      threshold,
      boxes,
      classes,
      classesDir
    );

    detections.forEach((item) => {
      const x = item["bbox"][0];
      const y = item["bbox"][1];
      const width = item["bbox"][2];
      const height = item["bbox"][3];

      ctx.strokeStyle = "#00FFFF";
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, width, height);

      ctx.fillStyle = "#00FFFF";
      const textWidth = ctx.measureText(
        `${item["label"]} ${(100 * item["score"]).toFixed(2)}%`
      ).width;
      const textHeight = parseInt(font, 10);

      ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
    });

    detections.forEach((item) => {
      const x = item["bbox"][0];
      const y = item["bbox"][1];
      ctx.fillStyle = "#000000";
      ctx.fillText(
        `${item["label"]} ${(100 * item["score"]).toFixed(2)}%`,
        x,
        y
      );
    });
  };

  render() {
    return (
      <div>
        <h1>Cultural Element Detection on Uploaded Videos</h1>
        <input
          type="file"
          accept="video/*"
          ref={this.fileInputRef}
          onChange={this.handleFileUpload}
        />
        <video
          ref={this.videoRef}
          id="frame"
          width="1280"
          height="720"
          controls
          style={{ display: "block", marginTop: "10px" }}
        />
        <canvas
          ref={this.canvasRef}
          width="1280"
          height="720"
          style={{ position: "absolute", top: 0, left: 0 }}
        />
      </div>
    );
  }
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);