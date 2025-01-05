let recorder;
let audioStream;
let audioContext;
let analyser;
let dataArray;
let bufferLength;

const canvas = document.getElementById("audio-visualizer");
const canvasCtx = canvas.getContext("2d");

function handleFileUpload(event) {
  const files = event.target.files;
  if (files.length > 0) {
    alert(`${files.length} file(s) selected for upload.`);
  }
  var data = new FormData();
  data.append("file", files[0]);
  fetch("/document?type=file", {
    method: "POST",
    body: data,
  })
    .then((response) => response.json())
    .then((result) => {
      alert(result["message"]);
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

function askQuestion() {
  const question = document.getElementById("question").value;
  if (!question.trim()) {
    alert("Please type a question.");
    return;
  }
  document.getElementById("answer").classList.remove("hidden");
  document.getElementById("answer").textContent = "Fetching answer...";

  fetch(`/ask?question=${question}`)
    .then((response) => response.json())
    .then((result) => {
      document.getElementById("answer").textContent = result["answer"];
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

function startVoiceInput() {
  navigator.mediaDevices
    .getUserMedia({ audio: true })
    .then((stream) => {
      audioStream = stream;
      audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      recorder = new Recorder(source, {
        numChannels: 1,
      });

      recorder.record();
      document.getElementById("stop-voice").disabled = false;

      analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      bufferLength = analyser.frequencyBinCount;
      dataArray = new Uint8Array(bufferLength);

      source.connect(analyser);

      visualize();
    })
    .catch((error) => {
      console.error("Error accessing microphone:", error);
    });
}

function stopVoiceInput() {
  if (recorder) {
    console.log("Stopping voice input...");
    recorder.stop();
    recorder.exportWAV((blob) => {
      const formData = new FormData();
      formData.append("audio_file", blob, "audio.wav");

      fetch("/transcript", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          console.log("Audio sent successfully:", data);
        })
        .catch((error) => {
          console.error("Error sending audio:", error);
        });
    });

    audioStream.getTracks().forEach((track) => track.stop());
    document.getElementById("stop-voice").disabled = true;

    const waveContainer = document.querySelector(".audio-wave-container");
    waveContainer.classList.add("hidden");
  }
}

function visualize() {
  const WIDTH = canvas.width;
  const HEIGHT = canvas.height;

  function draw() {
    requestAnimationFrame(draw);

    analyser.getByteTimeDomainData(dataArray);

    canvasCtx.fillStyle = "rgba(0, 0, 0, 0.2)";
    canvasCtx.fillRect(0, 0, WIDTH, HEIGHT);

    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = "#00ffcc";

    canvasCtx.beginPath();
    const sliceWidth = WIDTH / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * HEIGHT) / 2;

      if (i === 0) {
        canvasCtx.moveTo(x, y);
      } else {
        canvasCtx.lineTo(x, y);
      }
      x += sliceWidth;
    }

    canvasCtx.lineTo(WIDTH, HEIGHT / 2);
    canvasCtx.stroke();
  }

  draw();
}
