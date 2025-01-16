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
      fetchDocumentList();
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

  fetch(`/jarvis/ask?question=${question}`)
    .then((response) => response.json())
    .then((result) => {
      document.getElementById("answer").textContent = result["answer"];
      speak(result["answer"]);
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

      const waveContainer = document.querySelector(".audio-wave-container");
      waveContainer.classList.remove("hidden");

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

      fetch("/audio/transcript", {
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

async function fetchDocumentList() {
  try {
    const response = await fetch("/documents");
    const clusters = await response.json();
    displayDocumentList(clusters);
  } catch (error) {
    console.error("Error fetching document list:", error);
  }
}

function displayDocumentList(clusters) {
  const documentList = document.getElementById("document-list");
  documentList.innerHTML = "";
  for (let cluster in clusters) {
    clusters[cluster].forEach((doc) => {
      const docItem = document.createElement("div");
      docItem.classList.add("document-item");

      // Label for the document name
      const label = document.createElement("label");
      label.htmlFor = `doc-${doc.id}`;
      label.textContent = doc.name;

      // Delete button with trash icon
      const deleteButton = document.createElement("button");
      deleteButton.classList.add("delete-button");
      deleteButton.title = "Delete Document";
      deleteButton.onclick = () => deleteDocument(cluster, doc.id, docItem);

      const trashIcon = document.createElement("span");
      trashIcon.classList.add("trash-icon");
      trashIcon.textContent = "🗑️"; // Unicode for trash icon

      deleteButton.appendChild(trashIcon);

      // Append elements to the document item
      docItem.appendChild(label);
      docItem.appendChild(deleteButton);

      documentList.appendChild(docItem);
    });
  }
}

// Function to delete a document by ID
async function deleteDocument(clusterId, docId, docElement) {
  try {
    const response = await fetch(`/cluster/${clusterId}/document/${docId}`, {
      method: "DELETE",
    });

    if (response.ok) {
      console.log(
        `Document ${docId} from Cluster ${clusterId} deleted successfully.`
      );
      // Update documents list
      fetchDocumentList();
    } else {
      console.error(
        `Failed to delete document ${docId} from Cluster ${clusterId}:`,
        response.statusText
      );
      alert("Failed to delete the document. Please try again.");
    }
  } catch (error) {
    console.error("Error deleting document:", error);
    alert("An error occurred while deleting the document.");
  }
}

document.addEventListener("DOMContentLoaded", fetchDocumentList);

async function speak(text) {
  try {
    const apiUrl = `/audio/speak?text=${text}`;
    const response = await fetch(apiUrl);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    const audioPlayer = document.getElementById("audio-player");

    audioPlayer.src = audioUrl;

    audioPlayer.play();
  } catch (error) {
    console.error("Error fetching audio:", error);
  }
}

async function classifyImage() {
  try {
    const video = document.getElementById("video-preview");
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
    });
    video.style.display = "block"; // Show the video preview
    video.srcObject = stream;
    document.getElementById("video-preview").classList.remove("hidden");
    // Take a picture after 3 seconds (for demonstration)
    setTimeout(() => {
      // Capture the image from the video feed
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext("2d");
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Stop the video stream
      stream.getTracks().forEach((track) => track.stop());
      video.style.display = "none";

      // Convert the image to a Blob and send it to the backend
      canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append("image_file", blob, "captured-image.png");

        try {
          document.getElementById("answer").classList.remove("hidden");
          document.getElementById("answer").textContent = "Fetching answer...";
          const response = await fetch("/image/classify", {
            method: "POST",
            body: formData,
          });
          const result = await response.json();
          document.getElementById("answer").textContent = result["label"];
          speak(result["label"]);
          document.getElementById("video-preview").classList.add("hidden");
        } catch (error) {
          console.error("Error sending image:", error);
        }
      });
    }, 3000); // Adjust the delay as needed
  } catch (error) {
    console.error("Error accessing camera:", error);
  }
}
