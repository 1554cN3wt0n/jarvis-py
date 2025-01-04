let recorder;
let audioStream;

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
      recorder = new Recorder(
        new AudioContext().createMediaStreamSource(stream),
        {
          numChannels: 1,
        }
      );

      recorder.record();
      document.getElementById("stop-voice").disabled = false;
    })
    .catch((error) => {
      console.error("Error accessing microphone:", error);
    });
}

function stopVoiceInput() {
  console.log("her");
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
  }
}
