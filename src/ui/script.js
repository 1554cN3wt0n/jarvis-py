function handleFileUpload(event) {
  const files = event.target.files;
  if (files.length > 0) {
    alert(`${files.length} file(s) selected for upload.`);
  }
  var data = new FormData();
  data.append("file", files[0]);
  fetch("http://localhost:4200/document?type=file", {
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

  fetch(`http://localhost:4200/ask?question=${question}`)
    .then((response) => response.json())
    .then((result) => {
      console.log(result);
      document.getElementById("answer").textContent = result["answer"];
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

function startVoiceInput() {
  alert("Voice input not implemented yet!");
}
