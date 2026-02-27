let chatHistory = [];

// -----------------------------
// Load chat history on page load
// -----------------------------
window.onload = function () {
    const saved = localStorage.getItem("chatHistory");
    
    if (saved) {
        chatHistory = JSON.parse(saved);
        renderChatHistory();
    }
};

// -----------------------------
// Save chat history to browser
// -----------------------------
function saveChatHistory() {
    localStorage.setItem("chatHistory", JSON.stringify(chatHistory));
}

// -----------------------------
// Render chat history
// -----------------------------
function renderChatHistory() {
    const chatBox = document.getElementById("chat-box");
    chatBox.innerHTML = "";

    chatHistory.forEach(msg => {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", msg.role);
        messageDiv.innerHTML = msg.content;
        chatBox.appendChild(messageDiv);
    });

    chatBox.scrollTop = chatBox.scrollHeight;
}

// -----------------------------
// Send message
// -----------------------------
async function sendMessage() {

    const input = document.getElementById("user-input");
    const message = input.value.trim();
    if (!message) return;

    const chatBox = document.getElementById("chat-box");

    // USER MESSAGE
    const userContent = message;
    chatHistory.push({
        role: "user",
        content: userContent
    });

    saveChatHistory();
    renderChatHistory();

    input.value = "";

    // LOADING MESSAGE
    const loadingDiv = document.createElement("div");
    loadingDiv.classList.add("message", "bot");
    loadingDiv.innerText = "Typing...";
    chatBox.appendChild(loadingDiv);
    chatBox.scrollTop = chatBox.scrollHeight;

    try {

        const response = await fetch("http://127.0.0.1:8000/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();

        chatBox.removeChild(loadingDiv);

        let botContent = "";

        if (data.type === "details") {

            if (data.error) {
                botContent = data.error;
            } else {
                botContent = `
                    <strong>${data.title}</strong><br>
                    <small>${data.genres} • ${data.runtime} mins</small><br><br>
                    ${data.overview}
                `;
            }

        } else if (data.type === "recommendation") {

            botContent = "<strong>Recommended Movies:</strong><br><br>";

            data.recommendations.forEach(movie => {
                botContent += `
                    🎬 <b>${movie.title}</b><br>
                    <small>${movie.genres}</small><br><br>
                `;
            });
        }

        chatHistory.push({
            role: "bot",
            content: botContent
        });

        saveChatHistory();
        renderChatHistory();

    } catch (error) {

        chatBox.removeChild(loadingDiv);

        const errorMessage = "Error connecting to server.";

        chatHistory.push({
            role: "bot",
            content: errorMessage
        });

        saveChatHistory();
        renderChatHistory();

        console.error(error);
    }
}

// -----------------------------
// Clear chat
// -----------------------------
function clearChat() {
    chatHistory = [];
    localStorage.removeItem("chatHistory");
    renderChatHistory();
}