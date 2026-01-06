import { useState } from "react";
import Message from "./Message";
import SourceCard from "./SourceCard";

function getSessionId() {
  let id = localStorage.getItem("session_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("session_id", id);
  }
  return id;
}

export default function ChatBox() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sessionId = getSessionId();

  async function sendMessage() {
    if (!input.trim()) return;

    const userMsg = { role: "user", text: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json",
          Authorization: `Bearer ${localStorage.getItem("token")}`,
         },
        body: JSON.stringify({
          question: input,
          session_id: sessionId,
        }),
      });

      if (!res.ok) throw new Error("API error");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      let aiText = "";
      let meta = null;

      // Insert empty assistant message
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "", sources: [], faithful: null },
      ]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);

        // 🔑 Detect metadata chunk
        if (chunk.includes("__META__")) {
          const [textPart, metaPart] = chunk.split("__META__");
          aiText += textPart;
          meta = JSON.parse(metaPart);
        } else {
          aiText += chunk;
        }

        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];

          last.text = aiText;

          if (meta) {
            last.sources = meta.sources || [];
            last.faithful = meta.faithful;
            last.reason = meta.reason;
          }

          return updated;
        });
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "⚠️ Error contacting the server." },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col h-screen max-w-3xl mx-auto p-4 bg-gradient-to-b from-[#380e60] via-[#e9e7ea] to-[#9f4bdf] rounded-lg shadow-lg">
      <div className="flex-1 overflow-y-auto mb-4">
        {messages.map((msg, idx) => (
          <div key={idx}>
            <Message role={msg.role} text={msg.text} />

            {/* Faithfulness indicator */}
            {msg.faithful !== null && (
              <p
                className={`text-xs ml-2 ${
                  msg.faithful ? "text-green-600" : "text-red-600"
                }`}
              >
                {msg.faithful
                  ? "✔ Answer verified from documents"
                  : "⚠ Not fully supported by documents"}
              </p>
            )}

            {/* Sources */}
            {msg.sources && msg.sources.length > 0 && (
              <div className="ml-2 mb-3 space-y-2">
                <p className="text-xs text-gray-500">Sources:</p>
                {msg.sources.map((src, i) => (
                  <SourceCard key={i} source={src} />
                ))}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="text-gray-400 italic">Thinking…</div>
        )}
      </div>

      <div className="p-4 border-t flex gap-2 bg-pink-100 rounded-tl-xl">
        <input
          className="flex-1 border rounded-tl-2xl px-3 py-2 bg-black text-white focus:outline-none focus:ring-2 focus:ring-purple-600"
          placeholder="Ask about the tax reform bills..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button
          onClick={sendMessage}
          className="bg-[#410976] text-white px-4 py-2 rounded-md hover:bg-[#410974]/90"
        >
          Send
        </button>
      </div>
    </div>
  );
}
