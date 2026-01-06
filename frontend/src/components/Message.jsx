
import SourceCard from "./SourceCard"

export default function Message({ message }) {
  const isUser = message.role === "user"

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3`}>
      <div
        className={`max-w-lg px-4 py-2 rounded-lg text-sm ${
          isUser
            ? "bg-green-600 text-white"
            : "bg-gray-200 text-gray-900"
        }`}>
        <p>{message.text}</p>
        {!isUser && message.sources && (
          <SourceCard sources={message.sources} />
        )}
      </div>
    </div>
  )
}
