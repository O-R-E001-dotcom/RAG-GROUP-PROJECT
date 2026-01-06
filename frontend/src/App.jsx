
import './App.css'
import ChatBox from "./components/ChatBox"

export default function App() {
  return (
    <div className="min-h-screen bg-black flex items-center justify-center">
      <div className="w-full max-w-3xl bg-white shadow rounded-lg">
        <header className="p-4 border-b font-semibold text-xl  mb-4 text-center">
          🇳🇬 Nigerian Tax Reform AI Assistant
        </header>
        <ChatBox />
      </div>
    </div>
  )
}


