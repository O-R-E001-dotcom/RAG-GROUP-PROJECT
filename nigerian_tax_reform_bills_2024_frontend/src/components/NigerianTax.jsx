import React from "react";
import { useState, useRef, useEffect } from "react";
import cn from "../lib/utils";

const NigerianTax = () => {
  const [formData, setFormData] = useState("");
  const [userMessage, setUserMessage] = useState([]);
  const messageEndRef = useRef(null);

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [userMessage]);

  const handleSubmit = (e) => {
    e.preventDefault();
    handleRequest();
    setFormData("")
  };

  function handleRequest() {
    if (formData.trim() === "") return;

    setUserMessage((prev) => [...prev, { sender: "user", text: formData }]);
    const msg = formData;
    generateBotResponse(msg);
  }

  async function generateBotResponse(msg) {
    setUserMessage((prev) => [
      ...prev,
      {
        sender: "bot",
        text: "",
        typing: true,
      },
    ]);

    try {
      const response = await fetch("", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg }),
      });
      const data = await response.json();
      
      console.log(data)

      const words = data.response.split(" ");

      let i = 0;

        const interval = setInterval(() => {
        setUserMessage((prev) => {
          const updated = prev.slice(0, -1);

          const last = { ...prev[prev.length - 1], typing: false };

          if (i < words.length) {
            last.text += (i === 0 ? "" : " ") + words[i];
          }

          return [...updated, last];
        });

        i++;

          if (i >= words.length) clearInterval(interval);
        }, 200);
    } catch (err) {
      setUserMessage((prev) => [
        ...prev,
        { sender: "system", text: err.message },
      ]);
      setFormData("");
    }
  }

  return (
    <div className=" bg-gray-200 min-h-screen p-5">
      <div className=" absolute top-[50%] left-1/2 -translate-x-1/2 -translate-y-1/2 shadow-lg bg-blue-600 w-full sm:w-[70%]  md:w-[70%] lg:w-[65%] xl:w-[65%] h-[90vh] justify-center align-center rounded-l-2xl rounded-r-2xl ">
        <div className="h-[80vh] overflow-y-auto">
          {userMessage.map((req, idx) => (
            <div
              key={idx}
              className={
                req.sender === "bot" || req.sender === "system"
                  ? "w-full flex justify-start p-4"
                  : "w-full flex justify-end p-4"
              }
            >
              {req.typing ? (
               <div className="animate-bounce flex gap-2">
                 <div className="rounded-full bg-gray-200 transition-all w-5 h-5 duration-300 ">
                </div>
                  <div className="text-gray-500">typing..</div>
               </div>

              ) : (
                <div
                  className={
                    req.sender === "bot" || req.sender === "system" 
                      ? " p-3 bg-gray-200 max-w-[40%] rounded break-words"
                      : " p-3 bg-green-600 max-w-[40%] rounded break-words"
                  }
                >
                  {req.text}
                </div>
              )}
            </div>
          ))}
        <div ref={messageEndRef}></div>

        </div>

        <form
          onSubmit={handleSubmit}
          className="flex gap-2 justify-center items-center"
        >
          <input
            type="text"
            name="prompt"
            value={formData}
            onChange={(e) => setFormData(e.target.value)}
            className="p-2 h-10 w-[75%] bg-gray-200 rounded-xl outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Enter your message here"
          />
          <button
            className="bg-green-600 w-20 h-10 rounded-l-xl rounded-r-xl text-gray-100 font-bold "
            type="submit"
          >
            {" "}
            Send{" "}
          </button>
        </form>
      </div>
    </div>
  );
};

export default NigerianTax;