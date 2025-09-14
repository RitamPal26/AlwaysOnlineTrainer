"use client";
import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import Image from "next/image";

interface Message {
  id: string;
  type: "user" | "coach";
  content: string;
  timestamp: Date;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [userInput, setUserInput] = useState("");
  const [messageCounter, setMessageCounter] = useState(1);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Initialize welcome message on client side only
  useEffect(() => {
    if (messages.length === 0) {
      setMessages([
        {
          id: "welcome-message",
          type: "coach",
          content:
            "Hey there! I'm your always-on AI fitness and nutrition coach. Upload exercise videos for form analysis or food photos for nutrition guidance!",
          timestamp: new Date(),
        },
      ]);
    }
  }, [messages.length]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const addMessage = (type: "user" | "coach", content: string) => {
    const newMessage: Message = {
      id: `${type}-${Date.now()}-${messageCounter}`,
      type,
      content,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, newMessage]);
    setMessageCounter((prev) => prev + 1);
  };

  const handleVideoUpload = async (file: File) => {
    addMessage("user", `📹 Uploaded: ${file.name}`);
    addMessage(
      "coach",
      "Perfect! Let me analyze your movement and provide coaching feedback..."
    );

    setIsAnalyzing(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/analyze-video", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (result.status === "success") {
        addMessage(
          "coach",
          `🎬 **Video Analysis**\nDuration: ${result.duration}s | Frames analyzed: ${result.frames_analyzed}\nProcessing with holistic AI coaching...`
        );

        setTimeout(() => {
          addMessage("coach", result.coaching_response);
          setIsAnalyzing(false);
        }, 2000);
      } else {
        addMessage(
          "coach",
          `Had trouble with that video: ${result.message}. Try another one!`
        );
        setIsAnalyzing(false);
      }
    } catch (error) {
      console.error("Upload error:", error);
      addMessage(
        "coach",
        "Connection issue! Make sure my backend is running properly."
      );
      setIsAnalyzing(false);
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      handleVideoUpload(file);
    }
  };

  const handleFoodUpload = async (file: File) => {
    addMessage("user", `🍎 Uploaded food photo: ${file.name}`);
    addMessage(
      "coach",
      "Let me analyze your meal and provide nutrition guidance..."
    );

    setIsAnalyzing(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/analyze-food", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (result.status === "success") {
        addMessage("coach", result.analysis);
      } else {
        addMessage(
          "coach",
          `Had trouble analyzing that food: ${result.message}`
        );
      }
    } catch (error) {
      console.error("Food upload error:", error);
      addMessage("coach", "Connection issue with food analysis!");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleFoodFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      handleFoodUpload(file);
    }
  };

  const handleSendMessage = async () => {
    if (!userInput.trim()) return;

    addMessage("user", userInput);
    setIsAnalyzing(true);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userInput }),
      });

      const result = await response.json();

      setTimeout(() => {
        addMessage("coach", result.response);
        setIsAnalyzing(false);
      }, 1500);
    } catch (error) {
      setTimeout(() => {
        addMessage(
          "coach",
          "I'm having connection issues! Upload a video and I'll analyze your form!"
        );
        setIsAnalyzing(false);
      }, 1000);
    }

    setUserInput("");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex flex-col">
      {/* Header */}
      <div className="bg-white shadow-lg border-b p-4">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative w-12 h-12">
              <Image
                src="/always online trainer.png"
                alt="Always Online Trainer"
                fill
                className="object-contain"
              />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Always Online Trainer
              </h1>
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <div
                  className={`w-3 h-3 rounded-full ${
                    isAnalyzing ? "bg-green-500 animate-pulse" : "bg-green-400"
                  }`}
                />
                <span className="font-medium">
                  {isAnalyzing ? "Analyzing..." : "Always ready to help"}
                </span>
              </div>
            </div>
          </div>

          <div className="text-right">
            <div className="text-xs text-gray-500">Powered by</div>
            <div className="text-sm font-bold text-blue-600">
              LFM2-VL Vision AI & Llama 3.1
            </div>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 max-w-4xl mx-auto w-full p-4 overflow-y-auto">
        <div className="space-y-6">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${
                message.type === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`flex items-start gap-3 max-w-lg ${
                  message.type === "user" ? "flex-row-reverse" : ""
                }`}
              >
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-sm ${
                    message.type === "user"
                      ? "bg-white-500 text-white"
                      : "bg-white-500 text-white"
                  }`}
                >
                  {message.type === "user" ? (
                    "👤"
                  ) : (
                    <Image
                      src="/always online trainer.png"
                      alt="AI Trainer"
                      width={24}
                      height={24}
                      className="inline-block"
                    />
                  )}
                </div>

                <div
                  className={`px-4 py-3 rounded-2xl shadow-sm ${
                    message.type === "user"
                      ? "bg-blue-500 text-white"
                      : "bg-white text-gray-800 border"
                  }`}
                >
                  <div className="text-sm leading-relaxed prose prose-sm max-w-none">
                    <ReactMarkdown
                      components={{
                        p: ({ children }) => <p className="mb-2">{children}</p>,
                        strong: ({ children }) => (
                          <strong className="font-bold text-gray-900">
                            {children}
                          </strong>
                        ),
                        ul: ({ children }) => (
                          <ul className="list-disc pl-4 mb-2">{children}</ul>
                        ),
                        li: ({ children }) => (
                          <li className="mb-1">{children}</li>
                        ),
                      }}
                    >
                      {message.content}
                    </ReactMarkdown>
                  </div>
                  <div
                    className={`text-xs mt-2 ${
                      message.type === "user"
                        ? "text-blue-100"
                        : "text-gray-500"
                    }`}
                  >
                    {message.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            </div>
          ))}

          {/* Typing indicator */}
          {isAnalyzing && (
            <div className="flex justify-start">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-orange-500 rounded-full flex items-center justify-center text-sm text-white">
                  🏋️‍♂️
                </div>
                <div className="bg-white px-4 py-3 rounded-2xl shadow-sm border">
                  <div className="flex gap-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                    <div
                      className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                      style={{ animationDelay: "0.1s" }}
                    ></div>
                    <div
                      className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                      style={{ animationDelay: "0.2s" }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="bg-white border-t-2 border-gray-200 p-4">
        <div className="max-w-4xl mx-auto">
          {/* Upload Section */}
          <div className="flex justify-center mb-4 gap-4">
            {/* Exercise Video Upload */}
            <div>
              <input
                type="file"
                accept="video/*"
                onChange={handleFileChange}
                className="hidden"
                id="video-upload"
                disabled={isAnalyzing}
              />
              <label
                htmlFor="video-upload"
                className="bg-gradient-to-r from-blue-500 to-blue-600 text-white px-6 py-3 rounded-full cursor-pointer hover:from-blue-600 hover:to-blue-700 flex items-center gap-2 shadow-lg transition-all"
              >
                📹 Upload Exercise Video
              </label>
            </div>

            {/* Food Photo Upload */}
            <div>
              <input
                type="file"
                accept="image/*"
                onChange={handleFoodFileChange}
                className="hidden"
                id="food-upload"
                disabled={isAnalyzing}
              />
              <label
                htmlFor="food-upload"
                className="bg-gradient-to-r from-green-500 to-green-600 text-white px-6 py-3 rounded-full cursor-pointer hover:from-green-600 hover:to-green-700 flex items-center gap-2 shadow-lg transition-all"
              >
                🍎 Upload Food Photo
              </label>
            </div>
          </div>

          {/* Chat Input */}
          <div className="flex gap-3 items-end">
            <input
              type="text"
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
              placeholder="Ask about your form, technique, nutrition, or request tips..."
              className="flex-1 px-4 py-3 border-2 border-gray-200 rounded-full focus:outline-none focus:border-blue-500 transition-colors text-gray-900 placeholder-gray-500"
              disabled={isAnalyzing}
            />
            <button
              onClick={handleSendMessage}
              disabled={!userInput.trim() || isAnalyzing}
              className="bg-blue-500 text-white p-3 rounded-full hover:bg-blue-600 disabled:opacity-50 transition-colors"
            >
              ➤
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
