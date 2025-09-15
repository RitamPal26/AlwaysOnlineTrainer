# 🏋️‍♂️ AlwaysOnlineTrainer - AI Fitness & Nutrition Coach

> **Your personal AI trainer that never sleeps** - Combining cutting-edge vision AI with conversational coaching for the ultimate fitness experience.

[![Built for Liquid Hack #02](https://img.shields.io/badge/Liquid%20Hack%20%2302-LFMs%20with%20Eyes-blue?style=for-the-badge)](https://hackathons.liquid.ai/register)

<img width="550" height="425" alt="image" src="https://github.com/user-attachments/assets/b86b15b8-57bb-4329-b50e-07b056b71b0b" />


## 🎯 What Makes This Special?

**AlwaysOnlineTrainer** isn't just another fitness app - it's an always-on AI companion that **sees, understands, and coaches** you through your fitness journey using state-of-the-art vision models.

### ✨ The Magic Behind the Scenes

```
🎥 Upload Video → 👁️ LFM2-VL Vision Analysis → 🧠 Llama 3.1 Coaching → 💬 Personalized Feedback
📸 Food Photo → 👁️ Image Recognition → 🥗 Nutrition Analysis → 📋 Custom Meal Advice
```

## 🚀 Key Features

### 🎥 **Intelligent Exercise Analysis**
- **Multi-frame Video Processing**: Analyzes key movement phases (every 3 seconds)
- **Holistic Form Assessment**: Combines multiple frames for comprehensive feedback
- **Exercise Detection**: Automatically identifies squats, deadlifts, bench press, and more
- **Professional Coaching**: Converts technical analysis into actionable advice

### 🍎 **Smart Nutrition Guidance** 
- **Food Recognition**: Upload meal photos for instant nutritional analysis
- **Calorie Estimation**: AI-powered portion and calorie assessment
- **Personalized Advice**: Tailored nutrition tips based on your fitness goals
- **Balanced Meal Suggestions**: Recommends improvements for optimal nutrition

### 💬 **Conversational AI Coach**
- **24/7 Availability**: Always ready to answer fitness and nutrition questions
- **Contextual Responses**: Remembers your goals and provides relevant advice
- **Natural Language**: Chat like you would with a real personal trainer
- **Motivational Support**: Encouraging feedback to keep you on track

## 🛠️ Technical Architecture

### **Multi-Model AI Pipeline**
```
graph LR
    A[User Input] --> B[LFM2-VL Vision Model]
    B --> C[CloudRift Llama 3.1 70B]
    C --> D[Personalized Coaching]
    D --> E[Conversational Interface]
```

### **Tech Stack**
- **🎨 Frontend**: Next.js 14, React 18, TypeScript, Tailwind CSS
- **⚡ Backend**: FastAPI, Python 3.9+, Async processing  
- **🤖 AI Models**: 
  - **Vision**: LiquidAI LFM2-VL-450M (local processing)
  - **Language**: Meta Llama 3.1 70B via CloudRift API
- **📱 UI/UX**: Modern chat interface with real-time updates
- **🔧 Processing**: OpenCV, Pillow, Multi-frame sampling

## 🏃‍♂️ Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- CloudRift API key ([Get one here](https://console.cloudrift.ai/))

### 🔧 Backend Setup
```
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your CLOUDRIFT_API_KEY to .env

# Launch the AI backend
python main.py
```

### 🌐 Frontend Setup  
```
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000` and start training with your AI coach! 🎉

## 🎬 Live Demo & Screenshots

### **🎥 Demo Video**
> **Watch AlwaysOnlineTrainer in action!** See real-time exercise analysis and nutrition coaching.

https://vimeo.com/1118607256?share=copy

*2-minute demo showcasing exercise form analysis, food nutrition guidance, and conversational AI coaching*

---

### **📸 App Screenshots**

<div align="center">

#### **💬 Conversational AI Coach**
<img width="1123" height="859" alt="Screenshot 2025-09-14 190544" src="https://github.com/user-attachments/assets/7b6080a8-3add-49a9-96dc-b751e7c0e2c6" />

*Natural conversation with your always-on AI fitness coach*

#### **🍎 Smart Nutrition Guidance**
<img width="1000" height="667" alt="Screenshot 2025-09-14 190955" src="https://github.com/user-attachments/assets/f509aded-6698-49d2-b4ff-a365c2a3ec37" />

*Upload meal photos for instant nutritional advice*

</div>

---

## 🏆 Why This Wins

### **Innovation Highlights:**
- **🎯 Perfect Theme Alignment**: "Always-on AI agents" with 24/7 availability
- **👁️ Advanced Vision Processing**: Multi-frame analysis beats single-image approaches  
- **🧠 Sophisticated AI Pipeline**: Two specialized models working in harmony
- **🎨 Exceptional UX**: Chat-based interface feels like talking to a real trainer
- **⚡ Real-world Application**: Solves actual fitness coaching accessibility problems

### **Technical Achievements:**
- **Holistic Video Analysis**: Novel approach combining temporal sampling with contextual understanding
- **Privacy-First Design**: Vision processing happens locally, personal data stays secure
- **Scalable Architecture**: FastAPI backend can handle multiple concurrent users
- **Production Ready**: Proper error handling, environment management, and deployment setup

## 🔮 Future Enhancements

- **📊 Progress Tracking**: Long-term fitness journey monitoring
- **🏃‍♀️ Live Workout Sessions**: Real-time form correction during exercises  
- **👥 Social Features**: Share achievements and compete with friends
- **📱 Mobile App**: Native iOS/Android applications
- **🔗 Wearable Integration**: Sync with fitness trackers and smartwatches

## 🤝 Contributing

This project was built for **Liquid Hack #02** with the theme "LFMs with Eyes." We're excited to continue development post-hackathon!

### **Getting Involved:**
1. Fork the repository
2. Create a feature branch
3. Make your improvements  
4. Submit a pull request

## 📄 License

MIT License - Feel free to use this code for your own AI fitness innovations!

## 🏅 Hackathon Details

- **Event**: Liquid Hack #02 - LFMs with Eyes
- **Theme**: Always-on AI Agents with Vision Capabilities
- **Built by**: Ritam Pal
- **Timeline**: 36 hours of intensive development

---

**Ready to revolutionize your fitness journey? Start training with AlwaysOnlineTrainer today!** 🚀

*Built with ❤️ using cutting-edge AI technology*
```
