from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import cv2
import numpy as np
import io
import os
import tempfile
import time
import openai
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

app = FastAPI(title="Always-On Fitness Coach")

# CORS for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables (load once, use many times)
processor = None
model = None
device = None

def get_cloudrift_client():
    """Get CloudRift client with proper error handling"""
    api_key = os.getenv("CLOUDRIFT_API_KEY")
    if not api_key:
        raise Exception("CLOUDRIFT_API_KEY not found in .env file")
    
    return openai.OpenAI(
        api_key=api_key,
        base_url="https://inference.cloudrift.ai/v1"
    )

@app.on_event("startup")
async def load_model():
    """Load model on startup and test CloudRift connection"""
    global processor, model, device
    
    print("🤖 Loading LFM2-VL-450M...")
    model_id = "LiquidAI/LFM2-VL-450M"
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"✅ Model loaded on {device}")
    
    # Test CloudRift connection
    try:
        client = get_cloudrift_client()
        test_completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-FP8",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        print("✅ CloudRift connection successful")
    except Exception as e:
        print(f"⚠️ CloudRift connection failed: {e}")
        print("💡 Make sure CLOUDRIFT_API_KEY is set in your .env file")

def refine_with_coaching_llm(exercise_type: str, form_analysis: str, frame_timestamp: float) -> str:
    """Enhanced coaching with exercise type context"""
    
    coaching_prompt = f"""You are an expert fitness coach. Here's what happened at {frame_timestamp}s:

EXERCISE IDENTIFIED: {exercise_type}
FORM ANALYSIS: {form_analysis}

Provide specific coaching feedback:

1. If it's squats: Give squat-specific advice (depth, knees, core)
2. If it's running/walking: Give cardio form tips (posture, stride) 
3. If it's another exercise: Give relevant coaching for that movement
4. If unclear: Encourage them and ask for a clearer video

Be encouraging, specific, and actionable. Speak directly to the person like their personal trainer."""

    try:
        client = get_cloudrift_client()
        completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-FP8",
            messages=[
                {"role": "user", "content": coaching_prompt}
            ],
            max_tokens=120,
            temperature=0.6
        )
        
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"CloudRift error: {e}")
        return f"Great movement at {frame_timestamp}s! Keep up the good work! 💪"

def detect_exercise_type(image: Image.Image) -> str:
    """First pass: Clearly identify what exercise is happening"""
    
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": """Look at this image and identify the specific exercise or activity:

            Is the person:
            - Doing squats (bent knees, lowering body)
            - Running or walking (upright, forward motion)
            - Standing still/preparing for exercise
            - Doing push-ups, lunges, or other exercise
            - Not exercising at all
            
            Give a simple, clear answer about what activity you see."""}
        ]
    }]
    
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    try:
        if "assistant\n" in response:
            return response.split("assistant\n")[-1].strip()
        else:
            return str(response).strip()
    except:
        return "unclear activity"

def analyze_exercise_form(image: Image.Image, exercise_type: str) -> str:
    """Second pass: Detailed form analysis for the identified exercise"""
    
    if "squat" in exercise_type.lower():
        analysis_prompt = """Analyze this squat exercise focusing on:
        - Knee alignment and depth
        - Back position and core engagement  
        - Foot placement and balance
        - Overall squat form quality"""
    elif any(word in exercise_type.lower() for word in ["running", "walking", "cardio"]):
        analysis_prompt = """Analyze this running/walking form focusing on:
        - Posture and stride
        - Arm movement and rhythm
        - Overall cardio form"""
    else:
        analysis_prompt = f"""Analyze this {exercise_type} focusing on:
        - Body positioning and alignment
        - Movement quality and control
        - Any form improvements needed"""
    
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": analysis_prompt}
        ]
    }]
    
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150)
    
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    try:
        if "assistant\n" in response:
            return response.split("assistant\n")[-1].strip()
        else:
            return str(response).strip()
    except:
        return "Form analysis completed"

def analyze_frame(image: Image.Image, frame_num: int, timestamp: float) -> str:
    """Enhanced two-step analysis pipeline"""
    
    # Step 1: Identify the exercise type clearly
    exercise_type = detect_exercise_type(image)
    print(f"🔍 Detected exercise: {exercise_type}")
    
    # Step 2: Detailed form analysis for that specific exercise
    form_analysis = analyze_exercise_form(image, exercise_type)
    
    # Step 3: CloudRift coaching refinement
    coaching_feedback = refine_with_coaching_llm(exercise_type, form_analysis, timestamp)
    
    # In your analyze_frame function, add:
    print(f"🔍 Frame {frame_num} - Exercise type detected: {exercise_type}")
    print(f"📝 Form analysis: {form_analysis[:100]}...")

    return coaching_feedback

@app.post("/chat")
async def chat_with_coach(request: dict):
    """Handle simple chat queries with CloudRift"""
    
    message = request.get("message", "")
    
    coaching_prompt = f"""You are an enthusiastic, knowledgeable fitness coach. Answer this question with encouraging, specific advice:

Question: {message}

Give a helpful, motivational response as if you're a personal trainer. Keep it conversational and under 100 words."""

    try:
        client = get_cloudrift_client()
        completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-FP8",
            messages=[
                {"role": "user", "content": coaching_prompt}
            ],
            max_tokens=120,
            temperature=0.8
        )
        
        coach_response = completion.choices[0].message.content
        return {"response": coach_response.strip()}
    
    except Exception as e:
        print(f"CloudRift chat error: {e}")
        return {"response": "Great question! Upload a video and I can give you specific feedback based on what I see! 💪"}

def analyze_video_holistically(frame_analyses: list, duration: float) -> str:
    """Analyze all frames together to determine the actual activity"""
    
    # Combine all frame analyses for context
    all_detections = [analysis['exercise_type'] for analysis in frame_analyses]
    all_descriptions = [analysis['form_analysis'] for analysis in frame_analyses]
    
    # Create comprehensive context for CloudRift
    context_prompt = f"""You are an expert fitness coach analyzing a {duration}s video with multiple frame observations:

FRAME ANALYSES:
"""
    
    for i, analysis in enumerate(frame_analyses):
        context_prompt += f"Frame {i+1} ({analysis['timestamp']}s): {analysis['exercise_type']} - {analysis['form_analysis'][:150]}...\n"
    
    context_prompt += f"""

Based on these {len(frame_analyses)} frame observations across {duration}s:

1. What is the person ACTUALLY doing overall? (ignore inconsistent single-frame detections)
2. Are they exercising or just standing/moving casually?
3. If exercising: What type and how is their overall form?
4. If not exercising: What activity are they doing?

Give ONE coherent coaching response as if you watched the whole video, not individual frames. Be encouraging and specific."""

    try:
        client = get_cloudrift_client()
        completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-FP8",
            messages=[
                {"role": "user", "content": context_prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Holistic analysis error: {e}")
        return f"I analyzed your {duration}s video across multiple moments. Keep up the great work with your movement! 💪"

def analyze_food_image(image: Image.Image) -> dict:
    """Analyze food image using LFM2-VL"""
    
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": """Analyze this food image and provide:

1. Identify the specific food items you can see
2. Estimate portion sizes (small/medium/large or specific measurements)
3. Estimate total calories for what's shown
4. Key nutritional information (protein, carbs, fats if relevant)
5. Overall healthiness assessment

Be specific about what you can actually see in the image."""}
        ]
    }]
    
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200)
    
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    try:
        if "assistant\n" in response:
            food_analysis = response.split("assistant\n")[-1].strip()
        else:
            food_analysis = str(response).strip()
    except Exception as e:
        food_analysis = f"Food analysis error: {str(e)}"
    
    return food_analysis

def refine_food_coaching(food_analysis: str) -> str:
    """Use CloudRift to provide nutrition coaching"""
    
    coaching_prompt = f"""You are a nutrition coach analyzing this food analysis:

{food_analysis}

Provide encouraging, practical nutrition advice:

1. Comment on the food choices shown
2. Suggest improvements if needed
3. Give practical tips for healthy eating
4. Be motivational and supportive
5. Keep it concise and actionable

Respond as if you're a personal nutrition coach."""

    try:
        client = get_cloudrift_client()
        completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-FP8",
            messages=[
                {"role": "user", "content": coaching_prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Food coaching error: {e}")
        return "Great food choices! Keep focusing on balanced nutrition to support your fitness goals."

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    """Enhanced video analysis with holistic understanding"""
    
    print(f"📄 Received file: {file.filename}") 
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        cap = cv2.VideoCapture(tmp_file_path)
        if not cap.isOpened():
            return {"status": "error", "message": "Could not open video file"}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"📹 Video: {duration:.1f}s, {fps:.0f} FPS, {total_frames} frames")
        
        # Collect frame analyses (but don't return them individually)
        frame_analyses = []
        
        # Smart sampling: Every 3 seconds for better context
        interval_seconds = 3.0
        frame_interval = int(fps * interval_seconds)
        
        frame_positions = []
        current_frame = 0
        
        while current_frame < total_frames:
            frame_positions.append(current_frame)
            current_frame += frame_interval
        
        # Limit to max 4 frames for processing speed
        frame_positions = frame_positions[:4]
        
        print(f"🔍 Analyzing frames at: {[f/fps for f in frame_positions]} seconds")
        
        for frame_pos in frame_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                pil_image = pil_image.resize((640, 480), Image.Resampling.LANCZOS)
                
                timestamp = round(frame_pos / fps, 1)
                
                # Get individual frame analysis (for context, not user display)
                exercise_type = detect_exercise_type(pil_image)
                form_analysis = analyze_exercise_form(pil_image, exercise_type)
                
                frame_analyses.append({
                    "timestamp": timestamp,
                    "exercise_type": exercise_type,
                    "form_analysis": form_analysis
                })
                
                print(f"✅ Collected context from frame at {timestamp}s")
        
        cap.release()
        
        # Generate ONE holistic coaching response
        holistic_coaching = analyze_video_holistically(frame_analyses, duration)
        
        return {
            "status": "success",
            "message": f"Analyzed {duration:.1f}s video with holistic AI coaching",
            "duration": round(duration, 1),
            "frames_analyzed": len(frame_analyses),
            "coaching_response": holistic_coaching,  # Single response
            "pipeline": "Multi-frame Context → Holistic CloudRift Analysis"
        }
    
    finally:
        try:
            time.sleep(0.1)
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        except PermissionError:
            pass

@app.post("/analyze-food")
async def analyze_food(file: UploadFile = File(...)):
    """Analyze food image for nutrition coaching"""
    
    print(f"🍎 Received food image: {file.filename}")
    
    try:
        # Read and process image
        image_content = await file.read()
        pil_image = Image.open(io.BytesIO(image_content))
        
        # Ensure good quality for analysis
        pil_image = pil_image.resize((640, 480), Image.Resampling.LANCZOS)
        
        # Analyze food with LFM2-VL
        food_analysis = analyze_food_image(pil_image)
        print(f"🔍 Food analysis: {food_analysis[:100]}...")
        
        # Get coaching advice with CloudRift
        nutrition_coaching = refine_food_coaching(food_analysis)
        
        return {
            "status": "success",
            "message": "Food analysis complete",
            "analysis": nutrition_coaching,
            "pipeline": "LFM2-VL Food Recognition → CloudRift Nutrition Coaching"
        }
        
    except Exception as e:
        print(f"Food analysis error: {e}")
        return {
            "status": "error",
            "message": f"Could not analyze food image: {str(e)}"
        }

@app.get("/")
async def root():
    return {
        "message": "🏋️‍♂️ Always-On Fitness Coach API", 
        "status": "ready",
        "model": "LFM2-VL-450M → CloudRift Llama 3.1 70B",
        "env_loaded": "✅" if os.getenv("CLOUDRIFT_API_KEY") else "❌",
        "cloudrift_ready": "✅" if os.getenv("CLOUDRIFT_API_KEY") else "❌ - Add API key to .env"
    }

@app.get("/test-cloudrift")
async def test_cloudrift():
    """Test CloudRift connection"""
    try:
        client = get_cloudrift_client()
        completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-FP8",
            messages=[
                {"role": "user", "content": "Say hello in one word"}
            ],
            max_tokens=10
        )
        
        return {
            "status": "success",
            "response": completion.choices[0].message.content,
            "api_key_set": bool(os.getenv("CLOUDRIFT_API_KEY"))
        }
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "api_key_set": bool(os.getenv("CLOUDRIFT_API_KEY"))
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
