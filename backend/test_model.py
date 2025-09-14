from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import cv2
import os

def test_video_analysis():
    print("🎬 Testing LFM2-VL with video frames...")
    
    # Load the model (reusing your working setup)
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
    
    # Video file path (you'll need to add a video file)
    video_path = "test_video.mp4"  # Put a short video here
    
    if not os.path.exists(video_path):
        print("❌ Please add a test video file named 'test_video.mp4' in the backend folder")
        print("You can use any short video (exercise, walking, anything)")
        return
    
    # Extract and analyze frames
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    print(f"📹 Processing video: {video_path}")
    
    while cap.read()[0] and frame_count < 5:  # Test first 5 frames only
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB (OpenCV uses BGR, PIL uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Analyze this frame
        print(f"\n🔍 Analyzing frame {frame_count + 1}...")
        
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": "Describe what you see in this video frame. Focus on any people, their posture, and activities."}
            ]
        }]
        
        try:
            inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            ).to(device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100)
            
            response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            analysis = response.split("assistant\n")[-1]
            
            print(f"📝 Frame {frame_count + 1}: {analysis}")
            
        except Exception as e:
            print(f"❌ Error analyzing frame {frame_count + 1}: {e}")
        
        frame_count += 1
    
    cap.release()
    print(f"\n✅ Video test complete! Processed {frame_count} frames")

if __name__ == "__main__":
    test_video_analysis()
