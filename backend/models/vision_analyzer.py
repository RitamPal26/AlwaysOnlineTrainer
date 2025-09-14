from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import os

class SquatAnalyzer:
    def __init__(self):
        self.model_id = "LiquidAI/LFM2-VL-450M"
        self.processor = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the model with proper error handling"""
        try:
            print(f"Loading {self.model_id}...")
            
            # Set cache directory (optional - uses default if not set)
            # os.environ['HF_HOME'] = './model_cache'  # Uncomment to use custom cache
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            print("✅ Processor loaded")
            
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                # low_cpu_mem_usage=True,  # Uncomment if you have memory issues
            )
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("Make sure you have internet connection for first-time download")
            raise e
    
    def analyze_squat_form(self, image: Image.Image) -> str:
        """Use detailed prompting instead of fine-tuning"""
        conversation = [{
            "role": "user", 
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": """
                You are an expert fitness trainer analyzing a squat exercise. 
                Look at this image and evaluate:
            
                1. DEPTH: Are thighs parallel to ground or below? 
                2. KNEES: Do knees track over toes? Any inward collapse?
                3. BACK: Is spine neutral? Any excessive forward lean?
                4. FEET: Are feet shoulder-width apart? Proper stance?
            
                Give specific feedback in this format:
                - Good: [what's correct]
                - Fix: [what needs improvement] 
                - Safety: [any injury risks]
            
                Be concise and actionable.
                """}
            ]
        }]
        
        try:
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=150,
                    temperature=0.1,
                    do_sample=True
                )
            
            response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            analysis = response.split("assistant\n")[-1]
            return analysis.strip()
            
        except Exception as e:
            return f"Analysis failed: {str(e)}"
