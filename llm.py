import pathlib
import textwrap
import google.generativeai as genai
import google.generativeai as genai

# API 키 입력
GOOGLE_API_KEY = '__________________'
genai.configure(api_key=GOOGLE_API_KEY)

def LLM(text):
  model = genai.GenerativeModel('gemini-pro')
  
  while True:
    user_input = text
    if user_input == "q":
      break
    else:
      instruction = """
      You are an advanced AI model specialized in understanding and interpreting detailed clothing descriptions. 
      Your task is to accurately extract the clothing details from the user's request and output them in a structured JSON format.
      The output should clearly specify the type, color, and style of the clothing items.
      
      IMPORTANT: 
      - Remove any existing clothing on the subject and replace it entirely with the clothing specified by the user. 
      - If the user describes additional features (e.g., fabric type, fit, patterns), these should be captured in the JSON format as well.
      - Ensure that the description is detailed, clear, and only reflects the new clothing items as instructed by the user.
      """
      
      prompt = f"""
      Examples:
      User: "I want a red slim-fit shirt with a round neck and blue ripped jeans."
      You: {{"top": ["red", "slim-fit shirt", "round neck"], "bottom": ["blue", "ripped jeans"]}}

      User: "She is wearing a green sleeveless dress made of silk."
      You: {{"top": ["", ""], "bottom": ["green", "sleeveless dress", "silk"]}}

      User: "He wants a black leather jacket with a zipper and blue denim jeans."
      You: {{"top": ["black", "leather jacket", "zipper"], "bottom": ["blue", "denim jeans"]}}

      User: {text}
      You: 
      """

      full_prompt = f"{instruction}\n{prompt}"

      response = model.generate_content(full_prompt)

      return response
