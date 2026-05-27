from dotenv import load_dotenv
import os
import openai

# Load your key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client (new style)
client = openai.OpenAI(api_key=api_key)

# Test the API
try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a test assistant."},
            {"role": "user", "content": "Is this API key working?"}
        ],
        max_tokens=30
    )

    print("✅ API key is working!")
    print("🔁 Sample response:", response.choices[0].message.content)

except openai.AuthenticationError:
    print("❌ Invalid or expired API key.")
except Exception as e:
    print("⚠️ Some other error occurred:", e)
