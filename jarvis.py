import os
import speech_recognition as sr
import pyttsx3
from langchain_ollama import OllamaLLM
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# --- Voice Functions ---
def speak(text):
    """Converts text to speech."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def take_command():
    """Listens for a voice command and returns it as a string."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}\n")
        return query.lower()
    except Exception as e:
        print("Say that again, please...")
        return "None"

# --- 1. Load the Environment Variables (The Keys) ---
# Now we load the keys securely from your environment variables
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.environ.get('GOOGLE_CSE_ID')

# --- 2. Instantiate the Local LLM (The Brain) ---
llm = OllamaLLM(model="llama3")

# --- 3. Instantiate the Search Tool (The Eyes) ---
search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)
google_search_tool = Tool(
    name="google_search",
    description="Searches Google for recent information.",
    func=search.run
)
tools = [google_search_tool]

# --- 4. Get the Agent Prompt ---
prompt = hub.pull("hwchase17/react")

# --- 5. Create the Agent ---
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# --- 6. The Voice-Enabled Main Loop ---
speak("Hello, I am Jarvis. How can I help you today?")
while True:
    query = take_command()

    if 'exit' in query or 'bye' in query:
        speak("Goodbye, sir.")
        break
    
    if query != "None":
        response = agent_executor.invoke({"input": query})
        speak(response['output'])
        
# --- 7. Print the Final Answer ---
print("\n--- Jarvis's Final Response ---")
print("Session ended.")