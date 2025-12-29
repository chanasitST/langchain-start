import uuid
from agent import app

def run_chat(thread_id: str, user_input: str):

    print(f"\n💬 User ({thread_id}): {user_input}")
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Run the agent
    # stream_mode="values" returns the full state at each step
    for event in app.stream({"question": user_input}, config=config, stream_mode="values"):
        if "answer" in event and event["answer"]:
            print(f"🤖 Agent: {event['answer']}")

def main():
    # Step 1 create a thread ID for user session
    thread_id = str(uuid.uuid4())
    
    # Step 2 First interaction: Save a fact
    run_chat(thread_id, "Remember that my name is Bob and i suck at coding")

    # Step 3 Recall from the memory
    run_chat(thread_id, "What is my name?")

    # Step 4 
    thread_id2 = str(uuid.uuid4())
    print(f"\n💬 User ({thread_id2}): What is my name?")

if __name__ == "__main__":
    main()
