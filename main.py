from dotenv import load_dotenv
from importlib.metadata import version

load_dotenv()

core_version = version("langchain-core")
lg_version = version("langgraph")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")

def main():
    # Test Google
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    response = llm.invoke("Say 'setup complete!' in one word")
    print(f"Response from ChatOpenAI: {response}")

    # Test Anthropic
    # llm_anthropic = ChatAnthropic(model="claude-haiku-4.5", temperature=0)
    # response_anthropic = llm_anthropic.invoke("Say 'setup complete!' in one word")
    # print(f"Response from ChatAnthropic: {response_anthropic}")

    print("Setup complete!")


if __name__ == "__main__":
    main()
