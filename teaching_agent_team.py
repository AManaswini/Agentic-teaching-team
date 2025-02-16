import streamlit as st
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from composio_phidata import Action, ComposioToolSet
from phi.utils.pprint import pprint_run_response

st.set_page_config(page_title="👨‍🏫 AI Teaching Agent Team")

# Initialize session state for openapi, composio(for google docs integration)
if 'openai_api_key' not in st.session_state:
    st.session_state['openai_api_key'] = ''
if 'composio_api_key' not in st.session_state:
    st.session_state['composio_api_key'] = ''

# Configure API keys in the sidebar
with st.sidebar:
    st.title("API Keys Configuration")
    st.session_state['openai_api_key'] = st.text_input(
        "OpenAI API Key", 
        type="password"
    )
    st.session_state['composio_api_key'] = st.text_input(
        "Composio API Key",
        type="password"
    )

# Stop execution if the keys are missing
if not st.session_state['openai_api_key'] or not st.session_state['composio_api_key']:
    st.error("Please enter all required API keys.")
    st.stop()

# Create Composio ToolSet instance
try:
    composio_toolset = ComposioToolSet(api_key=st.session_state['composio_api_key'])
except Exception as e:
    st.error(f"Error initializing ComposioToolSet: {e}")
    st.stop()

# Get the Google Docs tools
google_docs_tool = composio_toolset.get_tools(
    actions=[Action.GOOGLEDOCS_CREATE_DOCUMENT]
)[0]

# Initialize agents with the OpenAI API key
professor_agent = Agent(
    name="Professor",
    role="Research and Knowledge Specialist", 
    model=OpenAIChat(id="gpt-4o-mini", api_key=st.session_state['openai_api_key']),
    tools=[google_docs_tool],
    instructions=[
        "Create comprehensive knowledge base",
        "Explain from first principles",
        "Include key terminology",
        "Create formatted Google Doc"
    ]
)

topic = st.text_input(
    "Enter topic:",
    placeholder="e.g., Machine Learning"
)

if st.button("Start"):
    if not topic:
        st.error("Please enter a topic")
        st.stop()

    with st.spinner("Generating Knowledge Base..."):
        # Get the professor's response
        professor_response = professor_agent.run(f"topic: {topic}")

        # Extract the Google Docs link from the professor's response
        def extract_google_doc_link(response_content):
            if "https://docs.google.com" in response_content:
                return response_content.split("https://docs.google.com")[1].split()[0]
            return None

        professor_doc_link = extract_google_doc_link(professor_response.content)

        # Show the Google Docs link if available
        st.markdown("### Google Doc Links:")
        if professor_doc_link:
            st.markdown(f"- **Professor Document:** [View](https://docs.google.com{professor_doc_link})")
        else:
            st.markdown("No Google Doc link found.")

        # Display the professor's response
        st.markdown("### Professor Response:")
        st.markdown(professor_response.content)
        pprint_run_response(professor_response, markdown=True)
