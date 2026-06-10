app_code = """
import streamlit as st

st.set_page_config(page_title="Zyro HR Help Desk", page_icon="🏢")
st.title("Zyro Dynamics HR Help Desk")
st.markdown("Welcome! Ask me anything about Zyro Dynamics' HR policies.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("E.g., What is the maternity leave policy?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Note: In a fully deployed separate app, you would import and call ask_bot(prompt) here.
        # For the sake of this notebook generation requirement, we simulate the output area.
        st.info("RAG Backend Integration ready for deployment.")
"""

with open("app.py", "w") as f:
    f.write(app_code.strip())

print("app.py created.")
