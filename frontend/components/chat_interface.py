# frontend/components/chat_interface.py
def render_chat_interface(st):
    """Render the chat interface with history and input."""
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about research documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating..."):
                response = st.session_state.api_client.chat(
                    message=prompt,
                    conversation_history=st.session_state.messages[:-1]
                )

                st.markdown(response["message"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["message"]
                })
                st.session_state.retrieved_documents = response["retrieved_documents"]
                st.rerun()

    # Clear chat button
    if st.button("🔄 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.retrieved_documents = []
        st.rerun()