# user_app.py
import streamlit as st
import core_functions as cf  # Import the shared functions
import time
import random
import streamlit.components.v1 as components

def animated_text(text, speed=0.001):
    """Displays text with a typing animation."""
    placeholder = st.empty()
    full_text = ""
    for char in text:
        full_text += char
        placeholder.markdown(full_text)
        time.sleep(speed)
    return placeholder  # Return the placeholder for potential later updates

def clear_chat_history():
    """Clears the conversation history from Streamlit session state."""
    st.session_state.conversation_history = []

def main():
    # --- Inject Custom CSS ---
    st.markdown(
        """
        <style>
            #GithubIcon {
              visibility: hidden;
                }

            .user-message {
                
                margin-left: 50%; /* Push to the right */
                background-color: #acb7ff; /* background */
                padding: 8px;
                color: #000;
                border-radius: 8px;
                margin-bottom: 10px;
                margin-top: 10px;
            }
            .ai-message {
                text-align: left;
                background-color: #303030; /* background */
                padding: 8px;
                border-radius: 8px;
                margin-bottom: 5px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Fixed Title at the Top ---
    st.title("What Can I Help With")

    # --- Initialize Conversation History in Session State ---
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # --- Fixed Question Input and Clear Button at the Bottom ---
    col1, col2 = st.columns([0.8, 0.2])  # Divide the bottom area into two columns
    with col1:
        question = st.chat_input("Message Penda Support:", key="question_input")  # Unique key for the input
    with col2:

        st.button("Clear Chat", on_click=clear_chat_history, icon=":material/delete:", type="primary")  # Add clear chat button

    # --- Chat Interface Elements ---
    chat_area = st.container(height=350)  # holds messages for scrolling

    if question:
        # --- Add User Question to Conversation History ---
        st.session_state.conversation_history.append({"role": "user", "content": question})

        with st.spinner("Thinking...", show_time=True):
            try:
                # --- Build Context with Conversation History ---
                history_text = "\n".join([f"{entry['role']}: {entry['content']}" for entry in st.session_state.conversation_history[-5:]])  # Use last 5 turns
                query_embedding = cf.generate_embedding(question)
                if query_embedding is None:
                    st.error("Failed to generate embedding for the question.")
                    return

                conn = cf.create_connection()  # Create connection for querying
                similar_chunks = cf.search_similar_chunks(conn, query_embedding)
                conn.close()  # Close connection after querying

                context = "\n".join([chunk[1] for chunk in similar_chunks])

                #Combine context with history
                combined_context = f"{history_text}\nRelevant context:{context}" # Combine History with Knowledge

                answer = cf.generate_response(combined_context, question)

                # --- Add AI Response to Conversation History ---
                st.session_state.conversation_history.append({"role": "ai", "content": answer})

            except Exception as e:
                st.error(f"Error generating response: {e}")
                answer = "Error generating response. Please try again." # For smoother execution

    # --- Display Chat History and Auto-Scroll to Latest Response ---
    with chat_area:
        for entry in st.session_state.conversation_history:
            if entry["role"] == "user":
                st.markdown(f"<div class='user-message'>{entry['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='ai-message'>{entry['content']}</div>", unsafe_allow_html=True)

        # Add the scroll-to-bottom div
        html_code = f"""
        <div id="scroll-to-bottom" style='height: 1px;'></div>
        <script>
           var element = document.getElementById("scroll-to-bottom");
           if (element) {{
               element.scrollIntoView({{ behavior: "smooth", block: "end", inline: "nearest" }});
           }}
        </script>
        """
        st.components.v1.html(html_code, height=10)  # Ensure there's some minimal height

if __name__ == "__main__":
    main()
