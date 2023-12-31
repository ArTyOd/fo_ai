import streamlit as st
import openai
from ai_main import (
    load_index,
    answer_question,
    clear_chat_history,
)  # Import clear_chat_history function
import json
import pinecone
import pandas as pd
from google.oauth2 import service_account
from google.cloud import storage

from ai.main import initialize_process_request
from ai.llm_interactions import answer_question_streaming


pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone_environment = st.secrets["PINECONE_environment"]
openai_api_key = st.secrets["OPENAI_API_KEY"]

pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
openai.api_key = openai_api_key

index = load_index()
updated_stream = ""

# Create API client for Google Cloud Storage
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = storage.Client(credentials=credentials)

# Bucket name and file path
bucket_name = "bucket_g_cloud_service_1"
folder_path = "fo"
config_file_name = "dataframes.json"
process_request = initialize_process_request(bucket_name, folder_path, config_file_name)


def load_from_gcs(bucket_name, file_path):
    """Load a file from Google Cloud Storage."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    content = blob.download_as_text()  # or use download_as_bytes for binary files
    return json.loads(content)


def save_to_gcs(bucket_name, file_path, content):
    """Save a file to Google Cloud Storage."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    content = json.dumps(content)
    blob.upload_from_string(content, content_type="application/json")


unique_categories_file_path = "fo/unique_categories.json"
instructions_file_path = "fo/instructions.json"

# Load unique categories from GCS
unique_categories = load_from_gcs(bucket_name, unique_categories_file_path)

# Load instructions from GCS
instructions = load_from_gcs(bucket_name, instructions_file_path)

# def load_unique_categories():
#     with open("data/unique_categories.json", "r") as f:
#         return json.load(f)

# unique_categories = load_unique_categories()

# def load_instructions():
#     with open("data/instructions.json", "r") as f:
#         return json.load(f)

# instructions = load_instructions()

# def save_instructions(instructions):
#     with open("data/instructions.json", "w") as f:
#         json.dump(instructions, f, indent=2)


def app():
    st.title("Felgenoutlet Assistant")

    st.sidebar.header("How it Works")
    st.sidebar.write(
        "This AI Assistant uses GPT-3.5 to answer questions based on a chosen set of instructions and categories. Customize GPT-3.5 parameters and select categories to refine the AI's responses."
    )

    # Add a separator
    st.sidebar.markdown(
        "<hr style='height: 1px; border: none; background-color: gray; margin-left: -10px; margin-right: -10px;'>",
        unsafe_allow_html=True,
    )

    # GPT parameter fields

    st.sidebar.subheader("GPT Parameters")
    # Add GPT model selection buttons
    model_col1, model_col2 = st.sidebar.columns(2)
    with model_col1:
        gpt_35_button = st.button("GPT-3.5")
    with model_col2:
        gpt_4_button = st.button("GPT-4")
    # Store the selected model in session state
    if gpt_35_button:
        st.session_state.selected_model = "gpt-3.5-turbo"
    elif gpt_4_button:
        st.session_state.selected_model = "gpt-4"

    # Default to GPT-3.5 if no model is selected
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gpt-3.5-turbo"

    st.sidebar.write(f"Selected model: {st.session_state.selected_model}")
    max_token_question = st.sidebar.number_input("Max tokens (question):", min_value=1, value=1500)
    max_token_answer = st.sidebar.number_input("Max tokens (answer):", min_value=1, value=500)
    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=0.3)
    reframing = st.sidebar.checkbox("Enable reframing questions", value=False)

    # Add a separator
    st.sidebar.markdown(
        "<hr style='height: 1px; border: none; background-color: gray; margin-left: -20px; margin-right: -20px;'>",
        unsafe_allow_html=True,
    )

    # Token usage
    st.sidebar.subheader("Token Usage")

    with st.form(key="input_form", clear_on_submit=True):
        user_input = st.text_input("Ask a question:", key="ask_question")
        button_col1, button_col2 = st.columns(2)
        with button_col1:
            send_button = st.form_submit_button("Send")

    search_options_expander = st.expander("Search Options")
    with search_options_expander:
        selected_instruction = st.radio("Instructions", list(instructions.keys()))

        edit_instructions = st.checkbox("Edit instructions")
        if edit_instructions:
            instruction_key = st.selectbox("Instruction key:", list(instructions.keys()), index=0)  # Change to selectbox
            instruction_value = st.text_area("Instruction value:", value=instructions[instruction_key])  # Add value

            button_row = st.columns(2)
            with button_row[0]:
                update_button = st.button("Update")
            with button_row[1]:
                delete_button = st.button("Delete")

            add_new_key = st.text_input("Add new instruction key:")  # Add new input for a new instruction key
            add_new_value = st.text_area("Add new instruction value:")  # Add new input for a new instruction value
            add_button = st.button("Add")  # Add button for adding new instruction

            if update_button:
                instructions[instruction_key] = instruction_value
                save_to_gcs(bucket_name, instructions_file_path, instructions)
                st.success(f"Updated instruction: {instruction_key}")
                st.experimental_rerun()

            if delete_button and instruction_key in instructions:
                del instructions[instruction_key]
                save_to_gcs(bucket_name, instructions_file_path, instructions)
                st.success(f"Deleted instruction: {instruction_key}")
                st.experimental_rerun()

            if add_button:
                if add_new_key and add_new_value:  # Check if the new instruction key and value are not empty
                    instructions[add_new_key] = add_new_value
                    save_to_gcs(bucket_name, instructions_file_path, instructions)
                    st.success(f"Added new instruction: {add_new_key}")
                    st.experimental_rerun()
                else:
                    st.warning("Please provide both a key and a value for the new instruction.")

        checked_categories = get_checked_categories(unique_categories)

    if send_button:
        placeholder_response = st.empty()
        chat_container = st.container()
        prompt_tokens, completion_tokens, total_tokens = update_chat(
            user_input,
            selected_instruction,
            checked_categories,
            chat_container,
            placeholder_response,
            max_token_question,
            max_token_answer,
            temperature,
            reframing,
            st.session_state.selected_model,
        )

        # Update the token count in the sidebar
        st.sidebar.write(f"Tokens used for prompt: {prompt_tokens}")
        st.sidebar.write(f"Tokens used for completion: {completion_tokens}")
        st.sidebar.write(f"Total tokens: {total_tokens}")
        if "total_chat_tokens" not in st.session_state:
            st.session_state.total_chat_tokens = 0
        st.session_state.total_chat_tokens += total_tokens
        st.sidebar.write(f"Total tokens in chat session: {st.session_state.total_chat_tokens}")

    # Add a New Session/Chat button in app.py
    if st.button("New Session/Chat"):
        clear_chat_history()
        st.session_state.chat_history = []


def get_checked_categories(unique_categories):
    checked_categories = []
    for key in unique_categories:
        st.write(key)
        col1, col2, col3 = st.columns(3)
        for i, category in enumerate(unique_categories[key]):
            checked = category in checked_categories
            if not checked:
                checked_categories.append(category)
                checked = True
            if i % 3 == 0:
                checked = col1.checkbox(category, value=checked, key=f"{category}_checkbox")
            elif i % 3 == 1:
                checked = col2.checkbox(category, value=checked, key=f"{category}_checkbox")
            else:
                checked = col3.checkbox(category, value=checked, key=f"{category}_checkbox")
            if checked and category not in checked_categories:
                checked_categories.append(category)
            elif not checked and category in checked_categories:
                checked_categories.remove(category)
    return checked_categories


def update_chat(
    user_input,
    selected_instruction,
    checked_categories,
    chat_container,
    placeholder_response,
    max_token_question,
    max_token_answer,
    temperature,
    reframing,
    selected_model,
):
    if user_input:
        print(f"{selected_model = }")
        updated_stream = ""

        prompt, answer_instruction = process_request(
            text=user_input,
            instruction=instructions,
            debug=False,
            max_tokens=max_token_answer,
            max_len=max_token_question,
            temperature=temperature,
        )

        st.session_state.chat_history, prompt_tokens, completion_tokens, total_tokens, response = answer_question_streaming(
            model="gpt-4",
            instruction=answer_instruction,
            prompt=prompt,
            debug=False,
            max_tokens=max_token_answer,
            temperature=temperature,
            callback=lambda text: display_stream_answer(text, placeholder_response),
        )

        # display_context_details(context_details)
        display_chat(st.session_state.chat_history[1:-1], chat_container)
        return prompt_tokens, completion_tokens, total_tokens

    # print(f"{context_details =}")
    return 0, 0, 0


def display_stream_answer(r_text, placeholder_response):
    global updated_stream
    stream_text = ""
    updated_stream += r_text
    stream_text += f'<div style="background-color: #0d1116; margin: 0; padding: 10px;">{updated_stream}</div>'
    placeholder_response.markdown(stream_text, unsafe_allow_html=True)


def display_chat(chat_history, chat_container):
    chat_text = ""
    for entry in reversed(chat_history):
        if entry["role"] == "user":
            chat_text += f'<div style="background-color: #262730 ; margin: 0; padding: 10px;">{entry["content"]}</div>'
        else:
            chat_text += f'<div style="background-color: #0d1116; margin: 0; padding: 10px;">{entry["content"]}</div>'
    chat_container.write(
        f"""
    <div id="chatBox" style="height: 300px; overflow-y: scroll; ">
        {chat_text}
    </div>
    """,
        unsafe_allow_html=True,
    )


def display_context_details(context_details):
    context_details_expander = st.expander("Context Details")
    with context_details_expander:
        print(f"{context_details = }")
        # Convert context details to a Pandas DataFrame
        df_context_details = pd.DataFrame(context_details)
        # Transform the score into a percentage with two decimal places
        df_context_details["score"] = df_context_details["score"].apply(lambda x: f"{x * 100:.2f}%")
        df_context_details["token"] = df_context_details["token"].apply(lambda x: f"{x:.0f}")
        df_context_details = df_context_details.sort_values(by="score", ascending=False)
        # Display the DataFrame as a table
        st.table(df_context_details[1:])


if __name__ == "__main__":
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    app()
