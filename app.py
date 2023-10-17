import streamlit as st
import openai
import json
import pinecone
import pandas as pd
from google.oauth2 import service_account
from google.cloud import storage
from datetime import datetime
import time

from ai.main import initialize_process_request


from ai.context_retrieval import context_retrieval_entry
from ai.embed_context import embed_context_entry
from ai.document_retrieval import gather_contexts, create_context_pinecone
from ai.llm_interactions import answer_question, answer_question_streaming
from ai.initialize import initialize_all_dataframes
from ai.utils import load_from_gcs, save_to_gcs, load_index, serialize_all_columns

openai_api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = openai_api_key

st.set_page_config(layout="wide")
# global
updated_stream = ""

pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone_environment = st.secrets["PINECONE_environment"]
pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
pinecone_index = load_index(index_name="fo", pinecone_api_key=pinecone_api_key, pinecone_environment=pinecone_environment)

# Create API client for Google Cloud Storage
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = storage.Client(credentials=credentials)

# Bucket name and file path
bucket_name = "bucket_g_cloud_service_1"
folder_path = "fo"
config_file_name = "dataframes.json"
# process_request = initialize_process_request(bucket_name, folder_path, config_file_name)


instructions = load_from_gcs(bucket_name, "fo/instructions.json")

dfs = initialize_all_dataframes(bucket_name, folder_path, config_file_name)

st.session_state.df_file_paths = {df_name: f"{folder_path}/{df_name}.json" for df_name in dfs.keys()}
st.session_state.df_requests = dfs["df_requests"]
st.session_state.df_rim_details = dfs["df_rim_details"]
# st.session_state.df_all_rims = dfs["df_all_rims"]
# st.session_state.df_all_rims = pd.read_pickle("data/df_all_rims.pkl")


# initializing st.session.states:
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "submit_clicked" not in st.session_state:
    st.session_state.submit_clicked = False


def content_phase_1(main_container):
    """Render the content for Phase 1."""
    with main_container:
        with st.form(key="input_form", clear_on_submit=True):
            user_input = st.text_area("Kundenmail hier einfügen:", key="ask_question", height=300)
            button_col1, button_col2 = st.columns(2)
            with button_col1:
                send_button = st.form_submit_button("Send away")
    if send_button and user_input:
        st.session_state.phase = 2  # Update phase here
        st.session_state.user_input = user_input
        main_container.empty()
        return True
    return False


def processing_request():
    with st.status("Let's see what we can do...", expanded=True) as status:
        st.write("Reading through email...")
        if "Fahrzeug-Flag:" in st.session_state.user_input:
            st.session_state.temp_df_requests, st.session_state.temp_df_rim_details = context_retrieval_entry(
                text=st.session_state.user_input,
                instruction=instructions,
                debug=False,
                df_requests=st.session_state.df_requests,
                df_rim_details=st.session_state.df_rim_details,
                df_all_rims=st.session_state.df_all_rims,
                index=pinecone_index,
            )
            question = st.session_state.temp_df_requests.iloc[0]["question"]
            st.write("Ahh found the following details:")
            st.dataframe(st.session_state.temp_df_requests, use_container_width=True)
            st.dataframe(st.session_state.temp_df_rim_details, use_container_width=True)

            st.write("Now lets embed those this specific context...")
            df_dynamic_contexts = embed_context_entry(st.session_state.temp_df_rim_details)
            st.dataframe(df_dynamic_contexts, use_container_width=True)

            st.write("And find the right context based on the request...")
            specific_context = gather_contexts(
                question=question,
                matching_rows=st.session_state.temp_df_rim_details[(st.session_state.temp_df_rim_details["mapping_information"].notnull())],
                df_dynamic_contexts=df_dynamic_contexts,
                token_limit=3000,
            )
            st.dataframe(pd.DataFrame(specific_context), use_container_width=True)
            st.write("And of course the general context, regarding the request...")

            st.session_state.temp_df_requests = serialize_all_columns(st.session_state.temp_df_requests)
            st.session_state.temp_df_rim_details = serialize_all_columns(st.session_state.temp_df_rim_details)
        else:
            st.session_state.temp_df_requests = pd.DataFrame(columns=st.session_state.df_requests.columns)
            st.session_state.temp_df_rim_details = pd.DataFrame(columns=st.session_state.df_rim_details.columns)
            specific_context = ""
            question = st.session_state.user_input
            st.session_state.temp_df_requests.at[0, "question"] = question

        general_context, general_context_details = create_context_pinecone(
            question=question, index=pinecone_index, max_len=1500, size="ada"
        )
        st.dataframe(pd.DataFrame(general_context_details), use_container_width=True)
        time.sleep(0.5)
        status.update(label="request processed", state="complete", expanded=False)
        st.session_state.prompt = [
            {"role": "assistant", "content": f"[Vehicle Wheel Specific Context]: {specific_context}"},
            {"role": "assistant", "content": f"[General Context]: {general_context}"},
            {"role": "user", "content": f"[Question]: {question}"},
        ]

        st.session_state.temp_df_requests.at[0, "specific_context"] = specific_context
        st.session_state.temp_df_requests.at[0, "general_context"] = general_context_details

    return


def content_phase_2():
    """Render the content for Phase 2."""

    with st.expander("Input", expanded=False):
        st.write(f"{st.session_state.user_input}")
    with st.expander("Output", expanded=True):
        placeholder_response = st.empty()
        (
            st.session_state.chat_history,
            st.session_state.prompt_tokens,
            st.session_state.completion_tokens,
            st.session_state.total_tokens,
            st.session_state.response,
        ) = answer_question_streaming(
            model="gpt-4",
            instruction=instructions["gpt-response1"],
            prompt=st.session_state.prompt,
            debug=False,
            max_tokens=st.session_state.max_token_answer,
            temperature=st.session_state.temperature,
            callback=lambda text: display_stream_answer(text, placeholder_response),
        )

        if st.session_state.response:
            st.session_state.temp_df_requests.at[0, "final_response"] = st.session_state.response

            st.session_state.df_requests = pd.concat([st.session_state.df_requests, st.session_state.temp_df_requests], ignore_index=True)
            st.session_state.df_rim_details = pd.concat(
                [st.session_state.df_rim_details, st.session_state.temp_df_rim_details], ignore_index=True
            )

            #
            # Save to GCS
            save_to_gcs(
                bucket_name,
                st.session_state.df_file_paths["df_requests"],
                st.session_state.df_requests.to_json(orient="split"),
            )

            save_to_gcs(
                bucket_name,
                st.session_state.df_file_paths["df_rim_details"],
                st.session_state.df_rim_details.to_json(orient="split"),
            )
            return True
        return False


def display_stream_answer(r_text, placeholder_response):
    global updated_stream
    stream_text = ""
    updated_stream += r_text
    stream_text += f"\n{ updated_stream}\n"
    placeholder_response.markdown(stream_text, unsafe_allow_html=True)
    return updated_stream  # Return the updated stream to be used in feedback


def main():
    st.title("GPT-FO AI Assistant")
    tab1, tab2, tab3 = st.tabs(["Tab1", "Tab2", "Tab3"])

    with tab1:
        if "phase" not in st.session_state:
            st.session_state.phase = 1

        main_container = st.empty()
        sidebar_container = st.sidebar.empty()

        if st.session_state.phase == 1:
            with sidebar_container:
                sidebar_phase_1()
            should_move_to_phase_2 = content_phase_1(main_container)
            if should_move_to_phase_2:
                sidebar_container.empty()
                processing_request()

        if st.session_state.phase == 2:  # Use elif to prevent re-entry
            should_move_to_phase_3 = content_phase_2()
            with sidebar_container:
                sidebar_phase_2()
            if should_move_to_phase_3:
                st.session_state.phase = 3

        if st.session_state.phase == 3:
            display_feedback()

    with tab2:
        st.dataframe(st.session_state.df_requests, use_container_width=True)

    with tab3:
        st.dataframe(st.session_state.df_rim_details, use_container_width=True)


def display_feedback():
    with st.expander("Feedback", expanded=True):
        with st.form(key="feedback_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                feedback_quality = st.radio("Feedback Quality:", ["helpful", "not so helpful"], key="feedback_quality")
            with col2:
                feedback_tag = st.radio("Feedback Category:", ["full context", "missing context"], key="feedback_tag")

            helpful = True  # Default value
            comment = None

            if feedback_quality == "not so helpful":
                helpful = False

            comment = st.text_input("Please provide your comments here:", key="comment")
            ticket = st.text_input("Please provide your ticket Number or URL here:", key="helpwise")
            correct_response = st.text_area("Please provide the correct answer:", key="correct_response")

            if helpful:
                correct_response = st.session_state.response
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.update_dict = {
                "date": timestamp,
                "correct_response": correct_response,
                "comment": comment,
                "tag": feedback_tag,
                "helpful": helpful,
                "ticket": ticket,
                "tokens_used": [
                    st.session_state.prompt_tokens,
                    st.session_state.completion_tokens,
                    st.session_state.total_tokens,
                ],
                "model_parameters": [
                    st.session_state.selected_model,
                    st.session_state.max_token_question,
                    st.session_state.max_token_answer,
                    st.session_state.temperature,
                ],
            }

            send_feedback = st.form_submit_button("Send Feedback", on_click=update_feedback_info)

        if send_feedback:
            st.success("Feedback sent successfully! 2")


def update_feedback_info():
    last_index = st.session_state.df_requests.index.max()

    for column, value in st.session_state.update_dict.items():
        if isinstance(value, (list, dict)):
            value = json.dumps(value)
        st.session_state.df_requests.at[last_index, column] = value  # Moved this line inside the loop

    save_to_gcs(
        bucket_name,
        st.session_state.df_file_paths["df_requests"],
        st.session_state.df_requests.to_json(orient="split"),
    )

    st.success("Feedback sent successfully!")
    st.session_state.phase = 1


def sidebar_phase_1():
    """Render the sidebar for Phase 1."""
    st.sidebar.header("How it Works")
    st.sidebar.write(
        "Ich nutze GPT-4, um dir beim Kundensupport für Felgenout zu helfen. Ich kann allgemeine Fragen beantworten, aber für felgenspezifische Anfragen benötige ich das 'Fahrzeug-Flag' und Felgendetails."
    )

    # Add a separator
    st.sidebar.divider()

    # GPT parameter fields
    st.sidebar.subheader("GPT Parameters")
    # Default to GPT-3.5 if no model is selected

    st.session_state.selected_model = "gpt-4"
    st.sidebar.write(f"Selected 'model: {st.session_state.selected_model}")
    st.session_state.max_token_question = st.sidebar.number_input("Max tokens (question):", min_value=1, value=4000)
    st.session_state.max_token_answer = st.sidebar.number_input("Max tokens (answer):", min_value=1, value=1000)
    st.session_state.temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=0.3)


def sidebar_phase_2():
    """Render the sidebar for Phase 2."""
    st.sidebar.header("Token Usage")
    st.sidebar.write(f"Tokens used for prompt: {st.session_state.prompt_tokens}")
    st.sidebar.write(f"Tokens used for completion: {st.session_state.completion_tokens}")
    st.sidebar.write(f"Total tokens: {st.session_state.total_tokens}")
    if "total_chat_tokens" not in st.session_state:
        st.session_state.total_chat_tokens = 0
    st.session_state.total_chat_tokens += st.session_state.total_tokens
    st.sidebar.write(f"Total tokens in chat session: {st.session_state.total_chat_tokens}")


if __name__ == "__main__":
    main()
