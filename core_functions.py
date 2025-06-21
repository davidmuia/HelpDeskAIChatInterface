# core_functions.py
import os
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import hashlib
from streamlit_cookies_controller import CookieController
import time

load_dotenv()


# --- Database Configuration ---
DB_NAME = st.secrets["DB_NAME"]
DB_USER = st.secrets["DB_USER"]
DB_PASSWORD = st.secrets["DB_PASSWORD"]
DB_HOST = st.secrets["DB_HOST"]
DB_PORT = st.secrets["DB_PORT"]


# --- Gemini Configuration ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

EMBEDDING_MODEL_NAME = st.secrets["EMBEDDING_MODEL_NAME"]
GENERATIVE_MODEL = st.secrets["GENERATIVE_MODEL"]

# --- Initialize Models ---
embedding_model = genai.GenerativeModel(EMBEDDING_MODEL_NAME)  # Create embedding model
generative_model = genai.GenerativeModel(GENERATIVE_MODEL) #Create the text generation model


# --- Hashing Functions ---
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return True
    return False

# --- Database Functions ---
def create_connection():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    register_vector(conn)

    return conn

def create_table(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS pdf_chunks (
            id SERIAL PRIMARY KEY,
            pdf_name TEXT,
            chunk_id INTEGER,
            content TEXT,
            embedding VECTOR(768)
        );
    """)
    conn.commit()
    cur.close()


def insert_data(conn, data):
    cur = conn.cursor()
    insert_query = """
        INSERT INTO pdf_chunks (pdf_name, chunk_id, content, embedding)
        VALUES %s
    """
    execute_values(cur, insert_query, data)
    conn.commit()
    cur.close()

def search_similar_chunks(conn, query_embedding, top_k=5):
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT pdf_name, content FROM pdf_chunks
            ORDER BY embedding <#> %s::vector
            LIMIT %s
            """,
            (query_embedding, top_k)
        )  #  Explicitly cast to vector
        results = cur.fetchall()
        return results
    except Exception as e:
        print(f"Error in search_similar_chunks: {e}")
        return []
    finally:
        cur.close()

def get_all_pdf_names(conn):
    """Retrieves a list of all PDF names in the database."""
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT pdf_name FROM pdf_chunks")
    pdf_names = [row[0] for row in cur.fetchall()]
    cur.close()
    return pdf_names

def delete_pdf_chunks(conn, pdf_name):
    """Deletes all chunks associated with a specific PDF name."""
    cur = conn.cursor()
    cur.execute("DELETE FROM pdf_chunks WHERE pdf_name = %s", (pdf_name,))
    conn.commit()
    cur.close()

# --- New user table function ---
def create_user_table(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            first_name VARCHAR(255) NOT NULL,
            last_name VARCHAR(255) NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            department TEXT[],  -- Array of departments
            role VARCHAR(50) NOT NULL        );
    """)
    conn.commit()
    cur.close()

#New User data insert function
def create_user(conn, first_name, last_name, email, password, department, role):
    """Creates a new user in the database."""
    cur = conn.cursor()
    password_hash = make_hashes(password)
    try:
        cur.execute(
            """
            INSERT INTO users (first_name, last_name, email, password_hash, department, role)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (first_name, last_name, email, password_hash, department, role)
        )
        conn.commit()
        cur.close()
        return True # User creation success
    except psycopg2.errors.UniqueViolation:
        cur.close()
        return False  # Email already exists

def get_user_by_email(conn, email):
    """Retrieves a user from the database by email."""
    cur = conn.cursor()
    cur.execute("SELECT id, first_name, last_name, password_hash, department, role, email FROM users WHERE email = %s", (email,))
    user = cur.fetchone()
    cur.close()
    if user:
        return {
            "id": user[0],
            "first_name": user[1],
            "last_name": user[2],
            "password_hash": user[3],
            "department": user[4],
            "role": user[5],
            "email": user[6]
        }
    else:
        return None

#---Login cookies and persistence---
cookie_controller = CookieController()

def login(username, password):
    try:
        conn = create_connection()
        user = get_user_by_email(conn, username)
        conn.close()

        if user and check_hashes(password, user["password_hash"]) and user["role"] in ["admin", "trainer"]:
            u_email=user['email']
            u_role=user['role']
            # Set cookies for authentication and user information
            cookie_controller.set("email", u_email)
            st.session_state["email"] = u_email
            cookie_controller.set("role", u_role)
            st.session_state["role"] = u_role
            st.success(f"Logged in successfully")
            time.sleep(0.5)
            st.rerun()

        else:
            st.warning("Incorrect Username/Password or insufficient rights")
    except Exception as e:
        st.error(f"Error logging in: {e}")


def check_session():
    # Check if the email cookie exists
    c_email = cookie_controller.get("email")

    if c_email:
        # Restore session
        st.session_state["email"] = c_email


# Function to handle logout
def logout():
    # Clear cookie by setting it to an empty value with a past expiration
    cookie_controller.set("email", "", max_age=0)
    cookie_controller.set("role", "", max_age=0)
    #cookie_controller.remove("email")
    #cookie_controller.remove("role")
    st.session_state.pop("email", None)
    st.session_state.pop("role", None)
    st.success("Logged out successfully!")
    time.sleep(0.5)  # Pause briefly before rerun
    st.rerun()  # Rerun to clear the interface


# --- Embedding Functions ---
def generate_embedding(text):
    try:

        text = str(text)  # Ensure that text is a string

        task_type = "retrieval_query"

        # response = embedding_model.embed_content(content=text, task_type=task_type)
        response = genai.embed_content(model=EMBEDDING_MODEL_NAME, content=text, task_type=task_type)
        return response['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- PDF Processing Functions ---
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def chunk_text(text, chunk_size=300, chunk_overlap=30):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# --- Gemini Response Function ---
def generate_response(context, question, temperature=0.3):
    prompt = f"""
    You are a helpful assistant that is used to answer questions based on the documents from the database and provided context and
    also general IT queries and other topics not found in the documents. Your tone should be more conversational, personal and sound more human like. 
    Your Responses should be structured and in step-by-step format to help clarify why the response you gave is the most 
    appropriate one. The response should also be simple, rich in details, easy to understand and provide examples if necessary.
    Avoid using words such as According to the documentation or documents provided when responding to users. Do not infer to them that you 
    have been provided information to respond with so also avoid saying things like according to the information I have
    Always look at previous user questions to understand the context so you can easily answer follow up questions. 
    If a query is ambiguous, ask a clarifying question and base your response on the context  
    of their response and their previous questions and the responses you gave.
    If the provided context contains conflicting information, prioritize information from the document most relevant to the user's current query focus 
    (e.g., if the user asks about disciplinary matters, focus on disciplinary documents) Avoid mixing information from different topics if the context seems to contain multiple subjects, focus only on what's directly relevant to the user's question
    Emphasize Source Attribution (if needed): When answering, indicate which document or section the information came from.
    Based on the current conversation history you can come up with the most relevant follow up questions a user might need assistance on only. The questions can be one or two but at most 3 and have few wordings. Format them in italics
    For IT issues not sure of how to respond, advise the users to contact IT via 0746752351 or sending an email at ithelpdesk@pendahealth.com
    For Policy related issues where you are not sure of how to respond, advise the users to contact HR via hr@pendahealth.com

    Context:
    {context}

    Question: {question}
    Answer:
    """

    generation_config = genai.GenerationConfig(
        temperature=temperature,
        top_p=1.0,
        top_k=3,
        max_output_tokens=4096,
    )

    response=generative_model.generate_content(
        prompt,
        generation_config=generation_config
        )
    return response.text
