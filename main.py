import pandas as pd
import cohere
import streamlit as st

COHERE_KEY = st.secrets["COHERE_API_KEY"]
systemMessage = """
    You are an English education expert.
    You help teachers verify if vocabulary words match simplified student-level definitions.
    Return only 'correct' or 'error'. If it is 'error', also return a better definition or a more appropriate word.
"""

co = cohere.Client(COHERE_KEY)

uploaded_file = st.file_uploader(
    label="Upload File",
    accept_multiple_files=False,
    label_visibility="visible",
    type=["xlsx"],
)
localTest = st.secrets["ISLOCALDEV"] == "True"

if uploaded_file:
    st.title("📘 Results")

    rawData = pd.read_excel(uploaded_file, sheet_name=0, header=None, dtype=str)

    output = []
    for rowIndex, row in enumerate(rawData.itertuples(index=False)):
        word = row[0]
        definition = row[1]
        prompt = f"""
            Word: {word}
            Student Definition: "{definition}"
            Does the word match the student-friendly definition? Explain only if there's an error.
        """
        if not localTest:
            response = co.chat(
                model="command-a-03-2025",
                temperature=0,
                chat_history=[{"role": "SYSTEM", "message": systemMessage}],
                message=prompt,
            )
            cleanedResponse = response.text.strip()
        else:
            cleanedResponse = "THIS IS A TEST"
        output.append([word, definition, cleanedResponse])
        if cleanedResponse.lower() == "correct":
            st.success(
                f"**Word {rowIndex+1}:** {word}  \n**Response:** {cleanedResponse}"
            )
        else:
            st.error(
                f"**Word {rowIndex+1}:** {word}  \n**Response:** {cleanedResponse} \n**Original Definition:**{definition}"
            )
    df = pd.DataFrame.from_records(
        output,
        columns=["Word", "Original Definition", "AI Response"],
        index=[i + 1 for i in range(0, len(rawData))],
    )

    st.title("Table View")
    st.dataframe(df)
    dfErrors = df[df["AI Response"] != "correct"]
    st.title(f"Errors Only ({dfErrors.shape[0]})")
    for rowError in dfErrors.itertuples(index=True):
        st.error(
            f"**Word {rowError[0]}:** {rowError[1]}  \n**Response:** {rowError[3]}  \n\n**Original Definition:** {rowError[2]}"
        )
