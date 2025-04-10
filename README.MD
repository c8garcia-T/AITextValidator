Limitations: 1,000 Calls Per Month
20 Calls Per Minute. 
# 🧠 Vocabulary Definition Checker

This Streamlit application helps educators verify whether vocabulary words align with simplified, student-friendly definitions. Powered by Cohere's AI, it quickly flags mismatches and suggests improvements, making it a valuable tool for English language instructors.

## Features

- ✅ Upload an `.xlsx` file containing vocabulary words and student definitions.
- 🤖 AI verifies if the word matches the definition using a custom system prompt.
- ❌ Incorrect matches are flagged with suggestions for improvement.
- 📊 View all results in a clean table format.
- 🔍 Filter to view only errors for quick review.

## How It Works

1. Upload an Excel file with two columns:
   - **Column A**: Vocabulary word  
   - **Column B**: Student-friendly definition  
2. The app sends each row to Cohere's API using a structured prompt.
3. AI responds with either:
   - `correct`
   - `error` + suggestion (better word or definition)
4. Results are displayed interactively using Streamlit.
## Example

**Input Excel format:**

| Word           | Student Definition                |
|----------------|-----------------------------------|
| Predict        | To say what will happen next      |
| Photosynthesis | A small bug that eats plants      |

**The app returns:**

- ✅ `"Predict"`: correct  
- ❌ `"Photosynthesis"`: error – Suggests simpler definition or alternate word
