# Professional Car Price Predictor (Interview Edition)

This is a standalone, professional-grade version of the car price predictor, designed for technical interviews.

## Features
-   **Multi-Tab Interface**: Separate tabs for Prediction, Insights, and Prep.
-   **No Emojis**: Clean, corporate styling.
-   **Interview Cheat Sheet**: Built-in architecture diagrams and Q&A.

## How to Run Locally
1.  Navigate to this folder: `cd Pro_App`
2.  Install requirements: `pip install -r requirements.txt`
3.  Run the app: `streamlit run streamlit_app.py`

## How to Deploy (Render)
1.  Create a **New Web Service** on Render.
2.  Connect the SAME repository (`vikas872/Indian-Auto-Price-Predictor`).
3.  **Crucial Step**: In "Root Directory", enter: `Pro_App`
4.  Build Command: `pip install -r requirements.txt`
5.  Start Command: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
