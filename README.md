# Research Agent

## Setup

1.  **Create Virtual Environment**
    ```bash
    python -m venv .venv
    ```

2.  **Activate Virtual Environment**
    *   **macOS/Linux**:
        ```bash
        source .venv/bin/activate
        ```
    *   **Windows**:
        ```bash
        .venv\Scripts\activate
        ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Variables**
    Set your OpenAI API key:
    ```bash
    export OPENAI_API_KEY=your_key_here
    export SERPAPI_API_KEY=your_serpapi_key_here
    ```

    > **Note:** Web search functionality uses SerpAPI; you must supply a valid `SERPAPI_API_KEY`.

## Running the App

```bash
python app.py
```
