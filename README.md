# HealthPredictor Plus API

HealthPredictor Plus is a FastAPI-based application that allows users to upload health reports, get an analysis of the reports, and ask health-related questions based on the context of the reports. The application uses OpenAI's GPT-3.5-turbo model to provide detailed responses and analyses.

## Features

- **Upload Health Report**: Upload a PDF health report and extract its text content.
- **Ask Health-Related Questions**: Ask questions based on the uploaded health report context.
- **Get Report Analysis**: Get an analysis of the uploaded health report and summarize the key health concerns.

## Installation

### Prerequisites

- Python 3.7+
- Redis server
- OpenAI API key

### Setup

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd healthpredictor-plus/backend/
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file in the project root directory and add your OpenAI API key:**

   ```env
   OPENAI_API_KEY=your_openai_api_key
   ```

5. **Start your Redis server:**

   Follow the instructions for your operating system to install and start Redis. Typically, you can start Redis with:

   ```bash
   redis-server
   ```

6. **Run the FastAPI application:**

   ```bash
   python app.py
   ```

   The application will be available at `http://127.0.0.1:5000`.

7. **Access Swagger:**

   The swagger is available at `http://127.0.0.1:5000/docs`.

## Endpoints

### Upload Health Report

- **URL**: `/upload-report/`
- **Method**: `POST`
- **Description**: Upload a PDF health report and extract its text content.
- **Parameters**:
  - `file`: The PDF file to be uploaded.
- **Response**:
  - `session_id`: The unique session ID for the uploaded report.

### Ask Health-Related Question

- **URL**: `/ask-question/`
- **Method**: `POST`
- **Description**: Ask a question based on the uploaded health report context.
- **Parameters**:
  - `query`: The question to be asked.
  - `session_id`: The session ID received after uploading the report.
- **Response**:
  - `answer`: The answer to the question.

### Get Report Analysis

- **URL**: `/report-analysis/`
- **Method**: `GET`
- **Description**: Get an analysis of the uploaded health report and summarize the key health concerns.
- **Parameters**:
  - `session_id`: The session ID received after uploading the report.
- **Response**:
  - `analysis`: The analysis of the report.

## Example Usage

### Upload Report

```bash
curl -X POST "http://127.0.0.1:5000/upload-report/" -F "file=@path_to_your_report.pdf"
```

### Ask Question

```bash
curl -X POST "http://127.0.0.1:5000/ask-question/" -H "Content-Type: application/json" -d '{"query": "What are the main health concerns?", "session_id": "your_session_id"}'
```

### Get Report Analysis

```bash
curl -X GET "http://127.0.0.1:5000/report-analysis/?session_id=your_session_id"
```

## Contributing

1. **Fork the repository.**
2. **Create a new branch:**

   ```bash
   git checkout -b your-feature-branch
   ```

3. **Make your changes.**
4. **Commit your changes:**

   ```bash
   git commit -m 'Add some feature'
   ```

5. **Push to the branch:**

   ```bash
   git push origin your-feature-branch
   ```

6. **Create a pull request.**
