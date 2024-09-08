import io
import json
import os
import PyPDF2
import uuid

from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI

from redis import Redis

from health_prediction_model.make_prediction import make_prediction
from schema import HealthRecord

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="HealthPredictor Plus API",
    description="An API for uploading health reports and asking health-related questions based on the report context.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up Redis client
redis_client = Redis(host='localhost', port=6379, db=0)

# Set up OpenAI key and model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

template = """You are a helpful Health Predictor assistant, an advanced AI designed to provide detailed health prediction. 
Answer the question based on the given context.

Context: {context}

Question: {question}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])
chain = LLMChain(llm=llm, prompt=prompt, verbose=True)


# Function to extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text


def answer_query(query, context):
    response = chain({"context": context[-15000:], "question": query})
    return response['text']


@app.post(
    "/upload-report/",
    summary="Upload a health report",
    description="Upload a PDF health report and extract its text content."
)
async def upload_report(file: UploadFile = File(...)):
    # Keep adding other variations of the pdf file type
    if file.content_type not in ["application/pdf", "application/octet-stream"]:
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF file.")

    # TODO: Validation for size.

    content = await file.read()
    file_like_object = io.BytesIO(content)
    text = extract_text_from_pdf(file_like_object)
    print("text", text)
    text = f"**User's provided Report:**\n{text}"

    # Generate a unique user session ID
    user_session_id = str(uuid.uuid4())
    # Store the context in Redis with the session ID
    redis_client.set(user_session_id, text, ex=3600)
    return {"message": "Report uploaded successfully", "session_id": user_session_id}


@app.post(
    "/ask-question/",
    summary="Ask a health-related question",
    description="Ask a question based on the uploaded health report context."
)
async def ask_question(query: str, session_id: str):
    """
    Ask a question based on the context of the uploaded health report.
    
    - **query**: The question you want to ask.
    - **session_id**: The session ID received after uploading the report.
    """

    # Retrieve the context from Redis using the session ID
    context = redis_client.get(session_id)
    if context is None:
        raise HTTPException(status_code=404, detail="Session ID not found or expired.")

    context = context.decode('utf-8')
    response = answer_query(query, context)

    # Update the context with the latest question and answer
    updated_context = f"{context}\nQ: {query}\nA: {response}"
    redis_client.set(session_id, updated_context, ex=3600)
    print("response", response)
    return JSONResponse(content={"answer": response})


@app.get(
    "/report-analysis/",
    summary="Get report analysis",
    description="Get an analysis of the uploaded health report and summarize the key health concerns."
)
async def report_analysis(session_id: str):
    """
    Get an analysis of the uploaded health report.
    
    - **session_id**: The session ID received after uploading the report.
    """
    # Retrieve the context from Redis using the session ID
    context = redis_client.get(session_id)
    if context is None:
        raise HTTPException(status_code=404, detail="Session ID not found or expired.")

    context = context.decode('utf-8')

    query = """
    Analyze the health report and provide a detailed summary of the key health concerns.

    Follow the below instructions to generate a comprehensive health report analysis:
    
    1. **Health Report Analysis:**
    - Thoroughly analyze the uploaded health report of the user.
    - If the uploaded report is not related to health, inform the user that the provided report is not related to health
     and request a valid health report.
    - Summarize the key health metrics and identify any abnormal values or areas of concern.
    - Highlight any diagnosed conditions or notable medical history.
    - Do not mention any user uploaded report kind of the content in the response, just provide the analysis of the 
    health report.

    2. **Predicted Future Ailments:**
    - Based on the health report, identify potential future health risks or ailments the user might face.
    - Explain the reasons for these predictions, referencing specific data points from the health report.

    3. **Exercise Routine Recommendations:**
    - Suggest a personalized exercise routine tailored to the user's current health status and needs.
    - Include specific types of exercises, recommended duration and frequency, and any precautions the user should take.

    4. **Diet Plan Recommendations:**
    - Provide a detailed diet plan aimed at improving the user's overall health and addressing any identified concerns.
    - Recommend specific foods to include and avoid, portion sizes, and meal frequency.
    - Consider any dietary restrictions or preferences mentioned in the health report.

    5. **Additional Preventive Measures:**
    - Suggest additional lifestyle changes or preventive measures that can help maintain or improve the user's health.
    - Include tips on stress management, sleep hygiene, and regular health monitoring.

    Ensure the language is simple and easy to understand, avoiding complex medical jargon. The recommendations should be
     practical, actionable, and personalized to the user's health status.
    """

    response = answer_query(query, context)

    # Update the context with the report analysis
    updated_context = f"{context}\nQ: {query}\nA: {response}"
    redis_client.set(session_id, updated_context, ex=3600)

    print("response", response)
    return JSONResponse(content={"analysis": response})


@app.post(
    "/health-record/",
    summary="Create a health record",
    description="Create a health record with detailed health information."
)
def create_health_record(record: HealthRecord):
    """
    Create a health record with detailed health information.
    :param record:
    :return:
    """
    # Todo : Add the health record validations for now everything is accepted

    context = record.json()
    print("context", context)

    prediction = None
    try:
        prediction = make_prediction(json.loads(context))
        print("prediction", prediction)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("error", e)

    prediction_line = f"""- Prediction obtained from the trained model is {prediction}.
    - Analyze the user's provided information to determine if the prediction made by the trained model is accurate, 
    if it's not accurate then just ignore it.""" if prediction is not None else ""

    title_line = f""" - Create a title that incorporates the prediction (e.g., "Your health seems to be {prediction}")
     only if it aligns with the analysis. If it does not align, use a general title such as 
     "Comprehensive Health Prediction Report for [full_name]".""" if (
            prediction is not None
    ) else 'Create a title such as "Comprehensive Health Prediction Report for [full_name]"'

    query = f"""
    Based on the user's provided information, generate a comprehensive health prediction report that includes:
    
    **Instructions to follow strictly:**
    {prediction_line}
    - Do not use the words like trained model, AI, or machine learning in the prediction.
    - If any information is missing or not provided by the user, then do not include it in the health report.

    ### Title Creation:
    {title_line}
    - You can improvise here as per the provided instructions.

    1. Key Health Metrics:
       - BMI
       - Blood Pressure
       - Heart Rate
       - Blood Sugar Levels
    
    2. Predicted Health Risks:
       *Note*: Include only the predictions that align with the user's health metrics and information and do not include
        the trained model's prediction.
       - List potential future health conditions with their probabilities.
       - Provide reasons for each predicted condition.
    
    3. Personalized Preventive Measures:
       - Dietary recommendations specific to the user's needs.
       - Exercise recommendations tailored to improve the user's health metrics.
       - Lifestyle and wellness tips for managing stress, improving sleep, and other relevant areas.
    
    4. Summary of Current Health Status:
       - Overall health assessment based on provided metrics.
       - Specific areas of improvement.
    
    5. Conclusion:
       - Emphasize the importance of consulting healthcare professionals.
       - Encourage regular health monitoring and proactive management.
    
    Make some of the key highlighting points bold in the report.
       
    Use clear, easy-to-understand language and avoid complex medical jargon. Make sure the report is factual 
    and actionable.
    Make the report interactive by providing the predictions, probability, predict potential future ailments and 
    provide preventive required measures to take care of their health.
    
    """
    response = answer_query(query, context)
    print(response)

    # Generate a unique user session ID
    user_session_id = str(uuid.uuid4())
    # Store the health record and response in Redis with the session ID
    updated_context = f"Health Record:\n{context}\nResponse:\n{response}"
    redis_client.set(user_session_id, updated_context, ex=3600)

    return {"message": "Health record created successfully", "predictions": response, "session_id": user_session_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
