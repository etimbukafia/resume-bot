from semantic_router import Route
from semantic_router import RouteLayer

def intents(encoders):
    resume_chat = Route(
        name="resume_inquiries",
        utterances=[
        "Can you find applicants with experience in data science?",
        "I'm looking for candidates with a strong background in machine learning.",
        "What resumes match the keyword 'Python developer'?",
        "Do you have any recommendations for candidates with over 5 years of experience?",
        "Can you show me resumes of applicants who worked as a software engineer?",
        "I'm interested in resumes with experience in computer vision. What do you have?",
        "Which applicants have project management skills?",
        "Can you find candidates who have worked in the finance sector?",
        "Show me resumes of applicants proficient in natural language processing.",
        "Do you have resumes with expertise in cloud computing?"
        "Tell me about Lucius's work experience"
        "What are skills listed on Kandace's resume?"
        ],
    )

    pdf_chat = Route(
        name = "request_resume_pdfs",
        utterances = [
            "Can you send me the PDF of the applicant's resume?",
            "I need the resume document for this candidate.",
            "Please provide the resume file for this applicant.",
            "Can I download the PDF version of the resume?",
            "I'd like to get a copy of the resume in PDF format.",
            "Can you show me the full resume PDF?",
            "I want the original resume document.",
            "Can you provide the resume as a PDF?",
            "Send me the PDF resume for this candidate."
        ]
    )

    # we place all of our decisions together into single list
    routes = [resume_chat, pdf_chat]
    rl = RouteLayer(encoder=encoders, routes=routes)

    return rl