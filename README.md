# AI-examination-assistant
The purpose of this system is to enable examinees to collect all natural language materials related to a specific exam and, by constructing a Knowledge Graph or applying weighted segmentation techniques, generate questions and prompts in various forms. The system aims to create an AI capable of generating diverse questions at any time, with a level of question design and inquiry comparable to that of a real teacher.

This AI is expected to go beyond simple question types such as multiple-choice or fill-in-the-blank, and be capable of generating short-answer and even essay questions. Furthermore, the AI should be able to grade the responses and provide feedback on areas for improvement.

By building this AI-powered review system, I hope to recreate the intense questioning and extensive practice ("question sea") learning strategies from my student days and apply them to practical, purpose-driven exams.
or now, we are developing version 0.1 of this system using the ChatGPT API. We have adopted a web application architecture with a frontend-backend development approach.

On the backend, we use Flask to rapidly build the application logic.On the frontend, we use HTML/CSS/JavaScript to create an interactive user interface.The overall system architecture is divided into three modules: 
1. Material Extraction Module: Uses OCR or text processing techniques to extract learning materials. 、
2. Question Generation Module: Leverages the GPT API to generate exam questions.
3. Grading Module: Evaluates the user's responses.

The required versions for third-party libraries are:
Flask: 3.0.2
OpenAI Python API: 1.35.13
