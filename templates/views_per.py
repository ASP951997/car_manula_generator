import os
import assemblyai as aai
import PyPDF2
import docx
import re
import json
import threading
import csv
from gtts import gTTS
from pathlib import Path
from dotenv import load_dotenv
import pytz
from textblob import TextBlob
from collections import Counter
import traceback
import readtime
import time
import numpy as np
import cv2
import base64
import urllib
import pytesseract
import PIL.Image
from django.utils import timezone
from django.core.files.base import ContentFile
from deepface import DeepFace
from datetime import datetime, timedelta
from .utils  import call_perplexity_api

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.template import loader
from django.core.files.storage import default_storage
from django.conf import settings
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.template.loader import render_to_string
from django.utils.safestring import mark_safe
from django.core.mail import send_mail
from weasyprint import HTML

from .camera import VideoCamera
from .models import InterviewSession, WarningLog, InterviewQuestion
from .yolo_face_detector import detect_face_with_yolo

load_dotenv()
aai.api_key = os.getenv("ASSEMBLYAI_API_KEY")

try:
    assembly_client = aai.Client(aai.api_key)
    print("AssemblyAI client configured.")
except Exception as e:
    print(f"Error configuring AssemblyAI client: {e}"); assembly_client = None

FILLER_WORDS = ['um', 'uh', 'er', 'ah', 'like', 'okay', 'right', 'so', 'you know', 'i mean', 'basically', 'actually', 'literally']
CAMERAS, camera_lock = {}, threading.Lock()

THINKING_TIME, ANSWERING_TIME, REVIEW_TIME = 20, 60, 10

def get_camera_for_session(session_key):
    with camera_lock:
        if session_key in CAMERAS: return CAMERAS[session_key]
        try:
            session_obj = InterviewSession.objects.get(session_key=session_key)
            camera_instance = VideoCamera(session_id=session_obj.id)
            CAMERAS[session_key] = camera_instance
            return camera_instance
        except InterviewSession.DoesNotExist:
            print(f"Could not find session for session_key {session_key} to create camera.")
            return None
        except Exception as e:
            print(f"Error creating camera instance: {e}")
            return None

def release_camera_for_session(session_key):
    with camera_lock:
        if session_key in CAMERAS:
            print(f"--- Releasing camera for session {session_key} ---")
            CAMERAS[session_key].cleanup()
            del CAMERAS[session_key]

SUPPORTED_LANGUAGES = {'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 'hi': 'Hindi'}

def get_text_from_file(uploaded_file):
    name, extension = os.path.splitext(uploaded_file.name)
    text = ""
    if extension == '.pdf':
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages: text += page.extract_text() or ""
    elif extension == '.docx':
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs: text += para.text + "\n"
    else: text = uploaded_file.read().decode('utf-8', errors='ignore')
    return text

@login_required
def create_interview_invite(request):
    if request.method == 'POST':
        candidate_name = request.POST.get('candidate_name')
        candidate_email = request.POST.get('candidate_email')
        jd_text = request.POST.get('jd')
        resume_file = request.FILES.get('resume')
        language_code = request.POST.get('language', 'en')
        accent_tld = request.POST.get('accent', 'com')
        scheduled_at_str = request.POST.get('scheduled_at')

        if not all([candidate_name, candidate_email, jd_text, resume_file, scheduled_at_str]):
             return render(request, 'interview_app/create_invite.html', {'error': 'All fields are required.', 'languages': SUPPORTED_LANGUAGES})

        try:
            ist = pytz.timezone('Asia/Kolkata')
            naive_datetime = datetime.strptime(scheduled_at_str, '%Y-%m-%dT%H:%M')
            aware_datetime = ist.localize(naive_datetime)
        except (ValueError, pytz.exceptions.InvalidTimeError):
            return render(request, 'interview_app/create_invite.html', {'error': 'Invalid date and time format provided.', 'languages': SUPPORTED_LANGUAGES})

        resume_text = get_text_from_file(resume_file)
        if not resume_text:
            return render(request, 'interview_app/create_invite.html', {'error': 'Could not read the resume file.', 'languages': SUPPORTED_LANGUAGES})

        session = InterviewSession.objects.create(
            candidate_name=candidate_name, candidate_email=candidate_email,
            job_description=jd_text, resume_text=resume_text,
            language_code=language_code, accent_tld=accent_tld,
            scheduled_at=aware_datetime
        )

        interview_url = request.build_absolute_uri(f"/?session_key={session.session_key}")

        try:
            scheduled_time_str = aware_datetime.strftime('%A, %B %d, %Y at %I:%M %p %Z')
            
            email_subject = "Your AI Interview Invitation"
            email_body = (
                f"Dear {candidate_name},\n\n"
                f"Your AI screening interview has been scheduled for: {scheduled_time_str}.\n\n"
                "Please use the following unique link to begin your interview at the scheduled time. "
                "The link will become active at the start time and will expire 10 minutes after.\n"
                f"{interview_url}\n\n"
                "Best of luck!\n"
            )
            
            send_mail(
                email_subject,
                email_body,
                os.getenv('EMAIL_HOST_USER'),
                [candidate_email],
                fail_silently=False,
            )
            print(f"--- Invitation sent to {candidate_email} via Gmail SMTP ---")
        except Exception as e:
            print(f"ERROR sending email: {e}")
            return render(request, 'interview_app/create_invite.html', {'error': f'Could not send email. Please check your .env settings and ensure you are using a Google App Password. Error: {e}', 'languages': SUPPORTED_LANGUAGES})

        return redirect('dashboard')

    return render(request, 'interview_app/create_invite.html', {'languages': SUPPORTED_LANGUAGES})

def interview_portal(request):
    session_key = request.GET.get('session_key')
    if not session_key:
        return render(request, 'interview_app/invalid_link.html')
    
    session = get_object_or_404(InterviewSession, session_key=session_key)

    # This is the main validation logic block
    if session.status != 'SCHEDULED':
        return render(request, 'interview_app/invalid_link.html', {'error': 'This interview has already been completed or has expired.'})

    if session.scheduled_at:
        now = timezone.now()
        start_time = session.scheduled_at
        grace_period = timedelta(minutes=10)
        expiry_time = start_time + grace_period

        # Case 1: The user is too early.
        if now < start_time:
            start_time_local = start_time.astimezone(pytz.timezone('Asia/Kolkata'))
            # We pass all necessary context for the countdown timer here.
            return render(request, 'interview_app/invalid_link.html', {
                'page_title': 'Interview Not Started',
                'error': f"Your interview has not started yet. Please use the link at the scheduled time:",
                'scheduled_time_str': start_time_local.strftime('%Y-%m-%d %I:%M %p IST'),
                'start_time_iso': start_time.isoformat() # This is crucial for the JS countdown
            })

        # Case 2: The user is too late.
        if now > expiry_time:
            session.status = 'EXPIRED'
            session.save()
            return render(request, 'interview_app/invalid_link.html', {
                'page_title': 'Interview Link Expired',
                'error': 'This interview link has expired because the 10-minute grace period after the scheduled time has passed.'
            })
    else:
        # Case 3: The session has no scheduled time (should not happen in normal flow).
         return render(request, 'interview_app/invalid_link.html', {'error': 'This interview session is invalid as it does not have a scheduled time.'})

    # If the user is within the valid time window, proceed with the interview setup.
    try:
        if session.questions.exists():
            all_questions = [
                {'type': q.question_type, 'text': q.question_text, 'audio_url': q.audio_url}
                for q in session.questions.filter(question_level='MAIN').order_by('order')
            ]
        else:
            DEV_MODE = True
            all_questions = []

            if DEV_MODE:
                print("--- RUNNING IN DEV MODE: Using hardcoded questions and summary. ---")
                session.resume_summary = "This is a sample resume summary for developer mode. The candidate seems proficient in Python and Django."
                all_questions = [
                    {'type': 'Ice-Breaker', 'text': 'Welcome! To start, can you tell me about a challenging project you have worked on?'},
                    {'type': 'Technical Questions', 'text': 'What is the difference between `let`, `const`, and `var` in JavaScript?'},
                    {'type': 'Behavioral Questions', 'text': 'Describe a time you had a conflict with a coworker and how you resolved it.'}
                ]
            else:
                print("--- RUNNING IN PRODUCTION MODE: Calling Perplexity AI API. ---")
                
                
                summary_prompt = f"Summarize key skills from the following resume:\n\n{session.resume_text}"
                summary_response_text = call_perplexity_api(summary_prompt, model="llama-3.1-instruct")
                session.resume_summary = summary_response_text
                
                language_name = SUPPORTED_LANGUAGES.get(session.language_code, 'English')
                master_prompt = (
                    f"You are an expert AI interviewer. Your task is to generate 5 insightful interview questions in {language_name}. "
                    f"The interview is for a '{session.job_description.splitlines()[0]}' role. "
                    "Please base the questions on the provided job description and candidate's resume. "
                    "Start with a welcoming ice-breaker question that also references something specific from the candidate's resume. "
                    "Then, generate a mix of technical and behavioral questions. "
                    "You MUST format the output as Markdown. "
                    "You MUST include '## Technical Questions' and '## Behavioral Questions' headers. "
                    "Each question MUST start with a hyphen '-'. "
                    "Do NOT add any introductions, greetings (beyond the first ice-breaker question), or concluding remarks. "
                    f"\n\n--- JOB DESCRIPTION ---\n{session.job_description}\n\n--- RESUME ---\n{session.resume_text}"
                )
                response_text = call_perplexity_api(master_prompt, model="llama-3.1-instruct")          
                sections = re.findall(r"##\s*(.*?)\s*\n(.*?)(?=\n##|\Z)", response_text, re.DOTALL)
                if not sections: raise ValueError("Could not parse ## headers from AI response.")
                
                for category_name, question_block in sections:
                    lines = question_block.strip().split('\n')
                    for line in lines:
                        if line.strip().startswith('-'):
                            all_questions.append({'type': category_name.strip(), 'text': line.strip().lstrip('- ').strip()})
            
            if not all_questions: raise ValueError("No questions were generated or parsed.")

            if all_questions and "welcome" in all_questions[0]['text'].lower():
                all_questions[0]['type'] = 'Ice-Breaker'
            
            session.save()
            
            tts_dir = os.path.join(settings.MEDIA_ROOT, 'tts'); os.makedirs(tts_dir, exist_ok=True)
            for i, q_data in enumerate(all_questions):
                tts = gTTS(text=q_data['text'], lang=session.language_code, tld=session.accent_tld if session.language_code == 'en' else 'com')
                tts_path = os.path.join(tts_dir, f'q_{i}_{session.session_key}.mp3')
                tts.save(tts_path)
                audio_url = os.path.join(settings.MEDIA_URL, 'tts', os.path.basename(tts_path))
                q_data['audio_url'] = audio_url
                InterviewQuestion.objects.create(
                    session=session, 
                    question_text=q_data['text'], 
                    question_type=q_data['type'], 
                    order=i,
                    question_level='MAIN'
                )
        
        context = { 'session_key': session_key, 'interview_session_id': str(session.id), 'questions_data': all_questions, 'interview_started': True }
        return render(request, 'interview_app/portal.html', context)

    except Exception as e:
        print(f"ERROR during interview setup: {e}"); traceback.print_exc()
        return HttpResponse(f"An API or processing error occurred: {str(e)}", status=500)

@login_required
def dashboard(request):
    sessions = InterviewSession.objects.all().order_by('-created_at')
    context = {'sessions': sessions}
    template = loader.get_template('interview_app/dashboard.html')
    return HttpResponse(template.render(context, request))


@login_required
def interview_report(request, session_id):
    try:
        session = InterviewSession.objects.get(id=session_id)
        interview_data = list(session.questions.all())
        all_logs_list = list(session.logs.all())
        warning_counts = Counter([log.warning_type.replace('_', ' ').title() for log in all_logs_list if log.warning_type != 'excessive_movement'])

        if session.language_code == 'en' and not session.is_evaluated and interview_data:
            print(f"--- Performing all first-time AI evaluations for session {session.id} with Perplexity AI ---")    
            try:
                print("--- Evaluating Resume vs. Job Description ---")
                resume_eval_prompt = (
                    "You are an expert technical recruiter. Analyze the following resume against the provided job description. "
                    "Provide a score from 0.0 to 10.0 indicating how well the candidate's experience aligns with the job requirements. "
                    "Also provide a brief analysis. Format your response EXACTLY as follows:\n\n"
                    "SCORE: [Your score, e.g., 8.2]\n"
                    "ANALYSIS: [Your one-paragraph analysis here.]"
                    f"\n\nJOB DESCRIPTION:\n{session.job_description}\n\nRESUME:\n{session.resume_text}"
                )
                resume_response_text = call_perplexity_api(resume_eval_prompt, model="llama-3.1-instruct")
                score_match = re.search(r"SCORE:\s*([\d\.]+)", resume_response_text)
                if score_match: session.resume_score = float(score_match.group(1))
                session.resume_feedback = resume_response_text
            except Exception as e:
                print(f"ERROR during Resume evaluation: {e}"); session.resume_feedback = "An error occurred during resume evaluation."

            try:
                print("--- Evaluating Interview Answers ---")
                qa_text = "".join([f"Question: {item.question_text}\nAnswer: {item.transcribed_answer or 'No answer.'}\n\n" for item in interview_data])
                answers_eval_prompt = (
                    "You are an expert interviewer. Evaluate the candidate's answers to the following questions. "
                    "Provide an overall score from 0.0 to 10.0 for their performance. "
                    "Also provide a brief summary of their strengths and areas for improvement. "
                    "Format your response EXACTLY as follows:\n\n"
                    "SCORE: [Your score, e.g., 6.8]\n"
                    "FEEDBACK: [Your detailed feedback here.]"
                    f"\n\nQUESTIONS & ANSWERS:\n{qa_text}"
                )
                answers_response = call_perplexity_api(answers_eval_prompt, model="llama-3.1-instruct")
                answers_response_text = answers_response.text
                score_match = re.search(r"SCORE:\s*([\d\.]+)", answers_response_text)
                if score_match: session.answers_score = float(score_match.group(1))
                session.answers_feedback = answers_response_text
            except Exception as e:
                print(f"ERROR during Answers evaluation: {e}"); session.answers_feedback = "An error occurred during answers evaluation."

            try:
                print("--- Evaluating Keyword Match ---")
                qa_text = "".join([f"Q: {item.question_text}\nA: {item.transcribed_answer or 'No answer.'}\n\n" for item in interview_data])
                keyword_prompt = (
                    "You are a talent acquisition analyst. Your task is to analyze the candidate's interview answers against the provided job description. "
                    "Identify the top 5-7 most important technical skills or keywords from the job description. "
                    "Then, for each keyword, determine if the candidate successfully demonstrated that skill in their answers. "
                    "Provide specific quotes from their answers as evidence. Format your response EXACTLY as follows:\n\n"
                    "**Keyword:** [Keyword 1]\n"
                    "**Demonstrated:** [Yes/No/Partially]\n"
                    "**Evidence:** \"[Quote from candidate's answer]\"\n\n"
                    "**Keyword:** [Keyword 2]\n"
                    "**Demonstrated:** [Yes/No/Partially]\n"
                    "**Evidence:** \"[Quote from candidate's answer]\""
                    f"\n\nJOB DESCRIPTION:\n{session.job_description}\n\nINTERVIEW TRANSCRIPT:\n{qa_text}"
                )
                keyword_response = call_perplexity_api(keyword_prompt, model="llama-3.1-instruct")
                session.keyword_analysis = keyword_response.text
            except Exception as e:
                print(f"ERROR during Keyword analysis: {e}"); session.keyword_analysis = "An error occurred during keyword analysis."
            
            try:
                print("--- Generating Behavioral Pattern Analysis ---")
                events = []
                start_time = session.created_at
                for i, item in enumerate(interview_data):
                    question_time_seconds = i * (THINKING_TIME + ANSWERING_TIME + REVIEW_TIME)
                    events.append({'time': question_time_seconds, 'event': f"Started Question {i+1}: '{item.question_text[:50]}...'"})
                for log in all_logs_list:
                    if log.warning_type != 'excessive_movement':
                        event_time_seconds = (log.timestamp - start_time).total_seconds()
                        events.append({'time': event_time_seconds, 'event': f"WARNING: {log.warning_type.replace('_', ' ').title()}"})
                events.sort(key=lambda x: x['time'])
                event_timeline_text = "\n".join([f"At {int(e['time'])}s: {e['event']}" for e in events])
                correlation_prompt = (
                    "You are an expert behavioral analyst and proctor. You will be given a chronological timeline of events from an automated interview, including when questions were asked and when proctoring warnings were triggered. "
                    "Your task is to identify any concerning correlations or patterns. Look for instances where specific types of questions might have caused warnings like 'Low Concentration' or 'Tab Switched'. "
                    "Provide a brief, bulleted list of any notable patterns you find. If no significant patterns are found, simply state that. "
                    "Keep your analysis concise and focused on actionable insights for a recruiter.\n\n"
                    f"Here is the event timeline:\n\n{event_timeline_text}"
                )
                correlation_response = call_perplexity_api(correlation_prompt, model="llama-3.1-instruct")
                session.behavioral_analysis = correlation_response.text
            except Exception as e:
                print(f"ERROR during Behavioral Pattern Analysis: {e}"); session.behavioral_analysis = "An error occurred during behavioral analysis."

            try:
                print("--- Generating Overall Candidate Profile ---")
                warning_summary = ", ".join([f"{count}x {name}" for name, count in warning_counts.items()]) or "None"
                overall_prompt = (
                    "You are a senior hiring manager. You have been provided with a holistic view of a candidate's interview, "
                    "including their resume fit, interview answer performance, and proctoring warnings. "
                    "Your task is to synthesize all this information into a final recommendation. "
                    "Provide a final 'Overall Score' from 0.0 to 10.0 and a concluding 'Hiring Recommendation' paragraph.\n\n"
                    "DATA PROVIDED:\n"
                    f"- Resume vs. Job Description Score: {session.resume_score or 'N/A'}/10\n"
                    f"- Interview Answers Score: {session.answers_score or 'N/A'}/10\n"
                    f"- Proctoring Warnings: {warning_summary}\n\n"
                    "Format your response EXACTLY as follows:\n\n"
                    "OVERALL SCORE: [Your final blended score, e.g., 7.8]\n"
                    "HIRING RECOMMENDATION: [Your final concluding paragraph on whether to proceed with the candidate and why.]"
                )
                overall_response = call_perplexity_api(overall_prompt, model="llama-3.1-instruct")

                overall_response_text = overall_response.text
                score_match = re.search(r"OVERALL SCORE:\s*([\d\.]+)", overall_response_text)
                if score_match: session.overall_performance_score = float(score_match.group(1))
                session.overall_performance_feedback = overall_response_text
            except Exception as e:
                print(f"ERROR during Overall evaluation: {e}"); session.overall_performance_feedback = "An error occurred."
            
            session.is_evaluated = True
            session.save()
        
        total_filler_words = 0
        avg_wpm = 0
        wpm_count = 0
        sentiment_scores = []
        avg_response_time = 0
        response_time_count = 0

        if session.language_code == 'en':
            for item in interview_data:
                if item.transcribed_answer:
                    word_count = len(item.transcribed_answer.split())
                    read_time_result = readtime.of_text(item.transcribed_answer)
                    read_time_minutes = read_time_result.minutes + (read_time_result.seconds / 60)
                    if read_time_minutes > 0:
                        item.words_per_minute = round(word_count / read_time_minutes)
                        avg_wpm += item.words_per_minute
                        wpm_count += 1
                    else:
                        item.words_per_minute = 0
                    if item.response_time_seconds:
                        avg_response_time += item.response_time_seconds
                        response_time_count += 1
                    lower_answer = item.transcribed_answer.lower()
                    item.filler_word_count = sum(lower_answer.count(word) for word in FILLER_WORDS)
                    total_filler_words += item.filler_word_count
                    sentiment_scores.append({'question': f"Q{item.order + 1}", 'score': TextBlob(item.transcribed_answer).sentiment.polarity})
                else:
                    sentiment_scores.append({'question': f"Q{item.order + 1}", 'score': 0.0})
        
        final_avg_wpm = round(avg_wpm / wpm_count) if wpm_count > 0 else 0
        final_avg_response_time = round(avg_response_time / response_time_count, 2) if response_time_count > 0 else 0

        warning_timeline_data = []
        if all_logs_list:
            start_time = session.created_at
            for log in all_logs_list:
                 if log.warning_type != 'excessive_movement':
                     time_delta_seconds = (log.timestamp - start_time).total_seconds()
                     warning_timeline_data.append({'x': time_delta_seconds, 'y': log.warning_type.replace('_', ' ').title()})
        
        keyword_chart_data = []
        if session.is_evaluated and session.keyword_analysis:
            matches = re.findall(r"\*\*Keyword:\*\*\s*(.*?)\n\*\*Demonstrated:\*\*\s*(.*?)\n", session.keyword_analysis, re.DOTALL)
            for match in matches:
                keyword, status = match
                score = 0
                if 'Yes' in status: score = 10
                elif 'Partially' in status: score = 5
                keyword_chart_data.append({'skill': keyword.strip(), 'score': score})

        analytics_data = {
            'warning_counts': dict(warning_counts),
            'sentiment_scores': sentiment_scores,
            'evaluation_scores': {'Resume vs JD': session.resume_score or 0, 'Interview Answers': session.answers_score or 0},
            'warning_timeline': warning_timeline_data,
            'communication_radar': {
                'Pace (WPM)': final_avg_wpm,
                'Clarity (Few Fillers)': total_filler_words,
                'Responsiveness (sec)': final_avg_response_time
            },
            'keyword_chart': keyword_chart_data
        }
        
        main_questions_with_followups = session.questions.filter(question_level='MAIN').prefetch_related('follow_ups').order_by('order')

        context = {
            'session': session, 
            'main_questions_with_followups': main_questions_with_followups,
            'interview_data': interview_data, 
            'analytics_data': analytics_data,
            'total_filler_words': total_filler_words,
            'avg_wpm': final_avg_wpm,
            'behavioral_analysis_html': mark_safe((session.behavioral_analysis or "").replace('\n', '<br>')),
            'keyword_analysis_html': mark_safe((session.keyword_analysis or "").replace('\n', '<br>').replace('**', '<strong>').replace('**', '</strong>'))
        }

        if session.session_key: release_camera_for_session(session.session_key)
        template = loader.get_template('interview_app/report.html')
        return HttpResponse(template.render(context, request))
    except InterviewSession.DoesNotExist:
        return HttpResponse("Interview session not found.", status=404)

@login_required
def download_report_pdf(request, session_id):
    try:
        session = InterviewSession.objects.get(id=session_id)
        all_logs_list = list(session.logs.all())
        warning_counts = Counter([log.warning_type.replace('_', ' ').title() for log in all_logs_list if log.warning_type != 'excessive_movement'])
        chart_config = { 'type': 'doughnut', 'data': { 'labels': list(warning_counts.keys()), 'datasets': [{'data': list(warning_counts.values())}]}}
        chart_url = f"https://quickchart.io/chart?c={urllib.parse.quote(json.dumps(chart_config))}"
        context = { 'session': session, 'interview_data': session.questions.all(), 'warning_counts': dict(warning_counts), 'chart_url': chart_url }
        html_string = render_to_string('interview_app/report_pdf.html', context)
        pdf = HTML(string=html_string).write_pdf()
        response = HttpResponse(pdf, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="interview_report_{session.id}.pdf"'
        return response
    except InterviewSession.DoesNotExist:
        return HttpResponse("Interview session not found.", status=404)

@csrf_exempt
@require_POST
def end_interview_session(request):
    try:
        data = json.loads(request.body)
        session_key = data.get('session_key')
        if not session_key:
            return JsonResponse({"status": "error", "message": "Session key required."}, status=400)
        
        try:
            session = InterviewSession.objects.get(session_key=session_key)
            session.status = 'COMPLETED'
            session.save()
            print(f"--- Session {session_key} marked as COMPLETED. ---")
        except InterviewSession.DoesNotExist:
            print(f"Warning: Could not find session {session_key} to mark as completed.")
            
        release_camera_for_session(session_key)
        return JsonResponse({"status": "ok"})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

def interview_complete(request):
    template = loader.get_template('interview_app/interview_complete.html')
    return HttpResponse(template.render({}, request))

def generate_and_save_follow_up(session, parent_question, transcribed_answer):
    language_name = SUPPORTED_LANGUAGES.get(session.language_code, 'English')
    
    prompt = (
        f"You are an expert interviewer conducting an interview in {language_name}. "
        f"The candidate was asked the following question:\n'{parent_question.question_text}'\n\n"
        f"The candidate gave this transcribed answer:\n'{transcribed_answer}'\n\n"
        "Analyze the answer. If the answer is weak, evasive, or the candidate explicitly states they don't know the full answer "
        "(e.g., 'I am not sure', 'I don't have experience with that, but...', 'I know the basics'), "
        "then generate ONE single, simpler, follow-up question on the same core topic to test their foundational knowledge. "
        "The follow-up should be a natural continuation of the conversation. "
        "If the original answer is strong, confident, and complete, you MUST respond with the exact text: NO_FOLLOW_UP\n"
        "Do NOT add any other text, prefixes, or formatting to your response. Just the question text itself or NO_FOLLOW_UP."
    )

    try:
        follow_up_text = call_perplexity_api(prompt, model="llama-3.1-instruct").strip()
        if follow_up_text and "NO_FOLLOW_UP" not in follow_up_text and len(follow_up_text) > 10:
            print(f"--- Generated Follow-up Question: {follow_up_text} ---")
            
            tts = gTTS(text=follow_up_text, lang=session.language_code, tld=session.accent_tld if session.language_code == 'en' else 'com')
            tts_dir = os.path.join(settings.MEDIA_ROOT, 'tts'); os.makedirs(tts_dir, exist_ok=True)
            tts_filename = f'followup_{parent_question.id}_{int(time.time())}.mp3'
            tts_path = os.path.join(tts_dir, tts_filename)
            tts.save(tts_path)
            audio_url = os.path.join(settings.MEDIA_URL, 'tts', tts_filename)

            follow_up_question = InterviewQuestion.objects.create(
                session=session,
                question_text=follow_up_text,
                question_type=parent_question.question_type,
                question_level='FOLLOW_UP',
                parent_question=parent_question,
                order=parent_question.order,
                audio_url=audio_url
            )
            
            return {'text': follow_up_question.question_text, 'type': follow_up_question.question_type, 'audio_url': audio_url}
    except Exception as e:
        print(f"ERROR generating follow-up question: {e}")
    return None

@csrf_exempt
def transcribe_audio(request):
    if not assembly_client:
        return JsonResponse({'error': 'AssemblyAI client not available.'}, status=500)
    
    if request.method == 'POST' and request.FILES.get('audio_data'):
        audio_file = request.FILES['audio_data']
        session_id = request.POST.get('session_id')
        question_index = request.POST.get('question_index')
        response_time = request.POST.get('response_time')

        # We don't need to save the file locally for AssemblyAI, we can upload directly
        # For simplicity, we'll save it to a temporary path, but a more robust solution
        # would stream the file directly to AssemblyAI's API.
        file_path = default_storage.save('temp_audio.webm', audio_file)
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)

        try:
            # Use AssemblyAI to transcribe the audio file
            # You can specify various transcription options here
            # For a real-time solution, you would use the AssemblyAI Real-Time API
            # but for this file-based transcription, we'll use the ASR API
            with open(full_path, "rb") as f:
                transcriber = aai.Transcriber()
                transcript = transcriber.transcribe(f)
            
            if transcript.status == aai.TranscriptStatus.completed:
                transcribed_text = transcript.text
            else:
                raise Exception(f"Transcription failed with status: {transcript.status}")

            follow_up_data = None
            if session_id and question_index is not None:
                try:
                    question_to_update = InterviewQuestion.objects.get(session_id=session_id, order=int(question_index), question_level='MAIN')
                    question_to_update.transcribed_answer = transcribed_text
                    if response_time:
                        question_to_update.response_time_seconds = float(response_time)
                    question_to_update.save()

                    if transcribed_text and question_to_update.session.language_code == 'en':
                        follow_up_data = generate_and_save_follow_up(
                            session=question_to_update.session,
                            parent_question=question_to_update,
                            transcribed_answer=transcribed_text
                        )
                except InterviewQuestion.DoesNotExist:
                    print(f"Warning: Could not find MAIN question with index {question_index} to save answer or generate follow-up.")
            
            os.remove(full_path)
            return JsonResponse({'text': transcribed_text, 'follow_up_question': follow_up_data})
        except Exception as e:
            if os.path.exists(full_path): os.remove(full_path)
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request'}, status=400)


def video_feed(request):
    session_key = request.GET.get('session_key')
    if not session_key: return HttpResponse("Session key required.", status=400)
    camera = get_camera_for_session(session_key)
    if not camera: return HttpResponse("Camera not found for session.", status=404)
    return StreamingHttpResponse(gen(camera), content_type='multipart/x-mixed-replace; boundary=frame')

def gen(camera_instance):
    while True:
        frame = camera_instance.get_frame()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def get_proctoring_status(request):
    session_key = request.GET.get('session_key')
    if not session_key: return JsonResponse({}, status=400)
    camera = get_camera_for_session(session_key)
    if not camera: return JsonResponse({}, status=404)
    return JsonResponse(camera.get_latest_warnings())

@csrf_exempt
@require_POST
def report_tab_switch(request):
    data = json.loads(request.body)
    session_key = data.get('session_key')
    if not session_key: return JsonResponse({}, status=400)
    camera = get_camera_for_session(session_key)
    if not camera: return JsonResponse({}, status=404)
    camera.set_tab_switch_status(data.get('status') == 'hidden')
    return JsonResponse({"status": "ok"})

@csrf_exempt
def check_camera(request):
    session_key = request.GET.get('session_key')
    if not session_key: return JsonResponse({"status": "error", "message": "Session key is missing."}, status=400)
    camera = get_camera_for_session(session_key)
    if camera and camera.video.isOpened():
        return JsonResponse({"status": "ok"})
    else:
        release_camera_for_session(session_key)
        return JsonResponse({ "status": "error", "message": "Server could not start camera. Check console for errors." }, status=500)

@csrf_exempt
@require_POST
def release_camera(request):
    try:
        data = json.loads(request.body)
        session_key = data.get('session_key')
        if not session_key: return JsonResponse({'status': 'error', 'message': 'Session key required.'}, status=400)
        release_camera_for_session(session_key)
        return JsonResponse({'status': 'success', 'message': 'Camera released.'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

def extract_id_data(image_path):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    id_card_for_ocr = {'mime_type': 'image/jpeg', 'data': image_bytes}
    prompt = (
        "You are an OCR expert. Extract the following from the provided image of an ID card: "
        "- Full Name\n- ID Number\n"
        "If a value cannot be extracted, state 'Not Found'. Do not add any warnings.\n"
        "Format:\nName: <value>\nID Number: <value>"
    )
    # You probably need to send the image for OCR elsewhere; Perplexity cannot natively OCR images via API (as of 2025).
    # If you really want to send an image as base64 string in the prompt, you could, but results are unreliable.
    # Otherwise, use an OCR library (like pytesseract), get the text, and then prompt Perplexity with only the extracted text.

    # Example using pytesseract (recommended):
    
    pil_image = PIL.Image.open(image_path)
    ocr_text = pytesseract.image_to_string(pil_image)

    # Now ask Perplexity to parse name/id from OCR'd text
    ocr_prompt = (
        f"{prompt}\n\nID TEXT:\n{ocr_text}"
    )

    name = id_number = None
    try:
        perpl_resp = call_perplexity_api(ocr_prompt, model="llama-3.1-instruct")
        import re
        name_match = re.search(r"Name:\s*(.+)", perpl_resp, re.IGNORECASE)
        id_number_match = re.search(r"ID Number:\s*(.+)", perpl_resp, re.IGNORECASE)
        name = name_match.group(1).strip() if name_match else None
        id_number = id_number_match.group(1).strip() if id_number_match else None
    except Exception:
        pass
    return id_number, name

@csrf_exempt
@require_POST
def verify_id(request):
    try:
        image_data = request.POST.get('image_data') 
        session_id = request.POST.get('session_id')

        if not all([image_data, session_id]):
            return JsonResponse({'status': 'error', 'message': 'Missing required data.'}, status=400)

        session = InterviewSession.objects.get(id=session_id)
        
        format, imgstr = image_data.split(';base64,')
        ext = format.split('/')[-1]
        img_file = ContentFile(base64.b64decode(imgstr), name=f"id_{timezone.now().strftime('%Y%m%d%H%M%S')}.{ext}")
        session.id_card_image.save(img_file.name, img_file, save=True)
        
        tmp_path = session.id_card_image.path
        
        full_image = cv2.imread(tmp_path)
        if full_image is None:
            return JsonResponse({'status': 'error', 'message': 'Invalid image format.'})

        results = detect_face_with_yolo(full_image)
        boxes = results[0].boxes if results and hasattr(results[0], 'boxes') else []

        if len(boxes) < 2:
            return JsonResponse({'status': 'error', 'message': 'Verification failed. Please ensure both your face and the ID card are clearly visible, and that there are no other faces in the background.'})

        id_number, name = extract_id_data(tmp_path)
        session.extracted_id_details = f"Name: {name}, ID: {id_number}"
        
        invalid_phrases = ['not found', 'cannot be', 'unreadable', 'blurry', 'unavailable', 'missing']
        name_verified = name and len(name.strip()) > 2 and not any(phrase in name.lower() for phrase in invalid_phrases)

        if not name_verified:
            return JsonResponse({'status': 'error', 'message': f"Could not reliably read the name from the ID card. Extracted: '{name}'. Please try again."})
            
        if session.candidate_name.lower().split()[0] not in name.lower():
             return JsonResponse({'status': 'error', 'message': f"Name on ID ('{name}') does not match the registered name ('{session.candidate_name}')."})

        session.id_verification_status = 'Verified'
        session.save()

        return JsonResponse({'status': 'success', 'message': 'Verification successful!'})

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': f'An unexpected error occurred: {str(e)}'}, status=500)
    


    