## Problem Statement

An intent to expediate the investigation for any crime scene or to take preventive action to neutralize an intended antisocial activity (information received through intel), we wish to implement an extended application of Facial Recognition System (FRS). Real-Time validation of individuals with crime history, Movement Tracking of Suspects, Identification of People loitering in Unauthorized area, Smart Police Deployment.  

---

## Overview
OVERVIEW:
Our SENTINEL-AI utilizes facial recognition and AI by:

Ø Implementing Real-Time Identification of a suspect
Ø Reducing Human Errors
Ø Lowering time consumption
Ø Improving Crime Prevention

SENTINEL-AI is designed to serve a wide range of clients including national security, law enforcement, border patrol, and immigration checks.

SOLUTION:
This system automates decision making using facial recognition. It enhances the efficiency of agencies by enabling real-time facial recognition of suspects. By combining facial recognition and Intelligent Surveillance, we can predict criminals by monitoring their presence in an area.

---

## Technical Stack

- Frontend: HTML, CSS, JavaScript
- Backend: Flask, Python
- Database: MongoDB
- Other Tools: Git, PyTorch, TensorFlow, OpenCV, MediaPipe, YOLOv8

---

## Getting Started

Follow these steps to clone and run the application locally.

### Prerequisites

1. Install [Python 3.10+](https://www.python.org/downloads/).
2. Install [Git](https://git-scm.com/).
3. Install [MongoDB](https://www.mongodb.com/try/download/community) and ensure it's running locally on `localhost:27017`.
4. Clone this repository:
   ```bash
   git clone https://github.com/skk2006/SIH.git
   ```

### Installation

1. Navigate to the project directory:
   ```bash
   cd SIH
   ```
   
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
   
3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
     
4. Install dependencies:
   ```bash
   pip install -r AI/requirements.txt
   ```

### Environment Variables

Before running the application, set up your Twilio and Email credentials. Create a `.env` file in the `AI/source` folder or set these in your environment:

```env
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_email_password
EMAIL_RECEIVER=receiver_email@gmail.com
TWILIO_ACCOUNT_SID=your_account_sid_here
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=your_twilio_phone_number
RECIPIENT_PHONE_NUMBER=your_recipient_phone_number
```

---

## Start the Application

1. Navigate to the source folder:
   ```bash
   cd AI/source
   ```

2. Run the application:
   - Using the batch script (Windows):
     ```bash
     run.bat
     ```
   - Using Python directly:
     ```bash
     python app.py
     ```

3. Open your browser and navigate to:
   http://127.0.0.1:5000/
