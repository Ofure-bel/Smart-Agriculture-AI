# Smart-Agriculture-AI
# Installing Dependencies

Before running the Smart Agriculture AI application, ensure that Python is installed on your computer.

1. **Clone or download the project** and open a terminal in the project directory.
2. **Create a virtual environment** to isolate the project's dependencies:
   python -m venv venv

3. **Activate the virtual environment.**
   On Windows:
   venv\Scripts\activate
   
   On macOS or Linux:
   source venv/bin/activate

5. **Install the required dependencies** listed in requirements.txt`:
   pip install -r requirements.txt

6. **Run the Flask application** using the project's configured startup command. For example, if Gunicorn is installed and the application entry point is app.py`:
   gunicorn app:app

   Alternatively, use the development startup instructions provided by the project.
Ensure that any required model files and configuration files are present before starting the application.
