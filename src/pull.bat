@ECHO OFF

ECHO Setting up PIP ENVIRONMENT for PDF_similarity_tester with middle size model

ECHO Cloning PDF_similarity_tester from github
git clone https://github.com/belongtothenight/PDF_similarity_tester.git
cd PDF_similarity_tester

ECHO Updating pip
python -m pip install --upgrade pip

ECHO Installing virtualenv
pip install virtualenv -U

ECHO Create virtual environment
python -m venv env

ECHO Activate virtual environment
env/bin/activate.bat

ECHO Installing packages
pip install spacy numpy pandas PyPDF2 -U

ECHO Installing spacy models
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm

ECHO Setup complete!!!
ECHO You can now run "./main.py"

