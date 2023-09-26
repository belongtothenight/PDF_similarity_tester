@ECHO OFF

ECHO "PDFST>> Setting up PIP ENVIRONMENT for PDF_similarity_tester with middle size model"

ECHO "PDFST>> Cloning PDF_similarity_tester from github"
git clone https://github.com/belongtothenight/PDF_similarity_tester.git
cd PDF_similarity_tester

ECHO "PDFST>> Updating pip"
python -m pip install --upgrade pip

ECHO "PDFST>> Installing packages"
pip install spacy numpy pandas PyPDF2 -U

ECHO "PDFST>> Installing spacy models"
python -m spacy download en_core_web_md
python -m spacy download zh_core_web_md

ECHO "PDFST>> Setup complete!!!"
ECHO "PDFST>> You can now run 'python ./src/main.py -h'"


