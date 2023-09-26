@ECHO OFF

ECHO PDFST::Setting up PIP ENVIRONMENT for PDF_similarity_tester with middle size model

ECHO PDFST::Cloning PDF_similarity_tester from github
CALL git clone https://github.com/belongtothenight/PDF_similarity_tester.git

ECHO PDFST::Updating pip
CALL python -m pip install --upgrade pip

ECHO PDFST::Installing packages
CALL pip install spacy numpy pandas PyPDF2 -U

ECHO PDFST::Installing spacy models
CALL python -m spacy download en_core_web_trf
CALL python -m spacy download zh_core_web_trf

ECHO PDFST::Create config.txt file
CALL "trf">.\PDF_similarity_tester\config.txt

ECHO PDFST::Setup complete!!!
ECHO PDFST::You can now run 'python ./src/main.py -h'
