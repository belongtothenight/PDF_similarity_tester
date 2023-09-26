# PDF_similarity_tester
 
This repo aims for developing a automic batch PDF similarity tester.

## Install

1. Prepare working python environment.
2. Install [git](https://git-scm.com/download/win).
3. Download one of the following batch files.
   1. [cmd_install_sm.bat](https://github.com/belongtothenight/PDF_similarity_tester/blob/main/install/cmd_install_sm.bat) for small model.
   2. [cmd_install_md.bat](https://github.com/belongtothenight/PDF_similarity_tester/blob/main/install/cmd_install_md.bat) for medium model.
   3. [cmd_install_lg.bat](https://github.com/belongtothenight/PDF_similarity_tester/blob/main/install/cmd_install_lg.bat) for large model.
   4. [cmd_install_trf.bat](https://github.com/belongtothenight/PDF_similarity_tester/blob/main/install/cmd_install_trf.bat) for transformer model.
4. Execute "./src/main.py".

## Reference

1. [Compare documents similarity using Python | NLP](https://dev.to/thepylot/compare-documents-similarity-using-python-nlp-4odp)
2. [Extract text from PDF File using Python](https://www.geeksforgeeks.org/extract-text-from-pdf-file-using-python/)
3. [How to extract images from PDF in Python?](https://www.geeksforgeeks.org/how-to-extract-images-from-pdf-in-python/)
4. [PyPDF2 Documentation](https://pypdf2.readthedocs.io/en/3.0.0/index.html)
5. [Calculating Text Similarity in Python with NLP](https://www.youtube.com/watch?v=y-EjAuWdZdI)

## Release

- Can't pack NLP model used in spaCy inside of EXE, therefore no windows executable is released.
