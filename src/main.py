from PyPDF2 import PdfReader
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
import pandas as pd
import numpy as np
import logging
import os
import itertools
import sys
import getopt
import hashlib

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.WARNING)
# logging.basicConfig(level=logging.ERROR)
# logging.basicConfig(level=logging.CRITICAL)

class PDFST_Error:
    def __init__(self) -> None:
        pass

    def folder_invalid(folderpath:str) -> None:
        print("Folderpath is invalid: {}".format(folderpath))
        sys.exit(0)

    def file_read_error(filepath:str) -> None:
        print("File read error: {}".format(filepath))
        sys.exit(0)

class PDFSimilarityTester:
    def __init__(self, folderpath:str, export_detail_directory:str, export_similarity_directory:str, weight=[0.45, 0.45, 0.05, 0.05]) -> None:
        #* Initialize class
        self.folderpath = folderpath
        self.weight = weight
        self.export_detail_directory = export_detail_directory
        self.export_similarity_directory = export_similarity_directory
        #* Check if folderpath is valid
        if not os.path.isdir(folderpath):
            PDFST_Error.folder_invalid(folderpath)
        #* Create variable
        self.PDF_detail_dataframe = pd.DataFrame(columns=["filename", "text_hash", "img_hash"])
        self.PDF_similarity_dataframe = pd.DataFrame(columns=["filename1", "filename2", "text_similarity", "img_similarity", "text_hash_similarity", "img_hash_similarity", "mix_similarity"])
        self.PATH_list = np.empty(0, dtype=str)
        self.reader_list = []
        self.text_list = np.empty(0, dtype=str)
        self.hasher_list = []
        self.text_hash_list = np.empty(0, dtype=str)
        self.tokens = []
        self.dictionary = None

        self.filename1_list = np.empty(0, dtype=str)
        self.filename2_list = np.empty(0, dtype=str)
        self.text_similarity_list = np.empty(0, dtype=float)
        self.img_similarity_list = np.empty(0, dtype=float)
        self.text_hash_similarity_list = np.empty(0, dtype=float)
        self.img_hash_similarity_list = np.empty(0, dtype=float)
        self.mix_similarity_list = np.empty(0, dtype=float)
        #* Main Flow
        self._load_PATH()
        self._text_extractor()
        self._img_extractor()
        self._text_hasher()
        self._img_hasher()
        self._export_detail()
        self._generate_result()

    def _load_PATH(self) -> None:
        #* Load PATH of PDF files in folderpath
        for root, dirs, files in os.walk(self.folderpath):
            for file in files:
                if file.endswith(".pdf"):
                    self.PATH_list = np.append(self.PATH_list, os.path.join(root, file))
        self.PDF_detail_dataframe["filename"] = self.PATH_list
        logging.debug("PATH_list: {}".format(self.PATH_list))
        logging.debug("PATH_list.shape: {}".format(self.PATH_list.shape))
        print("Loaded {} PDF files".format(self.PATH_list.shape[0]))

    def _text_extractor(self) -> None:
        #* Extract text from PDF files
        for path in self.PATH_list:
            try:
                reader = PdfReader(path)
                self.reader_list.append(reader)
            except:
                PDFST_Error.file_read_error(path)
        logging.debug("reader_list: {}".format(self.reader_list))
        logging.debug("reader_list.shape: {}".format(len(self.reader_list)))
        for reader in self.reader_list:
            tmpString = ""
            for page in reader.pages:
                tmpString += page.extract_text()
            self.text_list = np.append(self.text_list, tmpString)
        logging.debug("text_list: {}".format(self.text_list))
        logging.debug("text_list.shape: {}".format(self.text_list.shape))
        print("Extracted text from {} PDF files".format(self.text_list.shape[0]))

    def _img_extractor(self) -> None:
        #* Extract images from PDF files
        pass

    def _text_hasher(self) -> None:
        #* Hash text
        for text in self.text_list:
            self.text_hash_list = np.append(self.text_hash_list, hashlib.sha256(text.encode()).hexdigest())
        self.PDF_detail_dataframe["text_hash"] = self.text_hash_list
        logging.debug("text_hash_list: {}".format(self.text_hash_list))
        logging.debug("text_hash_list.shape: {}".format(len(self.text_hash_list)))
        print("Hashed text from {} PDF files".format(len(self.text_hash_list)))

    def _img_hasher(self) -> None:
        #* Hash images
        pass

    def __text_similarity(self, idx1:int, idx2:int) -> float:
        #* Compare text
        return 0

    def __text_hash_similarity(self, idx1:int, idx2:int) -> float:
        #* Compare text hash
        return 0

    def __img_similarity(self, idx1:int, idx2:int) -> float:
        #* Compare images
        return 0

    def __img_hash_similarity(self, idx1:int, idx2:int) -> float:
        #* Compare images hash
        return 0

    def _export_detail(self) -> None:
        #* Export detail
        path = os.path.join(self.export_detail_directory, "PDF_detail.csv")
        path = os.path.abspath(path)
        self.PDF_detail_dataframe.to_csv(path, index=False)
        print("Exported detail to PDF_detail.csv")

    def _generate_result(self) -> None:
        #* Generate result
        for i, j in itertools.combinations(self.PATH_list, 2):
            obj_idx1 = self.PATH_list.tolist().index(i)
            obj_idx2 = self.PATH_list.tolist().index(j)
            self.filename1_list = np.append(self.filename1_list, i)
            self.filename2_list = np.append(self.filename2_list, j)
            self.text_similarity_list = np.append(self.text_similarity_list, self.__text_similarity(obj_idx1, obj_idx2))
            self.img_similarity_list = np.append(self.img_similarity_list, self.__img_similarity(obj_idx1, obj_idx2))
            self.text_hash_similarity_list = np.append(self.text_hash_similarity_list, self.__text_hash_similarity(obj_idx1, obj_idx2))
            self.img_hash_similarity_list = np.append(self.img_hash_similarity_list, self.__img_hash_similarity(obj_idx1, obj_idx2))
            self.mix_similarity_list = np.append(self.mix_similarity_list, self.weight[0]*self.text_similarity_list[-1] + self.weight[1]*self.img_similarity_list[-1] + self.weight[2]*self.text_hash_similarity_list[-1] + self.weight[3]*self.img_hash_similarity_list[-1])
        self.PDF_similarity_dataframe["filename1"] = self.filename1_list
        self.PDF_similarity_dataframe["filename2"] = self.filename2_list
        self.PDF_similarity_dataframe["text_similarity"] = self.text_similarity_list
        self.PDF_similarity_dataframe["img_similarity"] = self.img_similarity_list
        self.PDF_similarity_dataframe["text_hash_similarity"] = self.text_hash_similarity_list
        self.PDF_similarity_dataframe["img_hash_similarity"] = self.img_hash_similarity_list
        self.PDF_similarity_dataframe["mix_similarity"] = self.mix_similarity_list
        logging.debug("DataFrame shape: {}".format(self.PDF_similarity_dataframe.shape))
        path = os.path.join(self.export_similarity_directory, "PDF_similarity.csv")
        path = os.path.abspath(path)
        self.PDF_similarity_dataframe.to_csv(path, index=False)
        print("Exported result to PDF_similarity.csv")



        
if __name__ == "__main__":
    folderpath = "../similarity_test"
    PDFSimilarityTester(folderpath, folderpath, folderpath)
