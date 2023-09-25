from PyPDF2 import PdfReader
import pandas as pd
import numpy as np
import logging
import os
import itertools
import sys
import getopt
import hashlib
import spacy
import re

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
    def __init__(self, 
                 folderpath:str, 
                 export_detail_directory:str, 
                 export_similarity_directory:str, 
                 weight=[0.9, 0, 0.1, 0], # [text_similarity, img_similarity, text_hash_similarity, img_hash_similarity]
                 nlpm_name="en_core_web_md",
                 nlpm_weight=[0.45, 0.45, 0.1, 0]) -> None:
        #* Initialize class
        self.folderpath = folderpath
        self.weight = weight
        self.export_detail_directory = export_detail_directory
        self.export_similarity_directory = export_similarity_directory
        self.nlpm_name = nlpm_name # Natural Language Processing Model Name
        self.nlpm_weight = nlpm_weight
        #* Check if folderpath is valid
        if not os.path.isdir(folderpath):
            PDFST_Error.folder_invalid(folderpath)
        #* Create variable
        self.PDF_detail_dataframe = pd.DataFrame(columns=["filename", "text_hash", "img_hash"])
        self.PDF_similarity_dataframe = pd.DataFrame(columns=["filename1", 
                                                              "filename2", 
                                                              "text_en_similarity", 
                                                              "text_zh_similarity", 
                                                              "text_num_similarity", 
                                                              "text_other_similarity", 
                                                              "text_mix_similarity", 
                                                            #   "img_similarity", 
                                                              "text_hash_similarity", 
                                                            #   "img_hash_similarity", 
                                                              "mix_similarity"])
        self.PATH_list = []
        self.reader_list = []
        self.text_list = []
        self.hasher_list = []
        self.text_hash_list = []
        self.text_en_list = []
        self.text_zh_list = []
        self.text_num_list = []
        self.text_other_list = []

        self.filename1_list = []
        self.filename2_list = []
        self.text_en_similarity_list = []
        self.text_zh_similarity_list = []
        self.text_num_similarity_list = []
        self.text_other_similarity_list = []
        self.text_mix_similarity_list = []
        self.img_similarity_list = []
        self.text_hash_similarity_list = []
        self.img_hash_similarity_list = []
        self.mix_similarity_list = []
        #* Main Flow
        self._load_PATH()
        self._text_extractor()
        self._text_hasher(self.text_list, self.text_hash_list)
        self._text_language_splitter()
        self._export_detail()
        self._generate_result()

    def _load_PATH(self) -> None:
        #* Load PATH of PDF files in folderpath
        for root, dirs, files in os.walk(self.folderpath):
            for file in files:
                if file.endswith(".pdf"):
                    self.PATH_list.append(os.path.join(root, file))
        self.PDF_detail_dataframe["filename"] = self.PATH_list
        logging.debug("PATH_list: {}".format(self.PATH_list))
        logging.debug("PATH_list length: {}".format(len(self.PATH_list)))
        print("Loaded {} PDF files".format(len(self.PATH_list)))

    def _text_extractor(self) -> None:
        #* Extract text from PDF files
        for path in self.PATH_list:
            try:
                reader = PdfReader(path)
                self.reader_list.append(reader)
            except:
                PDFST_Error.file_read_error(path)
        logging.debug("reader_list: {}".format(self.reader_list))
        logging.debug("reader_list length: {}".format(len(self.reader_list)))
        for reader in self.reader_list:
            tmpString = ""
            for page in reader.pages:
                tmpString += page.extract_text()
            self.text_list.append(tmpString)
        logging.debug("text_list: {}".format(self.text_list))
        print("Extracted text from {} PDF files".format(len(self.text_list)))

    def _text_hasher(self, text_list:list, hash_list:list) -> None:
        #* Hash text
        for text in text_list:
            hash_list.append(hashlib.sha256(text.encode()).hexdigest())
        self.PDF_detail_dataframe["text_hash"] = hash_list
        print("Hashed text from {} PDF files".format(len(hash_list)))

    def _text_language_splitter(self) -> None:
        for text in self.text_list:
            self.text_en_list.append(' '.join(re.findall(r"[a-zA-Z]+", text)))
            self.text_zh_list.append(' '.join(re.findall(r"[\u4e00-\u9fa5]+", text)))
            self.text_num_list.append(' '.join(re.findall(r"[0-9]+", text)))
            self.text_other_list.append(' '.join(re.findall(r"[^a-zA-Z\u4e00-\u9fa5]+", text)))
        logging.debug("text_en_list.shape: {}".format(len(self.text_en_list)))
        logging.debug("text_zh_list.shape: {}".format(len(self.text_zh_list)))
        print("Splitted text from {} PDF files".format(len(self.text_en_list)))

    def __text_similarity(self, idx1:int, idx2:int, text_list:list) -> float:
        #* Compare text
        s1 = self.nlpm(text_list[idx1])
        s2 = self.nlpm(text_list[idx2])
        return s1.similarity(s2)

    def __text_hash_similarity(self, idx1:int, idx2:int, text_hash_list) -> float:
        #* Compare text hash
        diffcnt = 0
        hash_len = len(text_hash_list[idx1])
        for i in range(hash_len):
            if text_hash_list[idx1][i] != text_hash_list[idx2][i]:
                diffcnt += 1
        return (hash_len - diffcnt) / hash_len

    def __img_similarity(self, idx1:int, idx2:int) -> float:
        #* Compare image
        return 0
    
    def __img_hash_similarity(self, idx1:int, idx2:int) -> float:
        #* Compare image hash
        return 0

    def _export_detail(self) -> None:
        #* Export detail
        path = os.path.join(self.export_detail_directory, "PDF_detail.csv")
        path = os.path.abspath(path)
        self.PDF_detail_dataframe.to_csv(path, index=False)
        print("Exported detail to PDF_detail.csv")

    def _generate_result(self) -> None:
        #* Generate result
        print("Loading model...", end="\r")
        self.nlpm = spacy.load(self.nlpm_name)
        print("Loading model... Done")
        idx_list = range(len(self.PATH_list))
        for obj_idx1, obj_idx2 in itertools.combinations(idx_list, 2):
            self.filename1_list.append(obj_idx1)
            self.filename2_list.append(obj_idx2)
            self.text_en_similarity_list.append(self.__text_similarity(obj_idx1, obj_idx2, self.text_en_list))
            self.text_zh_similarity_list.append(self.__text_similarity(obj_idx1, obj_idx2, self.text_zh_list))
            self.text_num_similarity_list.append(self.__text_similarity(obj_idx1, obj_idx2, self.text_num_list))
            self.text_other_similarity_list.append(self.__text_similarity(obj_idx1, obj_idx2, self.text_other_list))
            self.text_mix_similarity_list.append(np.dot(self.nlpm_weight, [self.text_en_similarity_list[-1], self.text_zh_similarity_list[-1], self.text_num_similarity_list[-1], self.text_other_similarity_list[-1]]))
            self.img_similarity_list.append(self.__img_similarity(obj_idx1, obj_idx2))
            self.text_hash_similarity_list.append(self.__text_hash_similarity(obj_idx1, obj_idx2, self.text_hash_list))
            self.img_hash_similarity_list.append(self.__img_hash_similarity(obj_idx1, obj_idx2))
            self.mix_similarity_list.append(np.dot(self.weight, [self.text_mix_similarity_list[-1], self.img_similarity_list[-1], self.text_hash_similarity_list[-1], self.img_hash_similarity_list[-1]]))
        self.PDF_similarity_dataframe["filename1"] = self.filename1_list
        self.PDF_similarity_dataframe["filename2"] = self.filename2_list
        self.PDF_similarity_dataframe["text_en_similarity"] = self.text_en_similarity_list
        self.PDF_similarity_dataframe["text_zh_similarity"] = self.text_zh_similarity_list
        self.PDF_similarity_dataframe["text_num_similarity"] = self.text_num_similarity_list
        self.PDF_similarity_dataframe["text_other_similarity"] = self.text_other_similarity_list
        self.PDF_similarity_dataframe["text_mix_similarity"] = self.text_mix_similarity_list
        # self.PDF_similarity_dataframe["img_similarity"] = self.img_similarity_list
        self.PDF_similarity_dataframe["text_hash_similarity"] = self.text_hash_similarity_list
        # self.PDF_similarity_dataframe["img_hash_similarity"] = self.img_hash_similarity_list
        self.PDF_similarity_dataframe["mix_similarity"] = self.mix_similarity_list
        logging.debug("DataFrame shape: {}".format(len(self.PDF_similarity_dataframe)))
        path = os.path.join(self.export_similarity_directory, "PDF_similarity.csv")
        path = os.path.abspath(path)
        self.PDF_similarity_dataframe.to_csv(path, index=False)
        print("Exported result to PDF_similarity.csv")



        
if __name__ == "__main__":
    folderpath = "../similarity_test"
    PDFSimilarityTester(folderpath, folderpath, folderpath)
