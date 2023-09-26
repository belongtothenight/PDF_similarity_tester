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
        self.ERROR_str = "ERROR: "
        self.END_str = " ERROR"

    def folder_invalid(self, folderpath:str) -> None:
        print("{}Folderpath is invalid: {}.{}".format(self.ERROR_str, folderpath, self.END_str))
        sys.exit(0)

    def file_read_error(self, filepath:str) -> None:
        print("{}File read error: {}.{}".format(self.ERROR_str, filepath, self.END_str))
        sys.exit(0)

    def invalid_argument(self, args) -> None:
        print("{}Invalid arguments: {}.{}".format(self.ERROR_str, args, self.END_str))
        sys.exit(0)

    def argument_not_enough(self) -> None:
        print("{}Not enough arguments.{}".format(self.ERROR_str, self.END_str))
        sys.exit(0)

class PDFSimilarityTester:
    def __init__(self, 
                 nlpm_en, 
                 nlpm_zh,
                 folderpath:str, 
                 export_detail_directory:str, 
                 export_similarity_directory:str, 
                 weight=[0.9, 0, 0.1, 0], # [text_similarity, img_similarity, text_hash_similarity, img_hash_similarity]
                 nlpm_weight=[0.45, 0.45, 0.1, 0]) -> None:
        #* Initialize class
        self.nlpm_en = nlpm_en
        self.nlpm_zh = nlpm_zh
        self.folderpath = folderpath
        self.weight = weight
        self.export_detail_directory = export_detail_directory
        self.export_similarity_directory = export_similarity_directory
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
        self.__walk_dir(self.folderpath)
        self.PDF_detail_dataframe["filename"] = self.PATH_list
        logging.debug("PATH_list: {}".format(self.PATH_list))
        logging.debug("PATH_list length: {}".format(len(self.PATH_list)))
        print("Loaded {} PDF files".format(len(self.PATH_list)))

    def __walk_dir(self, dirpath:str) -> None:
        for root, dirs, files in os.walk(dirpath):
            for file in files:
                if file.endswith(".pdf"):
                    self.PATH_list.append(os.path.join(root, file))
            for dir in dirs:
                self.__walk_dir(dir)

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

    def __text_similarity(self, idx1:int, idx2:int, text_list:list, nlpm_object) -> float:
        #* Compare text
        s1 = nlpm_object(text_list[idx1])
        s2 = nlpm_object(text_list[idx2])
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
        idx_list = range(len(self.PATH_list))
        for obj_idx1, obj_idx2 in itertools.combinations(idx_list, 2):
            self.filename1_list.append(obj_idx1)
            self.filename2_list.append(obj_idx2)
            self.text_en_similarity_list.append(self.__text_similarity(obj_idx1, obj_idx2, self.text_en_list, self.nlpm_en))
            self.text_zh_similarity_list.append(self.__text_similarity(obj_idx1, obj_idx2, self.text_zh_list, self.nlpm_zh))
            self.text_num_similarity_list.append(self.__text_similarity(obj_idx1, obj_idx2, self.text_num_list, self.nlpm_en))
            self.text_other_similarity_list.append(self.__text_similarity(obj_idx1, obj_idx2, self.text_other_list, self.nlpm_en))
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

def validate_dir(dirpath:str) -> bool:
    #* Check if dirpath is valid
    if not os.path.isdir(dirpath):
        PDFST_Error.folder_invalid(dirpath)
    else:
        pass

def execute():
    """
    PDF Similarity Tester
    Usage: python main.py <input_dir> <export_detail_dir> <export_result_dir> [--weight=<weight>] [--nlpmweight=<w1>,<w2>,<w3>,<w4>] [-h/--help]
    !!!For URL links: ONLY ACCEPT YOUTUBE LINKS!!!
    Weight calculation: (text_mix * weight) + (tex_hash * (1-weight))
    NLPM Weight calculation: (en_similarity * w1) + (zh_similarity * w2) + (num_similarity * w3) + (other_similarity * w4)
    sys.argv[1] path of input folder (contains PDF files even inside subfolder)
    sys.argv[2] path of export detail folder
    sys.argv[3] path of export result folder
    sys.argv[?] (--weight) weight of videohash method (default: 0.9)
    sys.argv[?] (--nlpmweight) weight of NLPM method (default: 0.45, 0.45, 0.1, 0)
    sys.argv[?] (-h/--help) help (show available options)
    """
    #* Check arguments
    available_short_options = "h:"
    available_long_options = ["weight=", "nlpmweight=", "help"]
    try:
        opts, args = getopt.getopt(sys.argv[4:], available_short_options, available_long_options)
    except getopt.GetoptError:
        logging.critical("Invalid arguments.")
        PDFST_Error().invalid_argument(sys.argv[4:])
    #* Check for help
    if len(sys.argv) < 4:
        if '-h' in sys.argv or '--help' in sys.argv:
            print(execute.__doc__)
            sys.exit()
        else:
            logging.critical("Not enough arguments.")
            PDFST_Error().argument_not_enough()
    #* Initialize variable with default value
    weight = [0.9, 0, 0.1, 0]
    nlpm_weight = [0.47, 0.47, 0.06, 0]
    #* Parse arguments
    input_dir = sys.argv[1]
    export_detail_dir = sys.argv[2]
    export_result_path = sys.argv[3]
    #* Parse arguments
    for opt, arg in opts:
        if opt in ("--weight"):
            weight[0] = float(arg)
            weight[2] = 1 - weight[0]
        elif opt in ("--nlpmweight"):
            nlpm_weight = [float(num) for num in arg.split(",")]
    #* Validate input
    validate_dir(input_dir)
    validate_dir(export_detail_dir)
    validate_dir(export_result_path)
    #* Read config.txt
    with open("config.txt", "r") as f:
        config = str(f.readlines()[0])
        config = re.findall(r"[a-zA-Z]+", config)[0]
    if config == "sm":
        import en_core_web_sm as nlpm_en
        import zh_core_web_sm as nlpm_zh
    elif config == "md":
        import en_core_web_md as nlpm_en
        import zh_core_web_md as nlpm_zh
    elif config == "lg":
        import en_core_web_lg as nlpm_en
        import zh_core_web_lg as nlpm_zh
    elif config == "trf":
        import en_core_web_trf as nlpm_en
        import zh_core_web_trf as nlpm_zh
    else:
        logging.critical("Invalid config.txt")
        PDFST_Error().invalid_argument('config.txt')
    #* Load NLP model
    print("Loading en model...", end="\r")
    nlpm_en = nlpm_en.load()
    print("Loading en model... Done")
    print("Loading zh model...", end="\r")
    nlpm_zh = nlpm_zh.load()
    print("Loading zh model... Done")
    #* Execute
    PDFSimilarityTester(nlpm_en=nlpm_en,
                        nlpm_zh=nlpm_zh,
                        folderpath=input_dir, 
                        export_detail_directory=export_detail_dir, 
                        export_similarity_directory=export_result_path, 
                        weight=weight, 
                        nlpm_weight=nlpm_weight)
    

        
if __name__ == "__main__":
    # folderpath = "../similarity_test"
    # PDFSimilarityTester(folderpath, folderpath, folderpath)
    execute()
