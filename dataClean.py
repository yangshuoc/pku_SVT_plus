import os
import re
SRC_DIR = 'Training_Data'
TARGET_DIR = 'Clean_Data'
HEADER_END = '*END*THE'
WORD_SP = ' '
LINE_SP = '.'
MIN_LINE_NUM = 5
MIN_SENTENCE_NUM = 2


def open_file(filename, mode='r'):
    # return open(filename, mode=mode, encoding='utf-8', errors='ignore')
    return open(filename, mode=mode, encoding='ascii', errors='ignore')

def count_word_num(str):
    list = str.split(WORD_SP)
    return len(list)

def filter_data():
    for fname in sorted(os.listdir(SRC_DIR)):
        target_file = open_file(os.path.join(TARGET_DIR, fname), 'w')    
        lines = open_file(os.path.join(SRC_DIR, fname)).readlines()
        headerSkiped = False
        #句子列表
        slist = []
        sentence = ''
        for line in lines:
            #略过头信息
            if headerSkiped == False and line.startswith(HEADER_END) == False:
                continue
            headerSkiped = True
            if line.startswith(HEADER_END) == True:
                continue
            if count_word_num(line) <= MIN_LINE_NUM:
                continue
            line = line.replace('\n',' ')
            for char in line:
                #英文or空格字符就加入句子里
                if char.isalpha() or char.isspace() or char == '\'':
                    sentence += char
                #非英文字符作为句子结束
                else:
                    #05-29这里改成以.结尾
                    if char != '.':
                        sentence += ' '

                    #合并空格，转小写
                    sentence = re.sub(' +', ' ', sentence.strip().lower())                    
                     #句子长度不太短
                    if count_word_num(sentence) >= MIN_SENTENCE_NUM:
                        #加<bos> <eos>
                        # sentence = '_bos_ ' + sentence + ' _eos_'
                        slist.append(sentence)
                        sentence = ''
        for s in slist:
            target_file.write(s+'\n')      
        target_file.close()
        print(fname +' processed\n')

#python dataClean.py
if __name__ == '__main__':
    print("begin data clean")
    filter_data()
    print("data clean finished")
    print("all file saved to Clean_Data")