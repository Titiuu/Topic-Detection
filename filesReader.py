import jieba
import re
import os

lables = []  # 标签


# 获取answer目录下所有文件夹名称
def get_dirs(root):  # root = './answer'
    dirs = os.listdir(root)
    dirs.sort(key=lambda i: int(re.match(r'C(\d+)', i).group(1)))
    return dirs


#print(getdirs('./answer'))
# ['C3-Art', 'C4-Literature', 'C5-Education', 'C6-Philosophy', 'C7-History', 'C11-Space', 'C15-Energy',
# 'C16-Electronics', 'C17-Communication', 'C19-Computer', 'C23-Mine', 'C29-Transport', 'C31-Enviornment',
# 'C32-Agriculture', 'C34-Economy', 'C35-Law', 'C36-Medical', 'C37-Military', 'C38-Politics', 'C39-Sports']

# 获取所有要读取的文件的路径
def get_paths(root):
    dirs = get_dirs(root)
    paths = []
    for i in dirs:
        file_paths = os.listdir(root + '/' + i)
        if len(file_paths) > 50:  # 此参数用来控制读取每个类别文件夹下的文件数量
            file_paths = file_paths[:50]
        for j in range(0, len(file_paths)):
            paths.append(root + '/' + i + '/' + file_paths[j])
            lables.append(int(re.match(r'C(\d+)', i).group(1)))  # 生成标签列表
    return paths


#print(getpaths('./answer'))


# 停用词列表
def get_stopwords():
    stop_f = open('cn_stopwords.txt', "r", encoding='utf-8')
    stop_words = []
    for line in stop_f.readlines():
        line = line.strip()
        stop_words.append(line)
    stop_f.close()
    return stop_words


# 获取正文，分词，去停用词
def get_texts(paths):
    texts = []
    stopwords = get_stopwords()
    for filename in paths:
        print(filename)
        f = open(filename, "r", encoding='gb18030', errors='ignore')
        raw = f.read()
        raw = re.sub(r'\n', '', raw)
        raw = re.sub(r'[\d()〔〕（）.,]+', '', raw)  # 此正则语句会对分类准确度造成影响，分类时去掉
        outstr = ''
        seg_list = jieba.cut(raw, cut_all=False)
        for word in seg_list:
            if word not in stopwords:
                outstr += word
                outstr += ' '
        outstr = outstr.strip()
        f.close()
        texts.append(outstr)
    return texts


#返回内容列表
def contents(root):
    paths = get_paths(root)
    return get_texts(paths)


# 返回内容和标签列表
def contents_and_labels(root):
    paths = get_paths(root)
    lablelist = lables
    return get_texts(paths), lablelist


# 分成词向量的函数
def split_into_words(i):
    return i.split(" ")

# print(get_texts(['./answer/C3-Art/C3-Art0053.txt']))
