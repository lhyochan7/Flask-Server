import json
from flask import Flask
from flask import jsonify

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from gensim.models import KeyedVectors
from googletrans import Translator
import mysql.connector
import pymysql
import cv2
from mysql.connector import Error
import statistics
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances



DB_weights = []

# Multiple Feature Extraction
# input: model = resnet50 모델, dataset= 데이터셋 (파일 idx, 이미지 파일 경로)
# output: 추출된 feature 반환
def extract_features(model, dataset):
    feature_list = []
    for (a,b) in dataset:
        sampleImg = cv2.imread(b)
        sampleImg = cv2.resize(sampleImg, (224,224))
        
        transform = transforms.ToTensor()
        sampleImg = transform(sampleImg)

        img = sampleImg.reshape(1,3,224,224)

        out = model(img)

        feature_list.append((a,out))

    return feature_list
        
# Single Feature Extraction
# input: model = resnet50 모델, imgPath= 이미지 파일 경로
# output: 추출된 feature 반환
def extract_feature(model, imgPath):

    sampleImg = cv2.imread(imgPath)

    if(sampleImg is not None):
        sampleImg = cv2.resize(sampleImg, (224,224))
        
        transform = transforms.ToTensor()
        sampleImg = transform(sampleImg)

        img = sampleImg.reshape(1,3,224,224)

        out = model(img)
    else:
        print('\n' + imgPath + ' 파일을 못찾습니다')
        out = torch.Tensor([[0] * 2048])

    return out  #(1,2048)


# input: keywords = AI 인공지능 추천 키워드 list  | keyword_list = DB 전시회 추천 키워드 list
# output: keyword 유사도 list 반환
def getKeywordSimilarity(word2vec_model, AI_keywords, DB_keywords):
    keyword_similarity = []
    trans = Translator()
    
    for keyword in AI_keywords:
        # DB 전시회 keyword
        for key in DB_keywords:
            if keyword and key in word2vec_model:
                similarity = word2vec_model.similarity(keyword, key)
                keyword_similarity.append(similarity)
            else:
                print('not in vocab')

    return keyword_similarity

# input: AI 추천 이미지 index 번호
# output: feature extraction된 list 반환
def inputImageFeatList(values):
    rec_feat_list = []
    for v in values:
        filePath = "C:\\img\\AI_images\\" + str(v) + ".png"
        rec_feat_list.append(extract_feature(res50_model, filePath))

    print(len(rec_feat_list))

    return rec_feat_list


# input: exhib_weights = (exhib seq, 키워드 유사도, 추출된 특징(resnet)), rec_feat_list = AI 추천 이미지 추출된 feature list
# output: 추천 전시회 5개 반환
def findExhibRecommendation(exhib_weights, rec_feat_list):
    exhib_recommendation_list = []

    for seq, sim, feat in exhib_weights:
        print(seq)
        print(sorted(sim, reverse=True))    
        #print(feat)
        for f in rec_feat_list:
            dist = cosine_similarity(f, feat)
            print("dist = ", dist, "||  mean(sim) = ", statistics.mean(sim[:5]), "||  sum = ", dist+statistics.mean(sim))
            exhib_recommendation_list.append((seq, dist+statistics.mean(sim[:5])*0.3))

    exhib_recommendation_list = sorted(exhib_recommendation_list, key=lambda x:x[1], reverse=True)

    print(exhib_recommendation_list[:5])

    return exhib_recommendation_list[:5]


# input: X
# output: exhib_seq, exhib_theme, file_dir_name, file_uuid, file_name of DB tb_exhibit
def DBConnection():
      # MySQL connection 
    try:
        connection = mysql.connector.connect(host='52.79.127.168',
                                            database='peanart',
                                            port=3306,
                                            user='root',
                                            password='pean!@art4k@c#u!f$2@3',
                                            auth_plugin='mysql_native_password')
        if connection.is_connected():
            db_Info = connection.get_server_info()
            print("Connected to MySQL Server version ", db_Info)
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            cursor.execute("select database();")
            record = cursor.fetchone()
            print("You're connected to database: ", record)

            sql = """SELECT A.exhib_seq, A.exhib_theme, B.file_dir_name, B.file_uuid, B.file_name
                FROM tb_exhibit A
                LEFT JOIN tb_file_exhibit B
                ON A.exhib_seq = B.exhib_seq"""

            cursor.execute(sql)

            rows = cursor.fetchall()

    except Error as e:
        print("Error while connecting to MySQL", e)

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

            return rows



# input: 추천 키워드 list, AI 추천 이미지 번호
# output: 추천 전시회 seq 번호 list 반환
def recommend(res50_model, word2vec_model, AI_keywords, AI_idx):

    print('DB CONNECTION 시작')
    # DB 연결 후 데이터 반환
    rows = DBConnection()
    print('DB CONNECTION 완료')
    
    print('AI input 이미지 특징 추출 시작')
    # AI input 추천 이미지들의 feature list으로 반환
    rec_feat_list = inputImageFeatList(AI_idx)
    print('AI input 이미지 특징 추출 완료')

    print('AI keyword 바꿈 시작')
    # AI 키워드 한영 변환 및 lowercase으로 바꿈 ( Korean -> English )
    trans = Translator()
    result = trans.translate(AI_keywords, src='ko', dest='en')
    AI_keywords = result.text.split(', ')
    AI_keywords = [word.lower() for word in AI_keywords]
    print("AI keywords=", AI_keywords)
    print('AI keyword 바꿈 완료')

    exhib_weights = []

    print('DB 전시회 파일 탐색 시작')
    ########################  DB 전시회 각 이미지 파일들 탐색 ############################
    for exhib_seq, exhib_theme, file_dir_name, file_uuid, file_name in rows:
        print(exhib_theme)
        #exhib_theme_list = []
        trans = Translator()

        # exhib_theme 키워드 split 하기
        result = trans.translate(exhib_theme, src='ko', dest='en')
        DB_keywords = result.text.split(', ')
        DB_keywords = [word.lower() for word in DB_keywords]
        print("DB keywords=", DB_keywords)

        #exhib_theme_list.append(keywords)

        # keyword 유사도 찾기
        similarity_weights = getKeywordSimilarity(word2vec_model, AI_keywords, DB_keywords)

        # 이미지 feature extraction (resnet50)
        if(file_dir_name and file_name is not None):
            # Read Image Files from DB
            filePath = "C:\\img\\" + file_dir_name + "\\" + file_uuid + "_" + file_name
            #print(exhib_seq, file_uuid + "_" + file_name)
            print("filePath=",filePath)

            # ResNet50 Feature Extraction
            feature = extract_feature(res50_model, filePath)
            
        # (exhib seq, 키워드 유사도, 추출된 특징(resnet)) 추가
        exhib_weights.append((exhib_seq, similarity_weights, feature))

    print('DB 전시회 파일 탐색 완료')

    print('전시회 추천 시작')
    exhib_recommendation_list = findExhibRecommendation(exhib_weights, rec_feat_list)
    print('전시회 추천 완료')

    global DB_weights
    DB_weights = exhib_recommendation_list

    return exhib_recommendation_list


app = Flask(__name__)  # Flask 객체 생성
app.config['JSON_AS_ASCII'] = False


@app.route('/updateModel')
def updateModel():
    rows = DBConnection()

    exhib_weights = []

    print('DB 전시회 파일 탐색 시작')
    ########################  DB 전시회 각 이미지 파일들 탐색 ############################
    for exhib_seq, exhib_theme, file_dir_name, file_uuid, file_name in rows:
        print(exhib_theme)
        #exhib_theme_list = []
        trans = Translator()

        # exhib_theme 키워드 split 하기
        result = trans.translate(exhib_theme, src='ko', dest='en')
        DB_keywords = result.text.split(', ')
        DB_keywords = [word.lower() for word in DB_keywords]
        print("DB keywords=", DB_keywords)

        # 이미지 feature extraction (resnet50)
        if(file_dir_name and file_name is not None):
            # Read Image Files from DB
            filePath = "C:\\img\\" + file_dir_name + "\\" + file_uuid + "_" + file_name
            #print(exhib_seq, file_uuid + "_" + file_name)
            print("filePath=", filePath)

            # ResNet50 Feature Extraction
            feature = extract_feature(res50_model, filePath)
            
        # (exhib seq, 키워드 유사도, 추출된 특징(resnet)) 추가
        exhib_weights.append((exhib_seq, DB_keywords, feature))
    print('DB 전시회 파일 탐색 완료')

    print(exhib_weights)

    global DB_weights
    DB_weights = exhib_weights

    #return jsonify(exhib_weights)
    return "<h1> Model Updated! </h1>"

@app.route('/DBTest')
def dbTest():
    rows = DBConnection()
    #print(rows)

    global DB_weights
    DB_weights = rows

    return jsonify(rows)

from flask import request

@app.route('/test/<idx>/<keywords>')
def test(idx, keywords):
    print(idx, keywords)

    return "done"
    #return DB_weights
    

@app.route('/AIrecommend/<AI_idx>/<AI_keywords>')
def recommend(AI_idx, AI_keywords):
    
    print('AI input 이미지 특징 추출 시작')
    # AI input 추천 이미지들의 feature list으로 반환
    AI_idx = AI_idx.split(', ')
    print('AI idx = ', AI_idx)

    AI_feat_list = inputImageFeatList(AI_idx)
    print('AI input 이미지 특징 추출 완료')

    print('AI keyword 바꿈 시작')
    # AI 키워드 한영 변환 및 lowercase으로 바꿈 ( Korean -> English )
    trans = Translator()
    result = trans.translate(AI_keywords, src='ko', dest='en')
    AI_keywords = result.text.split(', ')
    AI_keywords = [word.lower() for word in AI_keywords]
    print("AI keywords=", AI_keywords)
    print('AI keyword 바꿈 완료')

    exhib_recommendation_list = []
    for exhib_seq, exhib_keywords, exhib_img_features in DB_weights:

        # keyword 유사도 찾기
        sim_weights = getKeywordSimilarity(word2vec_model, AI_keywords, exhib_keywords)

        for AI_feat in AI_feat_list:
            dist = cosine_similarity(AI_feat, exhib_img_features)
            print("dist = ", dist, "||  mean(sim) = ", statistics.mean(sim_weights[:5])*0.3, "||  sum = ", dist+statistics.mean(sim_weights))
            exhib_recommendation_list.append((exhib_seq, dist+statistics.mean(sim_weights[:5])*0.3))


    exhib_recommendation_list = sorted(exhib_recommendation_list, key=lambda x:x[1], reverse=True)
    
    print(exhib_recommendation_list)

    temp = [rec[0] for rec in exhib_recommendation_list[:5]]
    print("==================== RESULT RECOMMENDATION LIST ===================== \n", temp)


    result = []
    for t in temp:
        result.append(t)

    print(type(jsonify(result)))

    return jsonify(result)


if __name__ == "__main__":  # 모듈이 실행 됨을 알림

    print('resnet50 모델 선언 시작')
    # resnet 모델 선언
    res50_model = models.resnet50(pretrained=True)
    res50_model.fc = nn.Identity()

    for param in res50_model.parameters():
        param.requires_grad = False

    with torch.no_grad():
        res50_model.eval()
    print('resnet50 모델 선언 완료')


    print('word2vec 모델 선언 시작')
    # word2vec 모델 선언
    word2vec_model = KeyedVectors.load("word2vec.model")
    print('word2vec 모델 선언 완료')

    updateModel()

    app.run(debug=True, port=5000)  # 서버 실행, 파라미터로 debug 여부, port 설정 가능