
#prepare_image
'''
对需要进行预测的图片进行预处理
'''
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
import numpy as np
from imutils import paths
from flask import Flask
from flask_restful import reqparse, Api, Resource
from PIL import Image

def prepare_image(image, model):
    # 加载图像
    # img = load_img(imagePath, target_size=(224, 224))        # x = np.array(img, dtype='float32')test
    # 图像预处理
    #image = image.resize((224, 224), Image.BILINEAR)
    x = img_to_array(image)
    x = x/255.0
    x = np.expand_dims(x, axis=0)
    results = model.predict(x)
    return results

#predict
'''
加载垃圾分类模型
'''

def predict_model():

    # create the base pre-trained model
    Inp = Input((224, 224, 3))
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model(Inp)
    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(9, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=Inp, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    model.load_weights("weights-improvement-110-0.89.h5")
    return model

#plot
'''
用于对预测结果图片进行显示
'''

# 根据预测结果显示对应的文字label
classes_types = ['battery', 'cardboard', 'ceramics', 'fruits', 'glass',  'metal', 'napkin', 'paper', 'plastic']


def generate_result(result):
    base = 0
    result_type = 20
    for i in range(9):
        if(result[0][i] >= base):
            base = result[0][i];
            result_type = i;
    print(result)
    return classes_types[result_type]


def show(results):
    return generate_result(results)


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def detect(image,model):
    results = prepare_image(image, model=model)
    return show(results=results)


def main(image):
    # imagePaths = list(paths.list_images("home/"))
     #sum=0
    model = predict_model()
    # for (i, imagePath) in enumerate(imagePaths):
    result = detect(image,model)
    #     print(result)
    #     if result in imagePath:
    #         sum=sum+1
    # print(sum)
    print(result)
    return result
from flask import Flask,render_template,request
# main(load_img("test2/battery01.jpg"))

# def to_do(image):
#     token = main(image)
#     return token
# Flask相关变量声明
from flask import request
app = Flask(__name__)
# api = Api(app)
# import argparse
# # RESTfulAPI的参数解析 -- put / post参数解析
# parser_put = reqparse.RequestParser()
#
# import werkzeug
# parser_put.add_argument('image', type=werkzeug.datastructures.FileStorage)
#
#
# import tarfile
# from werkzeug.utils import secure_filename
# import werkzeug
# from flask import Flask
# import tarfile
# from flask_restful import Resource, Api, reqparse
# from werkzeug.datastructures import FileStorage
# from werkzeug.utils import secure_filename
#
# class TodoList(Resource):
#     def post(self):
#         """
#         添加一个新用户: curl http://127.0.0.1:5000/users -X POST -d "name=Brown&age=20" -H "Authorization: token fejiasdfhu"
#         """
#         args = parser_put.parse_args()
#         content = args.get('image')
#         # content.save(os.path.join('/home', "image.jpg"))
#
#         # 构建新用户
#         info = {"info": to_do(args)}
#
#         # 资源添加成功，返回201
#         return info, 201
#
# # 设置路由
# api.add_resource(TodoList, "/trash")
basedir=os.path.abspath(os.path.dirname(__file__))
@app.route('/api/start' , methods = ['POST','GET'])
def api_predict():
    image=request.files.get('image')
    path=basedir+"/home/"
    file_path=path+image.filename
    image.save(file_path)
    img=load_img(file_path, target_size=(224, 224))
    result = main(img)
    return render_template('ok.html',result=result)



if __name__ == "__main__":
    app.run(debug=True)

