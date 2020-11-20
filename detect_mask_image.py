# 이미지에서 얼굴을 검출하는 코드


# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # mobilent_v2 를 사용한다. 
# preprocess_input : 모델에 필요한 형식에 맞게 이미지를 변형
from tensorflow.keras.preprocessing.image import img_to_array # preprocessing.image : 이미지 데이터에서 실시간 데이터 확장을 위한 도구 세트
# img_to_array : PIL 이미지 인스턴스를 Numpy 배열로 변환합니다.
from tensorflow.keras.models import load_model # models : 훈련 및 추론 기능을 사용하여 도면층을 객체로 그룹화합니다.
# load_model : 저장된 모델을 로드합니다.
import numpy as np # numpy를 불러오고 'np' 로 칭한다.
import argparse # argparse를 불러온다  
# argparse 사용자 친화적인 명령행 인터페이스를 쉽게 작성하도록 합니다
import cv2 # opencv2를 불러온다.
import os # 운영체제를 제어할 수 있는 os를 불러온다.

def mask_image(): # 
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser() : #  ap라는 ArgumentParse객체 생성
 # ArgumentParser 객체는 명령행을 파이썬 데이터형으로 파싱하는데 필요한 모든 정보를 담고 있습니다.

	ap.add_argument("-i", "--image", required=True, 
		help="path to input image") # 사용할 이미지 인자를 추가한다
	ap.add_argument("-f", "--face", type=str, # 사용할 얼굼탐지기 인자를 추가한다
		default="face_detector",
		help="path to face detector model directory")
	ap.add_argument("-m", "--model", type=str, # 마스크 탐지 모델 인자를 추가한다.
		default="mask_detector.model",
		help="path to trained face mask detector model") # 사용할 탐색률 인자를 추가한다.
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())

	# load our serialized face detector model from disk
	print("[INFO] loading face detector model...") # 얼굴 탐지 모델을 로딩 
	prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"]) # 절대 주소로 얼굴 탐지 모델을 로딩
  # deploy.prototxt : 학습이 완료된 모델을 의미한다.
	weightsPath = os.path.sep.join([args["face"], # 절대 주소로  OpenCV의 딥러닝 페이스 검출기 로딩
		"res10_300x300_ssd_iter_140000.caffemodel"])
	net = cv2.dnn.readNet(prototxtPath, weightsPath) # 두 경로의 모델을 net에 저장

	# load the face mask detector model from disk
	print("[INFO] loading face mask detector model...") # 마스크 탐지 모델 로딩
	model = load_model(args["model"]) 

	# load the input image from disk, clone it, and grab the image spatial
	# dimensions
	image = cv2.imread(args["image"]) # 이미지 모델 로딩
	orig = image.copy()
	(h, w) = image.shape[:2]

	# construct a blob from the image
	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), # dnn 모듈이 사용하는 형태로 이미지를 변형
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	print("[INFO] computing face detections...") # 얼굴 검출
	net.setInput(blob) # 얼굴 검출 데이터를 dnn 모듈이 사용하는 형태로 이미지를 변형시킨다.
	detections = net.forward() # 검사결과를 저장

	# loop over the detections
	for i in range(0, detections.shape[2]): # 검사결과 확인
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2] # 탐색률에 검사 결과를 저장

    # 검사결과를 확보하여 취약한 탐지를 걸러냄
    # 최소 탐색률 이상

		if confidence > args["confidence"]: # 검사결과가 탐색률을 넘지 못한다면
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) # 검사결과에 이미지의 크기만큼의 박스를 생성한다.
			(startX, startY, endX, endY) = box.astype("int") # 박스의 크기 값을 가져온다

			# 생성된 박스가 이미지에 맞는지 확인
			# 이미지 박스
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = image[startY:endY, startX:endX] # 박스 크기만큼 이미지 생성
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) # BGR 채널을 RGB채널로 변경
			face = cv2.resize(face, (224, 224)) # 크기 조정
			face = img_to_array(face) # PIL 이미지 인스턴스를 Numpy 배열로 변환합니다.
			face = preprocess_input(face) # 수정한 값을 입력
			face = np.expand_dims(face, axis=0) # 새로운 차원을 추가한다.

			# 얼굴이 검출되었는가 안되었는 가
			# 마스크를 썼는가 안썼는 가
			(mask, withoutMask) = model.predict(face)[0] # 마스크를 썬느냐 마느냐는 face의 0번째 인자에 따라 결정

			# 마스크 착용 유무에 대한 라벨링
			# 박스에 텍스트를 추가한다
			label = "Mask" if mask > withoutMask else "No Mask" # 마스크를 썻는 가 안썻는 가 에 대한 라벨링
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

			# include the probability in the label
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100) # 라벨링 작업

			# 이미지에 검출결과 표시
			# 라벨링, 박스, 얼굴 검출
			cv2.putText(image, label, (startX, startY - 10), # 이미지에 라벨링 텍스르를 추가한다.
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2) # 이미지에 얼굴에 박스를 생성한다.

	# show the output image
	cv2.imshow("Output", image) #출력
	cv2.waitKey(0)
	
if __name__ == "__main__":
	mask_image()
