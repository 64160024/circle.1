from ultralytics import YOLO

# ในวงเล็บใส่ path ของโมเดล
model = YOLO('best1.pt')

# parameter ทั้งหมดจะเรียงดังนี้ 
#  (path ของรูปที่เราต้องการจะpredict,เซฟรูปที่ต้องการ predict,ค่าความมั่นใจที่จะทำนาย ถ้าเกิดทำนายออกมาเเล้วค่าความมั่นใจมากกว่าค่านี้ ถึงจะทำนายออกมา) 
           
model.predict('vdo2.MP4', save=True, conf=0.5)

