from flask import Flask, request, jsonify, render_template, Response
import os
import cv2



# Init app
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

camera = cv2.VideoCapture(0)

word = ""

def return_word():
    my_list = ["Fire fighter", "Cop", "Home", "Union", "Shirt"]
    for i in range(len(my_list)):
        yield my_list[i]

my_gen = return_word()

def generate_frames():
  while True:
    ## read the camera frame
    success,frame=camera.read()
    if not success:
        break
    else:
        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()

    yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Home route
@app.route('/', methods=['GET'])
def get_text():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/text')
def text():
  try:
    w = next(my_gen)
  except StopIteration:
    pass
  return jsonify({"text": w})
    


# Run Server
if __name__ == '__main__':
  app.run(debug=True)